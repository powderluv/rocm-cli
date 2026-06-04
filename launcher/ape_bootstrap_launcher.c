/*
 * rocm-cli APE bootstrap launcher.
 *
 * This launcher reads the ZIP payload appended to its own executable, extracts
 * uncompressed entries to a per-user launcher directory, then delegates to the
 * extracted platform rocm binary. It intentionally supports only STORED ZIP
 * entries; compression belongs in the release archive, not in the bootstrap
 * launcher hot path.
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__COSMOPOLITAN__)
#include <cosmo.h>
#endif

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#include <process.h>
#define access _access
#define chmod _chmod
#define execv _execv
#ifndef X_OK
#define X_OK 0
#endif
#else
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef S_ISDIR
#define S_ISDIR(mode) (((mode) & S_IFMT) == S_IFDIR)
#endif

#ifndef ROCM_CLI_APE_VERSION
#define ROCM_CLI_APE_VERSION "dev"
#endif

#ifndef ROCM_CLI_APE_MODEL_PAYLOAD
#define ROCM_CLI_APE_MODEL_PAYLOAD "payload/bootstrap/Qwen3.5-0.8B-Q8_0.llamafile"
#endif

#ifndef ROCM_CLI_APE_BOOTSTRAP_PORT
#define ROCM_CLI_APE_BOOTSTRAP_PORT "11435"
#endif

#define ZIP_EOCD_SIG 0x06054b50u
#define ZIP64_EOCD_SIG 0x06064b50u
#define ZIP64_LOCATOR_SIG 0x07064b50u
#define ZIP_CENTRAL_SIG 0x02014b50u
#define ZIP_LOCAL_SIG 0x04034b50u
#define ZIP_METHOD_STORE 0u
#define ZIP_EOCD_MIN 22u
#define ZIP_EOCD_SEARCH 66000u

static int g_startup_ui = 0;
static int g_startup_ui_last_percent = -1;
static char g_startup_ui_last_label[96];

struct file_buffer {
  unsigned char *data;
  size_t size;
};

struct zip_payload {
  size_t base;
  size_t end;
  size_t central_dir_offset;
  size_t central_dir_size;
  size_t entries;
  uint32_t payload_crc32;
};

struct launcher_payload {
  char *path;
  struct file_buffer file;
  struct zip_payload zip;
};

static void startup_ui_begin(const char *root) {
  if (!g_startup_ui) {
    return;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  printf("\033[36mROCm CLI Setup Assistant\033[0m\n");
  printf("Getting your AMD GPU ready for local AI.\n");
  printf("First launch can take a little while while files unpack.\n\n");
  printf("Install data: %s\n\n", root);
}

static void startup_ui_status(const char *message) {
  if (!g_startup_ui) {
    return;
  }
  printf("\n\033[33m%s\033[0m\n", message);
}

static void startup_ui_progress(const char *label, size_t done, size_t total) {
  int percent;
  int width = 28;
  int filled;
  int i;
  if (!g_startup_ui) {
    return;
  }
  if (total == 0) {
    percent = 0;
  } else if (done >= total) {
    percent = 100;
  } else {
    percent = (int)(((unsigned long long)done * 100ull) / (unsigned long long)total);
  }
  if (percent == g_startup_ui_last_percent && strcmp(label, g_startup_ui_last_label) == 0) {
    return;
  }
  g_startup_ui_last_percent = percent;
  snprintf(g_startup_ui_last_label, sizeof(g_startup_ui_last_label), "%s", label);
  filled = (percent * width) / 100;
  printf("\r  [");
  for (i = 0; i < width; ++i) {
    putchar(i < filled ? '#' : '-');
  }
  printf("] %3d%%  %s", percent, label);
  if (percent >= 100) {
    putchar('\n');
    g_startup_ui_last_percent = -1;
    g_startup_ui_last_label[0] = '\0';
  }
  fflush(stdout);
}

static uint16_t read_le16(const unsigned char *p) {
  return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static uint32_t read_le32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static uint64_t read_le64(const unsigned char *p) {
  return (uint64_t)read_le32(p) | ((uint64_t)read_le32(p + 4) << 32);
}

static void fail_errno(const char *message) {
  fprintf(stderr, "rocm ape launcher: %s: %s\n", message, strerror(errno));
  exit(1);
}

static void fail_message(const char *message) {
  fprintf(stderr, "rocm ape launcher: %s\n", message);
  exit(1);
}

static size_t zip64_to_size(uint64_t value) {
  if (value > (uint64_t)SIZE_MAX) {
    fail_message("ZIP64 payload is too large for this platform");
  }
  return (size_t)value;
}

static void apply_zip64_extra(
    const unsigned char *extra,
    size_t extra_len,
    uint64_t *compressed_size,
    uint64_t *uncompressed_size,
    uint64_t *local_offset) {
  size_t cursor = 0;
  while (cursor + 4 <= extra_len) {
    uint16_t tag = read_le16(extra + cursor);
    uint16_t data_len = read_le16(extra + cursor + 2);
    const unsigned char *payload = extra + cursor + 4;
    size_t remaining = data_len;
    if (cursor + 4u + data_len > extra_len) {
      fail_message("ZIP64 extra field extends past entry");
    }
    if (tag == 0x0001u) {
      if (*uncompressed_size == UINT32_MAX) {
        if (remaining < 8) {
          fail_message("ZIP64 uncompressed size is missing");
        }
        *uncompressed_size = read_le64(payload);
        payload += 8;
        remaining -= 8;
      }
      if (*compressed_size == UINT32_MAX) {
        if (remaining < 8) {
          fail_message("ZIP64 compressed size is missing");
        }
        *compressed_size = read_le64(payload);
        payload += 8;
        remaining -= 8;
      }
      if (*local_offset == UINT32_MAX) {
        if (remaining < 8) {
          fail_message("ZIP64 local header offset is missing");
        }
        *local_offset = read_le64(payload);
      }
      return;
    }
    cursor += 4u + data_len;
  }
}

static uint32_t crc32_update(uint32_t crc, const unsigned char *data, size_t size) {
  size_t i;
  crc = ~crc;
  for (i = 0; i < size; ++i) {
    int bit;
    crc ^= data[i];
    for (bit = 0; bit < 8; ++bit) {
      uint32_t mask = 0u - (crc & 1u);
      crc = (crc >> 1) ^ (0xedb88320u & mask);
    }
  }
  return ~crc;
}

static uint32_t crc32_bytes(const unsigned char *data, size_t size) {
  return crc32_update(0u, data, size);
}

static int crc32_file(const char *path, uint32_t *out_crc, size_t *out_size) {
  unsigned char buffer[8192];
  size_t got;
  uint32_t crc = 0u;
  size_t total = 0u;
  FILE *file = fopen(path, "rb");
  if (!file) {
    return 0;
  }
  while ((got = fread(buffer, 1, sizeof(buffer), file)) > 0) {
    crc = crc32_update(crc, buffer, got);
    total += got;
  }
  if (ferror(file)) {
    fclose(file);
    return 0;
  }
  fclose(file);
  *out_crc = crc;
  *out_size = total;
  return 1;
}

static int has_prefix(const char *value, const char *prefix) {
  return strncmp(value, prefix, strlen(prefix)) == 0;
}

static int is_path_separator(char ch) {
  return ch == '/' || ch == '\\';
}

static int runtime_is_windows(void) {
#if defined(__COSMOPOLITAN__)
  return IsWindows();
#elif defined(_WIN32)
  return 1;
#else
  return 0;
#endif
}

static const char *platform_id(void) {
  return runtime_is_windows() ? "windows-amd64" : "linux-amd64";
}

static const char *rocm_binary_name(void) {
  return runtime_is_windows() ? "rocm.exe" : "rocm";
}

static int is_drive_path(const char *path) {
  return path && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) &&
         path[1] == ':' && is_path_separator(path[2]);
}

static int is_cosmo_drive_root(const char *path) {
  return path && path[0] == '/' &&
         ((path[1] >= 'A' && path[1] <= 'Z') || (path[1] >= 'a' && path[1] <= 'z')) &&
         is_path_separator(path[2]);
}

static char *xstrdup(const char *value) {
  size_t length = strlen(value);
  char *copy = (char *)malloc(length + 1);
  if (!copy) {
    fail_errno("out of memory");
  }
  memcpy(copy, value, length + 1);
  return copy;
}

static char *normalize_runtime_path(const char *path) {
#if defined(__COSMOPOLITAN__)
  if (runtime_is_windows() && is_drive_path(path)) {
    size_t length = strlen(path);
    size_t i;
    char *normalized = (char *)malloc(length + 1);
    if (!normalized) {
      fail_errno("out of memory normalizing path");
    }
    normalized[0] = '/';
    normalized[1] = path[0];
    for (i = 2; i < length; ++i) {
      normalized[i] = is_path_separator(path[i]) ? '/' : path[i];
    }
    normalized[length] = '\0';
    return normalized;
  }
#endif
  return xstrdup(path);
}

static char *native_child_path(const char *path) {
  if (runtime_is_windows() && is_cosmo_drive_root(path)) {
    size_t length = strlen(path);
    size_t i;
    size_t out_index = 2;
    char *native = (char *)malloc(length + 1);
    if (!native) {
      fail_errno("out of memory normalizing child path");
    }
    native[0] = path[1];
    native[1] = ':';
    for (i = 2; i < length; ++i) {
      native[out_index++] = is_path_separator(path[i]) ? '\\' : path[i];
    }
    native[out_index] = '\0';
    return native;
  }
  return xstrdup(path);
}

static char *join_path2(const char *left, const char *right) {
  size_t left_len = strlen(left);
  size_t right_len = strlen(right);
  int needs_sep = left_len > 0 && !is_path_separator(left[left_len - 1]);
  char *path = (char *)malloc(left_len + (needs_sep ? 1 : 0) + right_len + 1);
  if (!path) {
    fail_errno("out of memory");
  }
  memcpy(path, left, left_len);
  if (needs_sep) {
    path[left_len++] = '/';
  }
  memcpy(path + left_len, right, right_len);
  path[left_len + right_len] = '\0';
  return path;
}

static char *default_root(void) {
  const char *configured = getenv("ROCM_CLI_APE_ROOT");
  if (configured && configured[0]) {
    return normalize_runtime_path(configured);
  }
  const char *home = getenv("HOME");
#if defined(_WIN32)
  if (!home || !home[0]) {
    home = getenv("USERPROFILE");
  }
#endif
  if (!home || !home[0]) {
    fail_message("unable to determine home directory; set ROCM_CLI_APE_ROOT");
  }
  char *a = join_path2(home, ".rocm");
  char *b = join_path2(a, "launcher");
  char *c = join_path2(b, ROCM_CLI_APE_VERSION);
  char *d = join_path2(c, platform_id());
  free(a);
  free(b);
  free(c);
  char *normalized = normalize_runtime_path(d);
  free(d);
  return normalized;
}

static char *self_path(const char *argv0) {
#if defined(__linux__)
  char proc_path[PATH_MAX];
  ssize_t got = readlink("/proc/self/exe", proc_path, sizeof(proc_path) - 1);
  if (got > 0) {
    proc_path[got] = '\0';
    return xstrdup(proc_path);
  }
#endif
  if (argv0 && argv0[0]) {
    return xstrdup(argv0);
  }
  fail_message("unable to determine launcher executable path");
  return NULL;
}

static int read_entire_file_optional(const char *path, struct file_buffer *buffer) {
  FILE *file;
  long size;
  size_t got;
  buffer->data = NULL;
  buffer->size = 0;
  if (!path || !path[0]) {
    return 0;
  }
  file = fopen(path, "rb");
  if (!file) {
    return 0;
  }
  if (fseek(file, 0, SEEK_END) != 0) {
    fail_errno("failed to seek launcher executable");
  }
  size = ftell(file);
  if (size < 0) {
    fail_errno("failed to measure launcher executable");
  }
  if (fseek(file, 0, SEEK_SET) != 0) {
    fail_errno("failed to rewind launcher executable");
  }
  buffer->data = (unsigned char *)malloc((size_t)size);
  if (!buffer->data && size > 0) {
    fail_errno("out of memory reading launcher");
  }
  got = 0;
  while (got < (size_t)size) {
    size_t want = (size_t)size - got;
    size_t chunk;
    if (want > 8u * 1024u * 1024u) {
      want = 8u * 1024u * 1024u;
    }
    chunk = fread(buffer->data + got, 1, want, file);
    if (chunk == 0) {
      fail_errno("failed to read launcher executable");
    }
    got += chunk;
    startup_ui_progress("Loading embedded package", got, (size_t)size);
  }
  fclose(file);
  buffer->size = (size_t)size;
  return 1;
}

static struct file_buffer read_entire_file(const char *path) {
  struct file_buffer buffer;
  if (!read_entire_file_optional(path, &buffer)) {
    fail_errno("failed to open launcher executable");
  }
  return buffer;
}

static struct zip_payload empty_zip_payload(void) {
  return (struct zip_payload){0, 0, 0, 0, 0};
}

static struct zip_payload find_zip_payload(const unsigned char *data, size_t size) {
  size_t start;
  size_t i;
  if (size < ZIP_EOCD_MIN) {
    return empty_zip_payload();
  }
  start = size > ZIP_EOCD_SEARCH ? size - ZIP_EOCD_SEARCH : 0;
  for (i = size - ZIP_EOCD_MIN + 1; i-- > start;) {
    if (read_le32(data + i) == ZIP_EOCD_SIG) {
      uint16_t disk = read_le16(data + i + 4);
      uint16_t cd_disk = read_le16(data + i + 6);
      uint16_t entries_disk = read_le16(data + i + 8);
      uint16_t entries = read_le16(data + i + 10);
      uint32_t cd_size = read_le32(data + i + 12);
      uint32_t cd_offset = read_le32(data + i + 16);
      uint16_t comment_len = read_le16(data + i + 20);
      struct zip_payload payload;
      size_t end = i + ZIP_EOCD_MIN + comment_len;
      uint64_t cd_size64 = cd_size;
      uint64_t cd_offset64 = cd_offset;
      uint64_t entries64 = entries;
      size_t payload_base;
      if (end > size) {
        continue;
      }
      if (disk != 0 || cd_disk != 0 || entries_disk != entries) {
        fail_message("multi-disk ZIP payloads are not supported");
      }
      if (entries == 0 || cd_size == 0) {
        fail_message("ZIP payload is empty");
      }
      if (i >= 20 && read_le32(data + i - 20) == ZIP64_LOCATOR_SIG) {
        uint64_t zip64_offset = read_le64(data + i - 20 + 8);
        size_t zip64_scan_start = i > ZIP_EOCD_SEARCH ? i - ZIP_EOCD_SEARCH : 0;
        size_t zip64_abs = 0;
        size_t z;
        for (z = i - 20; z-- > zip64_scan_start;) {
          if (read_le32(data + z) == ZIP64_EOCD_SIG) {
            uint64_t record_size = read_le64(data + z + 4);
            if (z + zip64_to_size(record_size) + 12 == i - 20) {
              zip64_abs = z;
              break;
            }
          }
          if (z == 0) {
            break;
          }
        }
        if (zip64_abs == 0) {
          fail_message("ZIP64 locator is present but ZIP64 EOCD was not found");
        }
        disk = (uint16_t)read_le32(data + zip64_abs + 16);
        cd_disk = (uint16_t)read_le32(data + zip64_abs + 20);
        if (disk != 0 || cd_disk != 0) {
          fail_message("multi-disk ZIP64 payloads are not supported");
        }
        entries64 = read_le64(data + zip64_abs + 32);
        cd_size64 = read_le64(data + zip64_abs + 40);
        cd_offset64 = read_le64(data + zip64_abs + 48);
        if (zip64_offset > (uint64_t)zip64_abs) {
          fail_message("ZIP64 payload offset is invalid");
        }
        payload_base = zip64_abs - zip64_to_size(zip64_offset);
      } else {
        if ((uint64_t)cd_offset + (uint64_t)cd_size > (uint64_t)i) {
          fail_message("ZIP payload central directory is invalid");
        }
        payload_base = i - (size_t)cd_offset - (size_t)cd_size;
      }
      if (entries64 == 0 || entries64 > SIZE_MAX) {
        fail_message("ZIP payload entry count is invalid");
      }
      if (cd_offset64 + cd_size64 > (uint64_t)(i - payload_base)) {
        fail_message("ZIP payload central directory is invalid");
      }
      payload.base = payload_base;
      payload.end = end;
      payload.central_dir_offset = payload.base + zip64_to_size(cd_offset64);
      payload.central_dir_size = zip64_to_size(cd_size64);
      payload.entries = zip64_to_size(entries64);
      payload.payload_crc32 = crc32_bytes(data + payload.base, payload.end - payload.base);
      return payload;
    }
    if (i == 0) {
      break;
    }
  }
  return empty_zip_payload();
}

static struct launcher_payload read_launcher_payload(const char *argv0) {
  const char *candidates[5];
  char proc_path[PATH_MAX];
  const char *configured_self;
  int candidate_count = 0;
  int i;
  candidates[candidate_count++] = argv0;
  configured_self = getenv("ROCM_CLI_APE_SELF");
  if (configured_self && configured_self[0]) {
    candidates[candidate_count++] = configured_self;
  }
#if defined(__COSMOPOLITAN__)
  {
    char *program = GetProgramExecutableName();
    if (program && program[0]) {
      candidates[candidate_count++] = program;
    }
  }
#endif
#if defined(__linux__)
  {
    ssize_t got = readlink("/proc/self/exe", proc_path, sizeof(proc_path) - 1);
    if (got > 0) {
      proc_path[got] = '\0';
      candidates[candidate_count++] = proc_path;
    }
  }
#endif
  for (i = 0; i < candidate_count; ++i) {
    int duplicate = 0;
    int earlier;
    struct file_buffer file;
    struct zip_payload zip;
    if (!candidates[i] || !candidates[i][0]) {
      continue;
    }
    for (earlier = 0; earlier < i; ++earlier) {
      if (candidates[earlier] && strcmp(candidates[i], candidates[earlier]) == 0) {
        duplicate = 1;
      }
    }
    if (duplicate) {
      continue;
    }
    if (!read_entire_file_optional(candidates[i], &file)) {
      continue;
    }
    zip = find_zip_payload(file.data, file.size);
    if (zip.base != 0) {
      struct launcher_payload payload;
      payload.path = xstrdup(candidates[i]);
      payload.file = file;
      payload.zip = zip;
      return payload;
    }
    free(file.data);
  }
  fail_message("unable to locate appended APE payload in launcher executable");
  return (struct launcher_payload){NULL, {NULL, 0}, {0, 0, 0, 0, 0}};
}

static int safe_payload_name(const char *name) {
  const char *p;
  int at_component_start = 1;
  int component_len = 0;
  if (!name || !name[0]) {
    return 0;
  }
  if (name[0] == '/' || name[0] == '\\' || strchr(name, '\\')) {
    return 0;
  }
  if (!has_prefix(name, "payload/")) {
    return 0;
  }
  for (p = name; *p; ++p) {
    if (*p == ':') {
      return 0;
    }
    if (*p == '/') {
      if (component_len == 0) {
        return 0;
      }
      at_component_start = 1;
      component_len = 0;
      continue;
    }
    if (at_component_start && *p == '.') {
      if (p[1] == '/' || p[1] == '\0') {
        return 0;
      }
      if (p[1] == '.' && (p[2] == '/' || p[2] == '\0')) {
        return 0;
      }
    }
    at_component_start = 0;
    ++component_len;
  }
  return component_len > 0;
}

static void mkdir_one(const char *path) {
  struct stat st;
  if (!path[0]) {
    return;
  }
#if defined(_WIN32)
  if (_mkdir(path) != 0 && errno != EEXIST) {
#else
  if (mkdir(path, 0755) != 0 && errno != EEXIST) {
#endif
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
      return;
    }
    fail_errno("failed to create extraction directory");
  }
}

static void mkdir_path(const char *path) {
  char *copy = xstrdup(path);
  char *p;
  p = copy + 1;
#if defined(_WIN32)
  if (((copy[0] >= 'A' && copy[0] <= 'Z') || (copy[0] >= 'a' && copy[0] <= 'z')) && copy[1] == ':') {
    p = copy + 2;
    if (is_path_separator(*p)) {
      ++p;
    }
  }
#endif
  if (is_cosmo_drive_root(copy)) {
    p = copy + 3;
  }
  for (; *p; ++p) {
    if (*p == '/' || *p == '\\') {
      char saved = *p;
      *p = '\0';
      mkdir_one(copy);
      *p = saved;
    }
  }
  mkdir_one(copy);
  free(copy);
}

static void mkdir_parents_for_file(const char *path) {
  char *copy = xstrdup(path);
  char *p;
  p = copy + 1;
#if defined(_WIN32)
  if (((copy[0] >= 'A' && copy[0] <= 'Z') || (copy[0] >= 'a' && copy[0] <= 'z')) && copy[1] == ':') {
    p = copy + 2;
    if (is_path_separator(*p)) {
      ++p;
    }
  }
#endif
  if (is_cosmo_drive_root(copy)) {
    p = copy + 3;
  }
  for (; *p; ++p) {
    if (*p == '/' || *p == '\\') {
      char saved = *p;
      *p = '\0';
      mkdir_one(copy);
      *p = saved;
    }
  }
  free(copy);
}

static void write_file_bytes(const char *path, const unsigned char *data, size_t size, uint32_t mode) {
  FILE *file;
  mkdir_parents_for_file(path);
  file = fopen(path, "wb");
  if (!file) {
    fail_errno("failed to create extracted file");
  }
  if (size > 0 && fwrite(data, 1, size, file) != size) {
    fail_errno("failed to write extracted file");
  }
  if (fclose(file) != 0) {
    fail_errno("failed to close extracted file");
  }
#if !defined(_WIN32)
  if ((mode & 0111u) != 0) {
    chmod(path, mode & 0777u);
  }
#else
  (void)mode;
#endif
}

static char *activation_marker_path(const char *root) {
  return join_path2(root, ".rocm-ape-launcher");
}

static void write_activation_marker(const char *root, struct zip_payload payload) {
  char *path = activation_marker_path(root);
  FILE *file = fopen(path, "wb");
  if (!file) {
    free(path);
    fail_errno("failed to write activation marker");
  }
  fprintf(
      file,
      "rocm-cli-ape-bootstrap/v1\nversion=%s\nplatform=%s\npayload_crc32=%08x\nentries=%zu\n",
      ROCM_CLI_APE_VERSION,
      platform_id(),
      payload.payload_crc32,
      payload.entries);
  if (fclose(file) != 0) {
    free(path);
    fail_errno("failed to close activation marker");
  }
  free(path);
}

static int activation_marker_matches(const char *root, struct zip_payload payload) {
  char *path = activation_marker_path(root);
  char expected[256];
  char actual[256];
  FILE *file;
  int ok = 0;
  snprintf(
      expected,
      sizeof(expected),
      "rocm-cli-ape-bootstrap/v1\nversion=%s\nplatform=%s\npayload_crc32=%08x\nentries=%zu\n",
      ROCM_CLI_APE_VERSION,
      platform_id(),
      payload.payload_crc32,
      payload.entries);
  file = fopen(path, "rb");
  free(path);
  if (!file) {
    return 0;
  }
  memset(actual, 0, sizeof(actual));
  if (fread(actual, 1, sizeof(actual) - 1, file) > 0) {
    ok = strcmp(actual, expected) == 0;
  }
  fclose(file);
  return ok;
}

static int extracted_payload_matches(const unsigned char *data, size_t size, struct zip_payload payload, const char *root) {
  size_t cursor = payload.central_dir_offset;
  size_t entry_index;
  for (entry_index = 0; entry_index < payload.entries; ++entry_index) {
    uint16_t method;
    uint32_t expected_crc;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint64_t local_offset_unused = 0;
    uint16_t name_len;
    uint16_t extra_len;
    uint16_t comment_len;
    char *name;
    char *target;
    uint32_t actual_crc;
    size_t actual_size;
    if (cursor + 46 > size || read_le32(data + cursor) != ZIP_CENTRAL_SIG) {
      return 0;
    }
    method = read_le16(data + cursor + 10);
    expected_crc = read_le32(data + cursor + 16);
    compressed_size = read_le32(data + cursor + 20);
    uncompressed_size = read_le32(data + cursor + 24);
    name_len = read_le16(data + cursor + 28);
    extra_len = read_le16(data + cursor + 30);
    comment_len = read_le16(data + cursor + 32);
    if (cursor + 46u + name_len + extra_len + comment_len > size || name_len == 0) {
      return 0;
    }
    name = (char *)malloc((size_t)name_len + 1);
    if (!name) {
      fail_errno("out of memory reading ZIP filename");
    }
    memcpy(name, data + cursor + 46, name_len);
    name[name_len] = '\0';
    apply_zip64_extra(
        data + cursor + 46u + name_len,
        extra_len,
        &compressed_size,
        &uncompressed_size,
        &local_offset_unused);
    cursor += 46u + name_len + extra_len + comment_len;
    if (name[name_len - 1] == '/') {
      free(name);
      continue;
    }
    if (!safe_payload_name(name) || method != ZIP_METHOD_STORE || compressed_size != uncompressed_size) {
      free(name);
      return 0;
    }
    target = join_path2(root, name);
    if (!crc32_file(target, &actual_crc, &actual_size)) {
      free(target);
      free(name);
      return 0;
    }
    free(target);
    free(name);
    if (actual_size != zip64_to_size(uncompressed_size) || actual_crc != expected_crc) {
      return 0;
    }
    startup_ui_progress("Checking unpacked files", entry_index + 1, payload.entries);
  }
  return cursor == payload.central_dir_offset + payload.central_dir_size;
}

static void extract_zip_payload(const unsigned char *data, size_t size, struct zip_payload payload, const char *root) {
  size_t cursor = payload.central_dir_offset;
  size_t entry_index;
  mkdir_path(root);
  for (entry_index = 0; entry_index < payload.entries; ++entry_index) {
    uint16_t method;
    uint32_t expected_crc;
    uint64_t compressed_size;
    uint64_t uncompressed_size;
    uint16_t name_len;
    uint16_t extra_len;
    uint16_t comment_len;
    uint32_t external_attr;
    uint64_t local_offset;
    char *name;
    char *target;
    size_t local_abs;
    size_t data_abs;
    uint16_t local_name_len;
    uint16_t local_extra_len;
    if (cursor + 46 > size || read_le32(data + cursor) != ZIP_CENTRAL_SIG) {
      fail_message("ZIP central directory is corrupt");
    }
    method = read_le16(data + cursor + 10);
    expected_crc = read_le32(data + cursor + 16);
    compressed_size = read_le32(data + cursor + 20);
    uncompressed_size = read_le32(data + cursor + 24);
    name_len = read_le16(data + cursor + 28);
    extra_len = read_le16(data + cursor + 30);
    comment_len = read_le16(data + cursor + 32);
    external_attr = read_le32(data + cursor + 38) >> 16;
    local_offset = read_le32(data + cursor + 42);
    if (cursor + 46u + name_len + extra_len + comment_len > size) {
      fail_message("ZIP central directory entry extends past file");
    }
    name = (char *)malloc((size_t)name_len + 1);
    if (!name) {
      fail_errno("out of memory reading ZIP filename");
    }
    memcpy(name, data + cursor + 46, name_len);
    name[name_len] = '\0';
    apply_zip64_extra(
        data + cursor + 46u + name_len,
        extra_len,
        &compressed_size,
        &uncompressed_size,
        &local_offset);
    cursor += 46u + name_len + extra_len + comment_len;

    if (name_len == 0) {
      free(name);
      fail_message("ZIP payload contains an empty filename");
    }
    if (!safe_payload_name(name)) {
      free(name);
      fail_message("ZIP payload contains an unsafe path");
    }
    if (name[name_len - 1] == '/') {
      free(name);
      continue;
    }
    if (method != ZIP_METHOD_STORE) {
      free(name);
      fail_message("ZIP payload contains a compressed entry; expected STORED");
    }
    if (compressed_size != uncompressed_size) {
      free(name);
      fail_message("ZIP payload has mismatched stored sizes");
    }
    local_abs = payload.base + zip64_to_size(local_offset);
    if (local_abs + 30 > size || read_le32(data + local_abs) != ZIP_LOCAL_SIG) {
      fail_message("ZIP local file header is corrupt");
    }
    local_name_len = read_le16(data + local_abs + 26);
    local_extra_len = read_le16(data + local_abs + 28);
    data_abs = local_abs + 30u + local_name_len + local_extra_len;
    if (data_abs + zip64_to_size(compressed_size) > size) {
      free(name);
      fail_message("ZIP file data extends past launcher");
    }
    if (crc32_bytes(data + data_abs, zip64_to_size(compressed_size)) != expected_crc) {
      free(name);
      fail_message("ZIP payload entry failed CRC verification");
    }
    target = join_path2(root, name);
    write_file_bytes(target, data + data_abs, zip64_to_size(compressed_size), external_attr);
    free(target);
    free(name);
    startup_ui_progress("Unpacking ROCm CLI", entry_index + 1, payload.entries);
  }
  write_activation_marker(root, payload);
}

static char *platform_rocm_path(const char *root) {
  char relative[PATH_MAX];
  snprintf(relative, sizeof(relative), "payload/platform/%s/bin/%s", platform_id(), rocm_binary_name());
  return join_path2(root, relative);
}

static char *model_path(const char *root) {
  return join_path2(root, ROCM_CLI_APE_MODEL_PAYLOAD);
}

static int run_rocm(char *rocm, int argc, char **argv, int start_index) {
  int forwarded = argc - start_index;
  char **child_argv = (char **)calloc((size_t)forwarded + 2, sizeof(char *));
  char *native_rocm = native_child_path(rocm);
  int i;
  if (!child_argv) {
    fail_errno("out of memory preparing rocm argv");
  }
  child_argv[0] = native_rocm;
  for (i = 0; i < forwarded; ++i) {
    child_argv[i + 1] = argv[start_index + i];
  }
  child_argv[forwarded + 1] = NULL;
  execv(native_rocm, child_argv);
  fprintf(stderr, "rocm ape launcher: failed to run %s: %s\n", native_rocm, strerror(errno));
  free(native_rocm);
  free(child_argv);
  return 127;
}

static int run_bootstrap(char *rocm, const char *root) {
  char *llamafile = model_path(root);
  char *native_rocm = native_child_path(rocm);
  char *native_llamafile = native_child_path(llamafile);
  char *child_argv[12];
  child_argv[0] = native_rocm;
  child_argv[1] = "bootstrap";
  child_argv[2] = "assistant";
  child_argv[3] = "--llamafile";
  child_argv[4] = native_llamafile;
  child_argv[5] = "--host";
  child_argv[6] = "127.0.0.1";
  child_argv[7] = "--port";
  child_argv[8] = ROCM_CLI_APE_BOOTSTRAP_PORT;
  child_argv[9] = "--device";
  child_argv[10] = "gpu_required";
  child_argv[11] = NULL;
  execv(native_rocm, child_argv);
  fprintf(stderr, "rocm ape launcher: failed to run %s: %s\n", native_rocm, strerror(errno));
  free(native_rocm);
  free(native_llamafile);
  free(llamafile);
  return 127;
}

int main(int argc, char **argv) {
  struct launcher_payload launcher;
  char *root;
  char *rocm;
  int arg_start = 1;
  int result;
  int extract_only = argc > 1 && strcmp(argv[1], "--ape-extract-only") == 0;
  int payload_ready;

  g_startup_ui = argc == 1;
  root = default_root();
  if (g_startup_ui) {
    startup_ui_begin(root);
  }

  launcher = read_launcher_payload(argv[0]);
  if (!extract_only) {
    startup_ui_status("Checking the embedded ROCm CLI files.");
  }
  payload_ready =
      activation_marker_matches(root, launcher.zip) &&
      extracted_payload_matches(launcher.file.data, launcher.file.size, launcher.zip, root);
  if (!payload_ready) {
    startup_ui_status("Preparing ROCm CLI for first launch.");
    extract_zip_payload(launcher.file.data, launcher.file.size, launcher.zip, root);
  } else {
    startup_ui_status("ROCm CLI files are ready.");
  }
  free(launcher.file.data);
  free(launcher.path);

  if (extract_only) {
    printf("%s\n", root);
    free(root);
    return 0;
  }
  if (argc > 1 && strcmp(argv[1], "--") == 0) {
    arg_start = 2;
  }
  rocm = platform_rocm_path(root);
  if (access(rocm, X_OK) != 0) {
    fprintf(stderr, "rocm ape launcher: extracted rocm binary is not executable: %s\n", rocm);
    free(rocm);
    free(root);
    return 1;
  }
  if (arg_start >= argc) {
    startup_ui_status("Starting the embedded Qwen assistant on your AMD GPU.");
    result = run_bootstrap(rocm, root);
  } else {
    result = run_rocm(rocm, argc, argv, arg_start);
  }
  free(rocm);
  free(root);
  return result;
}
