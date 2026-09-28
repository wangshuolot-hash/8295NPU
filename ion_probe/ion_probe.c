// ION heap probe for SA8295P GVM (msm-5.4 legacy ion)
// Enumerates available heaps via ION_IOC_HEAP_QUERY and per-bit ION_IOC_ALLOC sweep
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <sys/ioctl.h>

// msm-5.4 vendor ion interface
struct ion_allocation_data {
    uint64_t len;
    uint32_t heap_id_mask;
    uint32_t flags;
    int32_t  fd;
    uint32_t unused;
};

struct ion_heap_data {
    char name[64];
    uint32_t type;
    uint32_t heap_id;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

struct ion_heap_query {
    uint32_t cnt;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
    struct ion_heap_data heaps[64];
};

#define ION_IOC_MAGIC 'I'
#define ION_IOC_ALLOC      _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data)
#define ION_IOC_HEAP_QUERY _IOWR(ION_IOC_MAGIC, 5, struct ion_heap_query)

// old AOSP ion (pre-5.4) layout, in case the shim accepts it
struct ion_allocation_data_old {
    size_t len;
    size_t align;
    uint32_t heap_id_mask;
    uint32_t flags;
    int fd_handle;
};
#define ION_IOC_ALLOC_OLD _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data_old)

int main(void) {
    int fd = open("/dev/ion", O_RDONLY | O_DSYNC);
    if (fd < 0) { perror("open /dev/ion"); return 1; }
    printf("ion fd=%d\n", fd);

    // 1. heap query
    struct ion_heap_query q;
    memset(&q, 0, sizeof(q));
    q.cnt = 0;
    if (ioctl(fd, ION_IOC_HEAP_QUERY, &q) == 0) {
        printf("HEAP_QUERY: cnt=%u\n", q.cnt);
        for (uint32_t i = 0; i < q.cnt && i < 64; i++) {
            printf("  heap[%u]: name=%s type=%u heap_id=%u (bit 0x%x)\n",
                   i, q.heaps[i].name, q.heaps[i].type, q.heaps[i].heap_id,
                   q.heaps[i].heap_id);
        }
    } else {
        printf("HEAP_QUERY failed: %s\n", strerror(errno));
    }

    // 2. per-bit alloc sweep (new interface)
    printf("--- alloc sweep (new struct) ---\n");
    for (int bit = 0; bit < 32; bit++) {
        struct ion_allocation_data a;
        memset(&a, 0, sizeof(a));
        a.len = 4096;
        a.heap_id_mask = 1u << bit;
        a.flags = 0;
        int r = ioctl(fd, ION_IOC_ALLOC, &a);
        if (r == 0 && a.fd > 0) {
            printf("bit %2d (0x%08x): OK fd=%d\n", bit, 1u << bit, a.fd);
            close(a.fd);
        } else {
            printf("bit %2d (0x%08x): FAIL errno=%d (%s)\n", bit, 1u << bit, errno, strerror(errno));
        }
    }

    // 3. per-bit sweep with CACHED flag
    printf("--- alloc sweep (flags=1 cached) ---\n");
    for (int bit = 0; bit < 32; bit++) {
        struct ion_allocation_data a;
        memset(&a, 0, sizeof(a));
        a.len = 4096;
        a.heap_id_mask = 1u << bit;
        a.flags = 1;
        int r = ioctl(fd, ION_IOC_ALLOC, &a);
        if (r == 0 && a.fd > 0) {
            printf("bit %2d (0x%08x): OK fd=%d\n", bit, 1u << bit, a.fd);
            close(a.fd);
        }
    }

    // 4. size sweep on system heap (bit 25) — rpcmem failed at 8 bytes
    printf("--- size sweep on system heap 0x2000000, flags=1 ---\n");
    {
        static const uint64_t sizes[] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65536, 262144, 1048576, 16777216 };
        for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            struct ion_allocation_data a;
            memset(&a, 0, sizeof(a));
            a.len = sizes[i];
            a.heap_id_mask = 0x2000000;
            a.flags = 1;
            int r = ioctl(fd, ION_IOC_ALLOC, &a);
            if (r == 0 && a.fd > 0) {
                printf("len %10llu: OK\n", (unsigned long long) sizes[i]);
                close(a.fd);
            } else {
                printf("len %10llu: FAIL errno=%d (%s)\n", (unsigned long long) sizes[i], errno, strerror(errno));
            }
        }
    }

    // 5. try old struct ioctl number for reference
    printf("--- old struct single test (bit 25) ---\n");
    struct ion_allocation_data_old ao;
    memset(&ao, 0, sizeof(ao));
    ao.len = 4096;
    ao.align = 4096;
    ao.heap_id_mask = 1u << 25;
    int r = ioctl(fd, ION_IOC_ALLOC_OLD, &ao);
    printf("old-struct alloc bit25: r=%d errno=%d (%s)\n", r, errno, strerror(errno));

    close(fd);
    return 0;
}
