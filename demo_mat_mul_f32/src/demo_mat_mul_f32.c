/***************************************************************************
 *  Copyright (C) 2018-2023 Andes Technology Corporation                   *
 *  All rights reserved.                                                   *
 ***************************************************************************/
/*
 * In this demonstration, the program multiplies a ROW * COL
 * matrix with a COL * COL2 matrix then stores the multiplication
 * results in a ROW * COL2 matrix.
 */

#include "share.h"
#include "mat_mul_f32.h"

#define COL 128
#define ROW COL
#define MAT_SIZE (ROW * COL)
#define COL2 COL
#define MAT2_SIZE (COL * COL2)
#define MAT_MUL_OUT (ROW * COL2)

int main(void)
{
    float mat_src_f32_1[MAT_SIZE] __attribute__ ((aligned(64)));
    float mat_src_f32_2[MAT2_SIZE] __attribute__ ((aligned(64)));
    float mat_out_f32[MAT_MUL_OUT] __attribute__ ((aligned(64)));
    float mat_golden_f32[MAT_MUL_OUT] __attribute__ ((aligned(64)));
    FILE *f1, *f2, *fgolden;
    uint64_t cycle_c = 0, cycle_v;
#if defined(__linux) || defined(__linux__) || defined(linux)
    int fd_inst;
    struct perf_event_attr pe_inst;
    memset(&pe_inst, 0, sizeof(pe_inst));
    pe_inst.type = PERF_TYPE_HARDWARE;
    pe_inst.size = sizeof(pe_inst);
    pe_inst.config = PERF_COUNT_HW_INSTRUCTIONS;
    pe_inst.disabled = 1;
    pe_inst.exclude_kernel = 1;
    pe_inst.exclude_hv = 1;
    // pid == 0 and cpu == -1
    // This measures the calling process/thread on any CPU.
    fd_inst = perf_event_open(&pe_inst, 0, -1, -1, 0);
    if (fd_inst == -1) {
        fprintf(stderr, "Error opening leader %llx\n", pe_inst.config);
        exit(EXIT_FAILURE);
    }

    int fd_cycle;
    struct perf_event_attr pe_cycle;
    memset(&pe_cycle, 0, sizeof(pe_cycle));
    pe_cycle.type = PERF_TYPE_HARDWARE;
    pe_cycle.size = sizeof(pe_cycle);
    pe_cycle.config = PERF_COUNT_HW_CPU_CYCLES;
    pe_cycle.disabled = 1;
    pe_cycle.exclude_kernel = 1;
    pe_cycle.exclude_hv = 1;
    // pid == 0 and cpu == -1
    // This measures the calling process/thread on any CPU.
    fd_cycle = perf_event_open(&pe_cycle, 0, -1, -1, 0);
    if (fd_cycle == -1) {
        fprintf(stderr, "Error opening leader %llx\n", pe_cycle.config);
        exit(EXIT_FAILURE);
    }
#endif
    PRINT_VER;

    //initialize vector CSR
#if defined(__linux) || defined(__linux__) || defined(linux)
#else
    ENA_FS_VS;
#endif

    //read input data
    test_Fopen("bin/in1.bin", "rb", &f1);
    test_Fopen("bin/in2.bin", "rb", &f2);
    test_Fopen("bin/golden.bin", "rb", &fgolden);
    test_Fread(mat_src_f32_1, sizeof(*mat_src_f32_1), MAT_SIZE, f1, "mat_src_f32_1");
    test_Fread(mat_src_f32_2, sizeof(*mat_src_f32_2), MAT2_SIZE, f2, "mat_src_f32_2");
    test_Fread(mat_golden_f32, sizeof(*mat_golden_f32), MAT_SIZE, fgolden, "mat_golden_f32");
    fclose(f1);
    fclose(f2);
    fclose(fgolden);

    //Since the program is executed on cache, we will execute
    //the function twice to reduce the cache miss interference.

    //========== run pure C algorithm ==========
    printf("----- pure C algorithm -----\n");

    //run 1st time to fill data into cache
    ndsv_mat_mul_f32_c(mat_src_f32_1, mat_src_f32_2, mat_out_f32, ROW, COL, COL2);

    //run 2nd time to get performance data
#if defined(__linux) || defined(__linux__) || defined(linux)
    ioctl(fd_cycle, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_cycle, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_ENABLE, 0);
#elif
    startPFM;
#endif
    ndsv_mat_mul_f32_c(mat_src_f32_1, mat_src_f32_2, mat_out_f32, ROW, COL, COL2);
#if defined(__linux) || defined(__linux__) || defined(linux)
    ioctl(fd_cycle, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_DISABLE, 0);
    read(fd_cycle, &cycle, sizeof(cycle));
    read(fd_inst, &inst, sizeof(inst));
#elif
    stopPFM;
#endif
    readResult();
    cycle_c = cycle;

    //check accuracy
    verify_buffer_f32(mat_out_f32, mat_golden_f32, MAT_MUL_OUT);

    //clear output array
    for(int i = 0; i < MAT_MUL_OUT; i++){
        mat_out_f32[i] = 0;
    }

    //========== run V-ext algorithm ==========
    printf("\n----- V-ext algorithm -----\n");

    //run 1st time to fill data into cache
    ndsv_mat_mul_f32_v(mat_src_f32_1, mat_src_f32_2, mat_out_f32, ROW, COL, COL2);

    //clear output array
    for(int i = 0; i < MAT_MUL_OUT; i++){
		mat_out_f32[i] = 0;
	}

    //run 2nd time to get performance data
#if defined(__linux) || defined(__linux__) || defined(linux)
    ioctl(fd_cycle, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_cycle, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_ENABLE, 0);
#elif
    startPFM;
#endif
    ndsv_mat_mul_f32_v(mat_src_f32_1, mat_src_f32_2, mat_out_f32, ROW, COL, COL2);
#if defined(__linux) || defined(__linux__) || defined(linux)
    ioctl(fd_cycle, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_inst, PERF_EVENT_IOC_DISABLE, 0);
    read(fd_cycle, &cycle, sizeof(cycle));
    read(fd_inst, &inst, sizeof(inst));
#elif
    stopPFM;
#endif
    readResult();
    cycle_v = cycle;

    //check accuracy
    verify_buffer_f32(mat_out_f32, mat_golden_f32, MAT_MUL_OUT);

    //calculate the speedup of V-ext over pure C algorithm
    printf("\nThe speedup of V-ext over pure C algorithm: %.2fx\n", (float)cycle_c / cycle_v);


	printf("\nThe demo program is done.\n");
    return 0;
}
