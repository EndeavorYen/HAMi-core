#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>

#include "include/memory_limit.h"


void create_new() {
    load_env_from_file(ENV_OVERRIDE_FILE);
    umask(000);
	char* shrreg_file = getenv(MULTIPROCESS_SHARED_REGION_CACHE_ENV);
    if (shrreg_file == NULL) {
        shrreg_file = MULTIPROCESS_SHARED_REGION_CACHE_DEFAULT;
    }
    int fd = open(shrreg_file, O_RDWR | O_CREAT | O_TRUNC, 0666);
    if (fd < 0) {
    	LOG_ERROR("Fail to create new shrreg file\n");
        assert(0);
    }
    close(fd);
    ensure_initialized();
}


void print_shared_region(){
    ensure_initialized();
    print_all();
}


void send_stop_signal(){
    ensure_initialized();
    suspend_all();
    while (1){
        if (wait_status_all(2))
            break;
        sleep(1);
    }
}

void send_resume_signal(){
    ensure_initialized();
    resume_all();
    while (1){
        if (wait_status_all(1))
            break;
        sleep(1);
    }
}

int main(int argc, char* argv[]) {
	int k;
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf(
            "--create_new    Create new shared region file\n"
            "--set-sm-limit <dev_id> <limit_percent>    Dynamically set SM limit (e.g., 0 30)\n"
        );
        return 0;
    }

    for (k = 0; k < argc; ++k) {
        char* arg = argv[k];
        if (strcmp(arg, "--create_new") == 0) {
            create_new();
        }
        if (strcmp(arg, "--suspend") == 0){
            send_stop_signal();
        }
        if (strcmp(arg, "--print") == 0){
            print_shared_region();
        }
        if (strcmp(arg, "--resume") == 0){
            send_resume_signal();
        }
        if (strcmp(arg, "--set-sm-limit") == 0) {
            if (k + 2 < argc) {
                int dev_id = atoi(argv[k + 1]);
                int limit = atoi(argv[k + 2]);
                set_current_device_sm_limit(dev_id, limit);
                k += 2; // 跳過已經處理過的參數
            } else {
                fprintf(stderr, "Error: --set-sm-limit requires <device_id> and <limit_percent> arguments.\n");
            }
        }
        if (strcmp(arg, "--print") == 0){
            print_shared_region();
        }
        if (strcmp(arg, "--version") == 0){
            printf("shrreg size: %ld, version %d.%d\n", 
                    sizeof(shared_region_t),
                    shrreg_major_version(),
                    shrreg_minor_version());
        }
    }
    return 0;
}

