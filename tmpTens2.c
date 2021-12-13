#include "stdio.h"
#include "stdlib.h"

int main(int argc, char *argv[])
{
    const int cnt = 8192;
    float *buf = malloc(cnt * sizeof(float));
    // FILE *tens = fopen("tmpTensStemSigmoid.bin", "rb");
    FILE *tens = fopen(argv[1], "rb");
    fread(buf, sizeof(float), cnt, tens);
    fclose(tens);
    FILE *f = fopen(argv[2], "w");
    for (int i = 0; i < cnt; i++)
    {
        fprintf(f, "%f\n", buf[i]);
    }
    fclose(f);
    free(buf);
    return 0;
}
