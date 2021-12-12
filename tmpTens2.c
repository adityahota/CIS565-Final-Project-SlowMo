#include "stdio.h"
#include "stdlib.h"

int main()
{
    const int cnt = 64 * 3 * 3 * 7 * 7;
    float *buf = malloc(cnt * sizeof(float));
    // FILE *tens = fopen("tmpTensStemSigmoid.bin", "rb");
    FILE *tens = fopen("tensor_bins/module.encoder.stem.0.weight__64x3x3x7x7.bin", "rb");
    fread(buf, sizeof(float), cnt, tens);
    fclose(tens);
    FILE *f = fopen("tmpTensStemRaw.txt", "w");
    for (int i = 0; i < cnt; i++)
    {
        fprintf(f, "%f\n", buf[i]);
    }
    fclose(f);
    free(buf);
    return 0;
}
