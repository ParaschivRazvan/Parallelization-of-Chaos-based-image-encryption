__kernel void image_XOR(
    __constant const unsigned int *inputImage,
    __global unsigned int *outputImage,
    double decimalsPwr,
    double serpentine2Pwr,
    unsigned int decimals,
    __global unsigned int *signature
) {
    unsigned int i = get_global_id(0);
    //i = (i & 0xffffff80) |  ((i & 0x1) << 6) | ((i & 0x3e) >> 1);
    unsigned int aux;

    unsigned long long XORseq;

    XORseq = decimals ^ (unsigned long long) floor(( 1.0 / (i + 1)) * decimalsPwr);

    unsigned int xd = ((XORseq >> (8*0)) & 0xff),
    yd = ((XORseq >> (8*1)) & 0xff),
    zd = ((XORseq >> (8*2)) & 0xff),
    wd = ((XORseq >> (8*3)) & 0xff);

    double x  = 1.0 / (xd == 0 ? 1.0 : xd) ,
        y = 1.0 / (yd == 0 ? 1.0 : yd),
        z = 1.0 / (zd == 0 ? 1.0 : zd),
        w = 1.0 / (wd == 0 ? 1.0 : wd);

    if (i % 2 == 1) {

        aux = floor(decimalsPwr * fabs(atan2( 1.0,tan(serpentine2Pwr *
                                                       (2 * 3.14159265358979323846 * ((x + y + z + w)/5 + 1) - 3.14159265358979323846/2)
        ))));
    } else {
        aux = floor(decimalsPwr * fabs(sin(2 * serpentine2Pwr *
                                           ((x + y + z + w)/10.0)
        ) / 2.0));
    }

    aux = aux << 8u; // comment if alfa chanel should be crypted as well
    aux = aux >> 8u;

    outputImage[i] = inputImage[i] ^ aux;

    *signature = *signature ^ inputImage[i] ^ aux;

}
