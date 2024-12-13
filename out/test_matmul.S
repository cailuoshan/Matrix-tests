
/nfs/home/cailuoshan/AISim/llvm-project/matrix-tests/out/test_matmul.out:	file format elf64-littleriscv

Disassembly of section .text:

0000000000010120 <exit>:
   10120: 41 11        	addi	sp, sp, -16
   10122: 81 45        	mv	a1, zero
   10124: 22 e0        	sd	s0, 0(sp)
   10126: 06 e4        	sd	ra, 8(sp)
   10128: 2a 84        	mv	s0, a0
   1012a: ef 20 00 32  	jal	0x1244a <__call_exitprocs>
   1012e: 03 b5 81 83  	ld	a0, -1992(gp)
   10132: 3c 6d        	ld	a5, 88(a0)
   10134: 91 c3        	beqz	a5, 0x10138 <exit+0x18>
   10136: 82 97        	jalr	a5
   10138: 22 85        	mv	a0, s0
   1013a: ef b0 10 06  	jal	0x1b99a <_exit>

000000000001013e <register_fini>:
   1013e: 93 07 00 00  	mv	a5, zero
   10142: 91 c7        	beqz	a5, 0x1014e <register_fini+0x10>
   10144: 4d 65        	lui	a0, 19
   10146: 13 05 85 98  	addi	a0, a0, -1656
   1014a: 6f 90 50 06  	j	0x199ae <atexit>
   1014e: 82 80        	ret

0000000000010150 <_start>:
   10150: 97 11 01 00  	auipc	gp, 17
   10154: 93 81 01 e0  	addi	gp, gp, -512
   10158: 13 85 01 87  	addi	a0, gp, -1936
   1015c: 17 36 01 00  	auipc	a2, 19
   10160: 13 06 c6 95  	addi	a2, a2, -1700
   10164: 09 8e        	sub	a2, a2, a0
   10166: 81 45        	mv	a1, zero
   10168: ef 00 80 4f  	jal	0x10660 <memset>
   1016c: 17 a5 00 00  	auipc	a0, 10
   10170: 13 05 25 84  	addi	a0, a0, -1982
   10174: 19 c5        	beqz	a0, 0x10182 <_start+0x32>
   10176: 17 35 00 00  	auipc	a0, 3
   1017a: 13 05 25 81  	addi	a0, a0, -2030
   1017e: ef 90 10 03  	jal	0x199ae <atexit>
   10182: ef 00 60 36  	jal	0x104e8 <__libc_init_array>
   10186: 02 45        	lw	a0, 0(sp)
   10188: 2c 00        	addi	a1, sp, 8
   1018a: 01 46        	mv	a2, zero
   1018c: ef 00 c0 17  	jal	0x10308 <main>
   10190: 41 bf        	j	0x10120 <exit>

0000000000010192 <__do_global_dtors_aux>:
   10192: 41 11        	addi	sp, sp, -16
   10194: 22 e0        	sd	s0, 0(sp)
   10196: 83 c7 81 8a  	lbu	a5, -1880(gp)
   1019a: 06 e4        	sd	ra, 8(sp)
   1019c: 99 ef        	bnez	a5, 0x101ba <__do_global_dtors_aux+0x28>
   1019e: 93 07 00 00  	mv	a5, zero
   101a2: 89 cb        	beqz	a5, 0x101b4 <__do_global_dtors_aux+0x22>
   101a4: 37 f5 01 00  	lui	a0, 31
   101a8: 13 05 05 63  	addi	a0, a0, 1584
   101ac: 97 00 00 00  	auipc	ra, 0
   101b0: e7 00 00 00  	jalr	zero
   101b4: 85 47        	addi	a5, zero, 1
   101b6: 23 84 f1 8a  	sb	a5, -1880(gp)
   101ba: a2 60        	ld	ra, 8(sp)
   101bc: 02 64        	ld	s0, 0(sp)
   101be: 41 01        	addi	sp, sp, 16
   101c0: 82 80        	ret

00000000000101c2 <frame_dummy>:
   101c2: 93 07 00 00  	mv	a5, zero
   101c6: 99 cb        	beqz	a5, 0x101dc <frame_dummy+0x1a>
   101c8: 37 f5 01 00  	lui	a0, 31
   101cc: 93 85 01 8b  	addi	a1, gp, -1872
   101d0: 13 05 05 63  	addi	a0, a0, 1584
   101d4: 17 03 00 00  	auipc	t1, 0
   101d8: 67 00 00 00  	jr	zero
   101dc: 82 80        	ret
   101de: 00 00        	unimp	

00000000000101e0 <check_addr>:
;   printf("load addr(M,m,n)=%d,%d,%d\n", M,m,n);
   101e0: 97 d6 00 00  	auipc	a3, 13
   101e4: 93 86 06 f6  	addi	a3, a3, -160
   101e8: 13 07 06 00  	mv	a4, a2
   101ec: 13 86 05 00  	mv	a2, a1
   101f0: 93 05 05 00  	mv	a1, a0
;   printf("load addr(M,m,n)=%d,%d,%d\n", M,m,n);
   101f4: 13 85 06 00  	mv	a0, a3
   101f8: 93 06 07 00  	mv	a3, a4
   101fc: 6f 00 00 53  	j	0x1072c <printf>

0000000000010200 <matmul>:
; void matmul(Tensor *dist, Tensor *src1, Tensor *src2) {
   10200: 13 01 01 fd  	addi	sp, sp, -48
;   const int M = src1->shape[1];
   10204: 23 34 81 02  	sd	s0, 40(sp)
   10208: 23 30 91 02  	sd	s1, 32(sp)
   1020c: 23 3c 21 01  	sd	s2, 24(sp)
   10210: 23 38 31 01  	sd	s3, 16(sp)
   10214: 23 34 41 01  	sd	s4, 8(sp)
   10218: 23 30 51 01  	sd	s5, 0(sp)
   1021c: 03 a8 c5 00  	lw	a6, 12(a1)
;   msettype(E32, M1, BA);
   10220: f7 76 14 00  	msettypei	a3, e32, m1, ba
;   for (int m = 0; m < M; m += tile_m) {
   10224: 63 52 00 0d  	blez	a6, 0x102e8 <matmul+0xe8>
   10228: 93 08 00 00  	mv	a7, zero
   1022c: 03 33 05 00  	ld	t1, 0(a0)
   10230: 83 a7 05 01  	lw	a5, 16(a1)
   10234: 83 26 06 01  	lw	a3, 16(a2)
   10238: 83 bf 05 00  	ld	t6, 0(a1)
   1023c: 03 3a 06 00  	ld	s4, 0(a2)
   10240: b3 23 f0 00  	sgtz	t2, a5
   10244: b3 22 d0 00  	sgtz	t0, a3
   10248: 13 95 26 00  	slli	a0, a3, 2
   1024c: 93 9a 27 00  	slli	s5, a5, 2
   10250: 6f 00 c0 00  	j	0x1025c <matmul+0x5c>
;   for (int m = 0; m < M; m += tile_m) {
   10254: bb 88 c8 01  	addw	a7, a7, t3
   10258: 63 d8 08 09  	bge	a7, a6, 0x102e8 <matmul+0xe8>
;     tile_m = msettilem(M - m);
   1025c: bb 05 18 41  	subw	a1, a6, a7
   10260: 77 fe 05 30  	msettilem	t3, a1
;     for (int n = 0; n < N; n += tile_n) {
   10264: e3 88 02 fe  	beqz	t0, 0x10254 <matmul+0x54>
   10268: 93 0e 00 00  	mv	t4, zero
   1026c: 3b 8f 08 03  	mulw	t5, a7, a6
   10270: bb 85 f8 02  	mulw	a1, a7, a5
   10274: 6f 00 00 01  	j	0x10284 <matmul+0x84>
;     for (int n = 0; n < N; n += tile_n) {
   10278: bb 8e 2e 01  	addw	t4, t4, s2
;       msce32_m(out, dist_ptr + m * M + n, N * sizeof(float));
   1027c: 77 a0 a9 02  	msce32.m	tr0, (s3), a0, m1
;     for (int n = 0; n < N; n += tile_n) {
   10280: e3 da de fc  	bge	t4, a3, 0x10254 <matmul+0x54>
;       tile_n = msettilen(N - n);
   10284: 3b 84 d6 41  	subw	s0, a3, t4
   10288: 77 79 04 70  	msettilen	s2, s0
;       mfloat32m1_t out = mlce32_m1(dist_ptr + m * M + n, N * sizeof(float));
   1028c: 1b 84 0e 00  	sext.w	s0, t4
   10290: b3 04 e4 01  	add	s1, s0, t5
   10294: 93 94 24 00  	slli	s1, s1, 2
   10298: b3 09 93 00  	add	s3, t1, s1
   1029c: 77 a0 a9 00  	mlce32.m	tr0, (s3), a0, m1
   102a0: e3 8c 03 fc  	beqz	t2, 0x10278 <matmul+0x78>
   102a4: 93 04 00 00  	mv	s1, zero
;         tile_k = msettilek(K - k);
   102a8: 3b 86 97 40  	subw	a2, a5, s1
   102ac: 77 76 06 50  	msettilek	a2, a2
;         mfloat32m1_t tr0 = mlae32_m1(src1_ptr + m * K + k, K * sizeof(float));
   102b0: 1b 87 04 00  	sext.w	a4, s1
   102b4: 33 07 b7 00  	add	a4, a4, a1
   102b8: 13 17 27 00  	slli	a4, a4, 2
   102bc: 33 87 ef 00  	add	a4, t6, a4
   102c0: f7 20 57 05  	mlae32.m	tr1, (a4), s5, m1
;         mfloat32m1_t tr1 = mlbe32_m1(src2_ptr + k * N + n, N * sizeof(float));
   102c4: 3b 87 d4 02  	mulw	a4, s1, a3
   102c8: 33 07 87 00  	add	a4, a4, s0
   102cc: 13 17 27 00  	slli	a4, a4, 2
   102d0: 33 07 ea 00  	add	a4, s4, a4
   102d4: 77 21 a7 08  	mlbe32.m	tr2, (a4), a0, m1
;       for (int k = 0; k < K; k += tile_k) {
   102d8: bb 84 c4 00  	addw	s1, s1, a2
;         out = mfma_mm(out, tr0, tr1);
   102dc: 77 e0 20 02  	mfma.mm	tr0, tr1, tr2
   102e0: e3 c4 f4 fc  	blt	s1, a5, 0x102a8 <matmul+0xa8>
   102e4: 6f f0 5f f9  	j	0x10278 <matmul+0x78>
; }
   102e8: 83 3a 01 00  	ld	s5, 0(sp)
   102ec: 03 3a 81 00  	ld	s4, 8(sp)
   102f0: 83 39 01 01  	ld	s3, 16(sp)
   102f4: 03 39 81 01  	ld	s2, 24(sp)
   102f8: 83 34 01 02  	ld	s1, 32(sp)
   102fc: 03 34 81 02  	ld	s0, 40(sp)
   10300: 13 01 01 03  	addi	sp, sp, 48
   10304: 67 80 00 00  	ret

0000000000010308 <main>:
; int main() {
   10308: 13 01 01 e9  	addi	sp, sp, -368
;   float dist[] = {
   1030c: 23 34 11 16  	sd	ra, 360(sp)
   10310: 23 30 81 16  	sd	s0, 352(sp)
   10314: 23 3c 91 14  	sd	s1, 344(sp)
   10318: 23 38 21 15  	sd	s2, 336(sp)
   1031c: 97 d5 00 00  	auipc	a1, 13
   10320: 93 85 85 19  	addi	a1, a1, 408
   10324: 13 05 c1 00  	addi	a0, sp, 12
   10328: 13 06 40 14  	addi	a2, zero, 324
   1032c: 13 09 c1 00  	addi	s2, sp, 12
   10330: ef 00 a0 22  	jal	0x1055a <memcpy>
   10334: 13 08 00 00  	mv	a6, zero
;   msettype(E32, M1, BA);
   10338: 77 75 14 00  	msettypei	a0, e32, m1, ba
   1033c: 93 02 90 00  	addi	t0, zero, 9
   10340: 13 06 a0 00  	addi	a2, zero, 10
   10344: 93 0f 40 02  	addi	t6, zero, 36
   10348: 17 d7 00 00  	auipc	a4, 13
   1034c: 13 07 c7 e9  	addi	a4, a4, -356
   10350: 13 0f 80 02  	addi	t5, zero, 40
   10354: 17 d5 00 00  	auipc	a0, 13
   10358: 13 05 85 ff  	addi	a0, a0, -8
   1035c: 93 0e 00 00  	mv	t4, zero
;     tile_m = msettilem(M - m);
   10360: bb 85 02 41  	subw	a1, t0, a6
   10364: 93 95 05 02  	slli	a1, a1, 32
   10368: 93 d5 05 02  	srli	a1, a1, 32
   1036c: f7 f8 05 30  	msettilem	a7, a1
   10370: 9b 15 38 00  	slliw	a1, a6, 3
   10374: 3b 83 05 01  	addw	t1, a1, a6
   10378: bb 05 c8 02  	mulw	a1, a6, a2
;       tile_n = msettilen(N - n);
   1037c: bb 84 d2 41  	subw	s1, t0, t4
   10380: 93 94 04 02  	slli	s1, s1, 32
   10384: 93 d4 04 02  	srli	s1, s1, 32
   10388: f7 f3 04 70  	msettilen	t2, s1
;       mfloat32m1_t out = mlce32_m1(dist_ptr + m * M + n, N * sizeof(float));
   1038c: 9b 84 0e 00  	sext.w	s1, t4
   10390: 33 84 64 00  	add	s0, s1, t1
   10394: 13 14 24 00  	slli	s0, s0, 2
   10398: 33 0e 89 00  	add	t3, s2, s0
   1039c: 77 20 fe 01  	mlce32.m	tr0, (t3), t6, m1
   103a0: 13 04 00 00  	mv	s0, zero
;         tile_k = msettilek(K - k);
   103a4: bb 07 86 40  	subw	a5, a2, s0
   103a8: 93 97 07 02  	slli	a5, a5, 32
   103ac: 93 d7 07 02  	srli	a5, a5, 32
   103b0: f7 f7 07 50  	msettilek	a5, a5
;         mfloat32m1_t tr0 = mlae32_m1(src1_ptr + m * K + k, K * sizeof(float));
   103b4: 9b 06 04 00  	sext.w	a3, s0
   103b8: b3 86 b6 00  	add	a3, a3, a1
   103bc: 93 96 26 00  	slli	a3, a3, 2
   103c0: b3 86 e6 00  	add	a3, a3, a4
   103c4: f7 a0 e6 05  	mlae32.m	tr1, (a3), t5, m1
;         mfloat32m1_t tr1 = mlbe32_m1(src2_ptr + k * N + n, N * sizeof(float));
   103c8: 9b 16 34 00  	slliw	a3, s0, 3
   103cc: bb 86 86 00  	addw	a3, a3, s0
   103d0: b3 86 96 00  	add	a3, a3, s1
   103d4: 93 96 26 00  	slli	a3, a3, 2
   103d8: b3 86 a6 00  	add	a3, a3, a0
   103dc: 77 a1 f6 09  	mlbe32.m	tr2, (a3), t6, m1
;       for (int k = 0; k < K; k += tile_k) {
   103e0: 3b 04 f4 00  	addw	s0, s0, a5
;         out = mfma_mm(out, tr0, tr1);
   103e4: 77 e0 20 02  	mfma.mm	tr0, tr1, tr2
;       for (int k = 0; k < K; k += tile_k) {
   103e8: e3 4e c4 fa  	blt	s0, a2, 0x103a4 <main+0x9c>
;     for (int n = 0; n < N; n += tile_n) {
   103ec: bb 8e 7e 00  	addw	t4, t4, t2
;       msce32_m(out, dist_ptr + m * M + n, N * sizeof(float));
   103f0: 77 20 fe 03  	msce32.m	tr0, (t3), t6, m1
;     for (int n = 0; n < N; n += tile_n) {
   103f4: e3 c4 5e f8  	blt	t4, t0, 0x1037c <main+0x74>
;   for (int m = 0; m < M; m += tile_m) {
   103f8: 3b 08 18 01  	addw	a6, a6, a7
   103fc: e3 40 58 f6  	blt	a6, t0, 0x1035c <main+0x54>
;   EXCEPT_FP32_ARRAY_LAX_EQ(ans_ptr, dst_ptr, 9 * 9,
   10400: 93 85 01 87  	addi	a1, gp, -1936
   10404: 03 a6 05 00  	lw	a2, 0(a1)
   10408: 13 05 00 00  	mv	a0, zero
   1040c: 1b 06 16 00  	addiw	a2, a2, 1
   10410: 23 a0 c5 00  	sw	a2, 0(a1)
   10414: 93 85 81 84  	addi	a1, gp, -1976
   10418: 07 b0 05 00  	fld	ft0, 0(a1)
   1041c: 93 85 01 85  	addi	a1, gp, -1968
   10420: 87 b0 05 00  	fld	ft1, 0(a1)
   10424: 93 05 c1 00  	addi	a1, sp, 12
   10428: 17 d6 00 00  	auipc	a2, 13
   1042c: 13 06 06 1d  	addi	a2, a2, 464
   10430: 93 06 10 05  	addi	a3, zero, 81
;   EXCEPT_FP32_ARRAY_LAX_EQ(ans_ptr, dst_ptr, 9 * 9,
   10434: 87 21 06 00  	flw	ft3, 0(a2)
   10438: 07 a1 05 00  	flw	ft2, 0(a1)
   1043c: 53 f2 21 08  	fsub.s	ft4, ft3, ft2
   10440: 53 02 02 42  	fcvt.d.s	ft4, ft4
   10444: 53 17 40 a2  	flt.d	a4, ft0, ft4
   10448: d3 17 12 a2  	flt.d	a5, ft4, ft1
   1044c: 33 67 f7 00  	or	a4, a4, a5
   10450: 63 14 07 04  	bnez	a4, 0x10498 <main+0x190>
   10454: 13 05 15 00  	addi	a0, a0, 1
   10458: 93 85 45 00  	addi	a1, a1, 4
   1045c: 13 06 46 00  	addi	a2, a2, 4
   10460: e3 1a d5 fc  	bne	a0, a3, 0x10434 <main+0x12c>
   10464: 13 85 41 87  	addi	a0, gp, -1932
   10468: 83 25 05 00  	lw	a1, 0(a0)
   1046c: 9b 85 15 00  	addiw	a1, a1, 1
   10470: 23 20 b5 00  	sw	a1, 0(a0)
   10474: 17 d5 00 00  	auipc	a0, 13
   10478: 13 05 c5 d5  	addi	a0, a0, -676
   1047c: 97 d5 00 00  	auipc	a1, 13
   10480: 93 85 05 d2  	addi	a1, a1, -736
   10484: 97 d6 00 00  	auipc	a3, 13
   10488: 93 86 26 d3  	addi	a3, a3, -718
   1048c: 13 06 d0 05  	addi	a2, zero, 93
   10490: ef 00 c0 29  	jal	0x1072c <printf>
   10494: 6f 00 80 03  	j	0x104cc <main+0x1c4>
;   EXCEPT_FP32_ARRAY_LAX_EQ(ans_ptr, dst_ptr, 9 * 9,
   10498: 53 80 01 42  	fcvt.d.s	ft0, ft3
   1049c: d3 00 01 42  	fcvt.d.s	ft1, ft2
   104a0: 1b 07 05 00  	sext.w	a4, a0
   104a4: d3 07 00 e2  	fmv.x.d	a5, ft0
   104a8: 53 88 00 e2  	fmv.x.d	a6, ft1
   104ac: 17 d5 00 00  	auipc	a0, 13
   104b0: 13 05 f5 ca  	addi	a0, a0, -849
   104b4: 97 d5 00 00  	auipc	a1, 13
   104b8: 93 85 85 ce  	addi	a1, a1, -792
   104bc: 97 d6 00 00  	auipc	a3, 13
   104c0: 93 86 a6 cf  	addi	a3, a3, -774
   104c4: 13 06 d0 05  	addi	a2, zero, 93
   104c8: ef 00 40 26  	jal	0x1072c <printf>
;   return 0;
   104cc: 13 05 00 00  	mv	a0, zero
   104d0: 03 39 01 15  	ld	s2, 336(sp)
   104d4: 83 34 81 15  	ld	s1, 344(sp)
   104d8: 03 34 01 16  	ld	s0, 352(sp)
   104dc: 83 30 81 16  	ld	ra, 360(sp)
   104e0: 13 01 01 17  	addi	sp, sp, 368
   104e4: 67 80 00 00  	ret

00000000000104e8 <__libc_init_array>:
   104e8: 01 11        	addi	sp, sp, -32
   104ea: 22 e8        	sd	s0, 16(sp)
   104ec: b7 f7 01 00  	lui	a5, 31
   104f0: 37 f4 01 00  	lui	s0, 31
   104f4: 4a e0        	sd	s2, 0(sp)
   104f6: 93 87 47 63  	addi	a5, a5, 1588
   104fa: 13 07 44 63  	addi	a4, s0, 1588
   104fe: 06 ec        	sd	ra, 24(sp)
   10500: 26 e4        	sd	s1, 8(sp)
   10502: 33 89 e7 40  	sub	s2, a5, a4
   10506: 63 8d e7 00  	beq	a5, a4, 0x10520 <__libc_init_array+0x38>
   1050a: 13 59 39 40  	srai	s2, s2, 3
   1050e: 13 04 44 63  	addi	s0, s0, 1588
   10512: 81 44        	mv	s1, zero
   10514: 1c 60        	ld	a5, 0(s0)
   10516: 85 04        	addi	s1, s1, 1
   10518: 21 04        	addi	s0, s0, 8
   1051a: 82 97        	jalr	a5
   1051c: e3 ec 24 ff  	bltu	s1, s2, 0x10514 <__libc_init_array+0x2c>
   10520: 37 f4 01 00  	lui	s0, 31
   10524: b7 f7 01 00  	lui	a5, 31
   10528: 93 87 87 64  	addi	a5, a5, 1608
   1052c: 13 07 84 63  	addi	a4, s0, 1592
   10530: 33 89 e7 40  	sub	s2, a5, a4
   10534: 13 59 39 40  	srai	s2, s2, 3
   10538: 63 8b e7 00  	beq	a5, a4, 0x1054e <__libc_init_array+0x66>
   1053c: 13 04 84 63  	addi	s0, s0, 1592
   10540: 81 44        	mv	s1, zero
   10542: 1c 60        	ld	a5, 0(s0)
   10544: 85 04        	addi	s1, s1, 1
   10546: 21 04        	addi	s0, s0, 8
   10548: 82 97        	jalr	a5
   1054a: e3 ec 24 ff  	bltu	s1, s2, 0x10542 <__libc_init_array+0x5a>
   1054e: e2 60        	ld	ra, 24(sp)
   10550: 42 64        	ld	s0, 16(sp)
   10552: a2 64        	ld	s1, 8(sp)
   10554: 02 69        	ld	s2, 0(sp)
   10556: 05 61        	addi	sp, sp, 32
   10558: 82 80        	ret

000000000001055a <memcpy>:
   1055a: b3 47 b5 00  	xor	a5, a0, a1
   1055e: 9d 8b        	andi	a5, a5, 7
   10560: b3 08 c5 00  	add	a7, a0, a2
   10564: f9 eb        	bnez	a5, 0x1063a <memcpy+0xe0>
   10566: 9d 47        	addi	a5, zero, 7
   10568: 63 f9 c7 0c  	bgeu	a5, a2, 0x1063a <memcpy+0xe0>
   1056c: 93 77 75 00  	andi	a5, a0, 7
   10570: 2a 87        	mv	a4, a0
   10572: a9 eb        	bnez	a5, 0x105c4 <memcpy+0x6a>
   10574: 93 f7 88 ff  	andi	a5, a7, -8
   10578: 33 86 e7 40  	sub	a2, a5, a4
   1057c: 93 06 00 04  	addi	a3, zero, 64
   10580: 93 02 00 04  	addi	t0, zero, 64
   10584: 63 c3 c6 06  	blt	a3, a2, 0x105ea <memcpy+0x90>
   10588: 63 7b f7 02  	bgeu	a4, a5, 0x105be <memcpy+0x64>
   1058c: fd 17        	addi	a5, a5, -1
   1058e: 99 8f        	sub	a5, a5, a4
   10590: 8d 83        	srli	a5, a5, 3
   10592: 13 88 17 00  	addi	a6, a5, 1
   10596: ba 86        	mv	a3, a4
   10598: 2e 86        	mv	a2, a1
   1059a: c2 87        	mv	a5, a6
   1059c: 73 23 20 c2  	csrr	t1, vlenb
   105a0: 57 fe 87 05  	<unknown>
   105a4: 07 7c 06 02  	<unknown>
   105a8: b3 87 c7 41  	sub	a5, a5, t3
   105ac: 27 fc 06 02  	<unknown>
   105b0: 1a 96        	add	a2, a2, t1
   105b2: 9a 96        	add	a3, a3, t1
   105b4: f5 f7        	bnez	a5, 0x105a0 <memcpy+0x46>
   105b6: 93 17 38 00  	slli	a5, a6, 3
   105ba: 3e 97        	add	a4, a4, a5
   105bc: be 95        	add	a1, a1, a5
   105be: 63 61 17 09  	bltu	a4, a7, 0x10640 <memcpy+0xe6>
   105c2: 82 80        	ret
   105c4: 83 c6 05 00  	lbu	a3, 0(a1)
   105c8: 05 07        	addi	a4, a4, 1
   105ca: 93 77 77 00  	andi	a5, a4, 7
   105ce: a3 0f d7 fe  	sb	a3, -1(a4)
   105d2: 85 05        	addi	a1, a1, 1
   105d4: c5 d3        	beqz	a5, 0x10574 <memcpy+0x1a>
   105d6: 83 c6 05 00  	lbu	a3, 0(a1)
   105da: 05 07        	addi	a4, a4, 1
   105dc: 93 77 77 00  	andi	a5, a4, 7
   105e0: a3 0f d7 fe  	sb	a3, -1(a4)
   105e4: 85 05        	addi	a1, a1, 1
   105e6: f9 ff        	bnez	a5, 0x105c4 <memcpy+0x6a>
   105e8: 71 b7        	j	0x10574 <memcpy+0x1a>
   105ea: 90 65        	ld	a2, 8(a1)
   105ec: 94 7d        	ld	a3, 56(a1)
   105ee: 83 bf 05 00  	ld	t6, 0(a1)
   105f2: 03 bf 05 01  	ld	t5, 16(a1)
   105f6: 83 be 85 01  	ld	t4, 24(a1)
   105fa: 03 be 05 02  	ld	t3, 32(a1)
   105fe: 03 b3 85 02  	ld	t1, 40(a1)
   10602: 03 b8 05 03  	ld	a6, 48(a1)
   10606: 10 e7        	sd	a2, 8(a4)
   10608: b0 61        	ld	a2, 64(a1)
   1060a: 23 30 f7 01  	sd	t6, 0(a4)
   1060e: 23 38 e7 01  	sd	t5, 16(a4)
   10612: 23 3c d7 01  	sd	t4, 24(a4)
   10616: 23 30 c7 03  	sd	t3, 32(a4)
   1061a: 23 34 67 02  	sd	t1, 40(a4)
   1061e: 23 38 07 03  	sd	a6, 48(a4)
   10622: 14 ff        	sd	a3, 56(a4)
   10624: 13 07 87 04  	addi	a4, a4, 72
   10628: b3 86 e7 40  	sub	a3, a5, a4
   1062c: 23 3c c7 fe  	sd	a2, -8(a4)
   10630: 93 85 85 04  	addi	a1, a1, 72
   10634: e3 cb d2 fa  	blt	t0, a3, 0x105ea <memcpy+0x90>
   10638: 81 bf        	j	0x10588 <memcpy+0x2e>
   1063a: 2a 87        	mv	a4, a0
   1063c: 63 71 15 03  	bgeu	a0, a7, 0x1065e <memcpy+0x104>
   10640: b3 87 e8 40  	sub	a5, a7, a4
   10644: f3 26 20 c2  	csrr	a3, vlenb
   10648: 57 f6 07 04  	<unknown>
   1064c: 07 8c 05 02  	<unknown>
   10650: 91 8f        	sub	a5, a5, a2
   10652: 27 0c 07 02  	<unknown>
   10656: b6 95        	add	a1, a1, a3
   10658: 36 97        	add	a4, a4, a3
   1065a: fd f7        	bnez	a5, 0x10648 <memcpy+0xee>
   1065c: 82 80        	ret
   1065e: 82 80        	ret

0000000000010660 <memset>:
   10660: 3d 43        	addi	t1, zero, 15
   10662: 2a 87        	mv	a4, a0
   10664: 63 71 c3 02  	bgeu	t1, a2, 0x10686 <memset+0x26>
   10668: 93 77 f7 00  	andi	a5, a4, 15
   1066c: c1 e3        	bnez	a5, 0x106ec <memset+0x8c>
   1066e: bd e1        	bnez	a1, 0x106d4 <memset+0x74>
   10670: 93 76 06 ff  	andi	a3, a2, -16
   10674: 3d 8a        	andi	a2, a2, 15
   10676: ba 96        	add	a3, a3, a4
   10678: 0c e3        	sd	a1, 0(a4)
   1067a: 0c e7        	sd	a1, 8(a4)
   1067c: 41 07        	addi	a4, a4, 16
   1067e: e3 6d d7 fe  	bltu	a4, a3, 0x10678 <memset+0x18>
   10682: 11 e2        	bnez	a2, 0x10686 <memset+0x26>
   10684: 82 80        	ret
   10686: b3 06 c3 40  	sub	a3, t1, a2
   1068a: 8a 06        	slli	a3, a3, 2
   1068c: 97 02 00 00  	auipc	t0, 0
   10690: 96 96        	add	a3, a3, t0
   10692: 67 80 a6 00  	jr	10(a3)
   10696: 23 07 b7 00  	sb	a1, 14(a4)
   1069a: a3 06 b7 00  	sb	a1, 13(a4)
   1069e: 23 06 b7 00  	sb	a1, 12(a4)
   106a2: a3 05 b7 00  	sb	a1, 11(a4)
   106a6: 23 05 b7 00  	sb	a1, 10(a4)
   106aa: a3 04 b7 00  	sb	a1, 9(a4)
   106ae: 23 04 b7 00  	sb	a1, 8(a4)
   106b2: a3 03 b7 00  	sb	a1, 7(a4)
   106b6: 23 03 b7 00  	sb	a1, 6(a4)
   106ba: a3 02 b7 00  	sb	a1, 5(a4)
   106be: 23 02 b7 00  	sb	a1, 4(a4)
   106c2: a3 01 b7 00  	sb	a1, 3(a4)
   106c6: 23 01 b7 00  	sb	a1, 2(a4)
   106ca: a3 00 b7 00  	sb	a1, 1(a4)
   106ce: 23 00 b7 00  	sb	a1, 0(a4)
   106d2: 82 80        	ret
   106d4: 93 f5 f5 0f  	andi	a1, a1, 255
   106d8: 93 96 85 00  	slli	a3, a1, 8
   106dc: d5 8d        	or	a1, a1, a3
   106de: 93 96 05 01  	slli	a3, a1, 16
   106e2: d5 8d        	or	a1, a1, a3
   106e4: 93 96 05 02  	slli	a3, a1, 32
   106e8: d5 8d        	or	a1, a1, a3
   106ea: 59 b7        	j	0x10670 <memset+0x10>
   106ec: 93 96 27 00  	slli	a3, a5, 2
   106f0: 97 02 00 00  	auipc	t0, 0
   106f4: 96 96        	add	a3, a3, t0
   106f6: 86 82        	mv	t0, ra
   106f8: e7 80 26 fa  	jalr	-94(a3)
   106fc: 96 80        	mv	ra, t0
   106fe: c1 17        	addi	a5, a5, -16
   10700: 1d 8f        	sub	a4, a4, a5
   10702: 3e 96        	add	a2, a2, a5
   10704: e3 71 c3 f8  	bgeu	t1, a2, 0x10686 <memset+0x26>
   10708: 9d b7        	j	0x1066e <memset+0xe>

000000000001070a <_printf_r>:
   1070a: 5d 71        	addi	sp, sp, -80
   1070c: 32 f0        	sd	a2, 32(sp)
   1070e: 36 f4        	sd	a3, 40(sp)
   10710: 3a f8        	sd	a4, 48(sp)
   10712: 3e fc        	sd	a5, 56(sp)
   10714: c2 e0        	sd	a6, 64(sp)
   10716: c6 e4        	sd	a7, 72(sp)
   10718: 2e 86        	mv	a2, a1
   1071a: 0c 69        	ld	a1, 16(a0)
   1071c: 14 10        	addi	a3, sp, 32
   1071e: 06 ec        	sd	ra, 24(sp)
   10720: 36 e4        	sd	a3, 8(sp)
   10722: ef 00 60 03  	jal	0x10758 <_vfprintf_r>
   10726: e2 60        	ld	ra, 24(sp)
   10728: 61 61        	addi	sp, sp, 80
   1072a: 82 80        	ret

000000000001072c <printf>:
   1072c: 03 b3 81 85  	ld	t1, -1960(gp)
   10730: 1d 71        	addi	sp, sp, -96
   10732: 32 f8        	sd	a2, 48(sp)
   10734: 36 fc        	sd	a3, 56(sp)
   10736: 2e f4        	sd	a1, 40(sp)
   10738: ba e0        	sd	a4, 64(sp)
   1073a: be e4        	sd	a5, 72(sp)
   1073c: c2 e8        	sd	a6, 80(sp)
   1073e: c6 ec        	sd	a7, 88(sp)
   10740: 83 35 03 01  	ld	a1, 16(t1)
   10744: 34 10        	addi	a3, sp, 40
   10746: 2a 86        	mv	a2, a0
   10748: 1a 85        	mv	a0, t1
   1074a: 06 ec        	sd	ra, 24(sp)
   1074c: 36 e4        	sd	a3, 8(sp)
   1074e: ef 00 a0 00  	jal	0x10758 <_vfprintf_r>
   10752: e2 60        	ld	ra, 24(sp)
   10754: 25 61        	addi	sp, sp, 96
   10756: 82 80        	ret

0000000000010758 <_vfprintf_r>:
   10758: 13 01 01 d9  	addi	sp, sp, -624
   1075c: 23 34 11 26  	sd	ra, 616(sp)
   10760: 23 30 81 26  	sd	s0, 608(sp)
   10764: 23 38 21 25  	sd	s2, 592(sp)
   10768: 2e e0        	sd	a1, 0(sp)
   1076a: 32 89        	mv	s2, a2
   1076c: 36 f0        	sd	a3, 32(sp)
   1076e: 23 3c 91 24  	sd	s1, 600(sp)
   10772: 23 34 31 25  	sd	s3, 584(sp)
   10776: 23 30 41 25  	sd	s4, 576(sp)
   1077a: 23 3c 51 23  	sd	s5, 568(sp)
   1077e: 23 38 61 23  	sd	s6, 560(sp)
   10782: 23 34 71 23  	sd	s7, 552(sp)
   10786: 23 30 81 23  	sd	s8, 544(sp)
   1078a: 23 3c 91 21  	sd	s9, 536(sp)
   1078e: 23 38 a1 21  	sd	s10, 528(sp)
   10792: 23 34 b1 21  	sd	s11, 520(sp)
   10796: 2a 84        	mv	s0, a0
   10798: aa e8        	sd	a0, 80(sp)
   1079a: ef 40 f0 52  	jal	0x154c8 <_localeconv_r>
   1079e: 1c 61        	ld	a5, 0(a0)
   107a0: 3e 85        	mv	a0, a5
   107a2: be f0        	sd	a5, 96(sp)
   107a4: ef 60 c0 6d  	jal	0x16e80 <strlen>
   107a8: aa ec        	sd	a0, 88(sp)
   107aa: 82 f9        	sd	zero, 240(sp)
   107ac: 82 fd        	sd	zero, 248(sp)
   107ae: 01 c4        	beqz	s0, 0x107b6 <_vfprintf_r+0x5e>
   107b0: 38 48        	lw	a4, 80(s0)
   107b2: 63 0c 07 46  	beqz	a4, 0x10c2a <_vfprintf_r+0x4d2>
   107b6: 82 67        	ld	a5, 0(sp)
   107b8: 03 97 07 01  	lh	a4, 16(a5)
   107bc: 93 16 27 03  	slli	a3, a4, 50
   107c0: 63 c1 06 02  	bltz	a3, 0x107e2 <_vfprintf_r+0x8a>
   107c4: 83 a6 c7 0a  	lw	a3, 172(a5)
   107c8: 09 66        	lui	a2, 2
   107ca: 51 8f        	or	a4, a4, a2
   107cc: 79 76        	lui	a2, 1048574
   107ce: 1b 17 07 01  	slliw	a4, a4, 16
   107d2: 7d 16        	addi	a2, a2, -1
   107d4: 1b 57 07 41  	sraiw	a4, a4, 16
   107d8: f1 8e        	and	a3, a3, a2
   107da: 23 98 e7 00  	sh	a4, 16(a5)
   107de: 23 a6 d7 0a  	sw	a3, 172(a5)
   107e2: 93 76 87 00  	andi	a3, a4, 8
   107e6: 63 85 06 26  	beqz	a3, 0x10a50 <_vfprintf_r+0x2f8>
   107ea: 82 67        	ld	a5, 0(sp)
   107ec: 94 6f        	ld	a3, 24(a5)
   107ee: 63 81 06 26  	beqz	a3, 0x10a50 <_vfprintf_r+0x2f8>
   107f2: 69 8b        	andi	a4, a4, 26
   107f4: a9 46        	addi	a3, zero, 10
   107f6: 63 0b d7 26  	beq	a4, a3, 0x10a6c <_vfprintf_r+0x314>
   107fa: 1c 03        	addi	a5, sp, 384
   107fc: 75 67        	lui	a4, 29
   107fe: 3e 83        	mv	t1, a5
   10800: 3e e2        	sd	a5, 256(sp)
   10802: 4a 8b        	mv	s6, s2
   10804: 93 07 c7 79  	addi	a5, a4, 1948
   10808: 79 67        	lui	a4, 30
   1080a: 3e f4        	sd	a5, 40(sp)
   1080c: 93 07 07 9a  	addi	a5, a4, -1632
   10810: 03 47 0b 00  	lbu	a4, 0(s6)
   10814: 02 ea        	sd	zero, 272(sp)
   10816: 23 24 01 10  	sw	zero, 264(sp)
   1081a: 02 fc        	sd	zero, 56(sp)
   1081c: 82 f4        	sd	zero, 104(sp)
   1081e: 82 f8        	sd	zero, 112(sp)
   10820: 02 e1        	sd	zero, 128(sp)
   10822: 02 e5        	sd	zero, 136(sp)
   10824: 82 fc        	sd	zero, 120(sp)
   10826: 02 e8        	sd	zero, 16(sp)
   10828: 3e e4        	sd	a5, 8(sp)
   1082a: c6 69        	ld	s3, 80(sp)
   1082c: 1a 84        	mv	s0, t1
   1082e: 63 01 07 1c  	beqz	a4, 0x109f0 <_vfprintf_r+0x298>
   10832: da 84        	mv	s1, s6
   10834: 93 06 50 02  	addi	a3, zero, 37
   10838: 63 07 d7 24  	beq	a4, a3, 0x10a86 <_vfprintf_r+0x32e>
   1083c: 03 c7 14 00  	lbu	a4, 1(s1)
   10840: 85 04        	addi	s1, s1, 1
   10842: 7d fb        	bnez	a4, 0x10838 <_vfprintf_r+0xe0>
   10844: 3b 89 64 41  	subw	s2, s1, s6
   10848: 63 04 09 1a  	beqz	s2, 0x109f0 <_vfprintf_r+0x298>
   1084c: d2 6b        	ld	s7, 272(sp)
   1084e: 83 26 81 10  	lw	a3, 264(sp)
   10852: 23 30 64 01  	sd	s6, 0(s0)
   10856: 33 87 2b 01  	add	a4, s7, s2
   1085a: 1b 86 16 00  	addiw	a2, a3, 1
   1085e: 23 34 24 01  	sd	s2, 8(s0)
   10862: 3a ea        	sd	a4, 272(sp)
   10864: 23 24 c1 10  	sw	a2, 264(sp)
   10868: 1d 47        	addi	a4, zero, 7
   1086a: 41 04        	addi	s0, s0, 16
   1086c: 63 44 c7 22  	blt	a4, a2, 0x10a94 <_vfprintf_r+0x33c>
   10870: c2 67        	ld	a5, 16(sp)
   10872: 03 c7 04 00  	lbu	a4, 0(s1)
   10876: bb 87 27 01  	addw	a5, a5, s2
   1087a: 3e e8        	sd	a5, 16(sp)
   1087c: 63 0a 07 16  	beqz	a4, 0x109f0 <_vfprintf_r+0x298>
   10880: 7d 53        	addi	t1, zero, -1
   10882: 03 ce 14 00  	lbu	t3, 1(s1)
   10886: a3 07 01 0c  	sb	zero, 207(sp)
   1088a: 85 04        	addi	s1, s1, 1
   1088c: 01 4a        	mv	s4, zero
   1088e: 01 49        	mv	s2, zero
   10890: 93 0b a0 05  	addi	s7, zero, 90
   10894: 9a 8a        	mv	s5, t1
   10896: 85 04        	addi	s1, s1, 1
   10898: 1b 0d 0e 00  	sext.w	s10, t3
   1089c: 9b 07 0d fe  	addiw	a5, s10, -32
   108a0: 1b 87 07 00  	sext.w	a4, a5
   108a4: 63 e5 eb 04  	bltu	s7, a4, 0x108ee <_vfprintf_r+0x196>
   108a8: 13 97 07 02  	slli	a4, a5, 32
   108ac: 93 57 e7 01  	srli	a5, a4, 30
   108b0: 22 77        	ld	a4, 40(sp)
   108b2: ba 97        	add	a5, a5, a4
   108b4: 9c 43        	lw	a5, 0(a5)
   108b6: 82 87        	jr	a5
   108b8: 01 4a        	mv	s4, zero
   108ba: a5 46        	addi	a3, zero, 9
   108bc: 03 c7 04 00  	lbu	a4, 0(s1)
   108c0: 9b 17 2a 00  	slliw	a5, s4, 2
   108c4: 3b 8a 47 01  	addw	s4, a5, s4
   108c8: 1b 0d 0d fd  	addiw	s10, s10, -48
   108cc: 1b 1a 1a 00  	slliw	s4, s4, 1
   108d0: 9b 07 07 fd  	addiw	a5, a4, -48
   108d4: 3b 0a 4d 01  	addw	s4, s10, s4
   108d8: 85 04        	addi	s1, s1, 1
   108da: 1b 0d 07 00  	sext.w	s10, a4
   108de: e3 ff f6 fc  	bgeu	a3, a5, 0x108bc <_vfprintf_r+0x164>
   108e2: 9b 07 0d fe  	addiw	a5, s10, -32
   108e6: 1b 87 07 00  	sext.w	a4, a5
   108ea: e3 ff eb fa  	bgeu	s7, a4, 0x108a8 <_vfprintf_r+0x150>
   108ee: 63 01 0d 10  	beqz	s10, 0x109f0 <_vfprintf_r+0x298>
   108f2: 23 0c a1 11  	sb	s10, 280(sp)
   108f6: a3 07 01 0c  	sb	zero, 207(sp)
   108fa: 05 4c        	addi	s8, zero, 1
   108fc: 85 4d        	addi	s11, zero, 1
   108fe: 13 0b 81 11  	addi	s6, sp, 280
   10902: 02 ec        	sd	zero, 24(sp)
   10904: 01 43        	mv	t1, zero
   10906: 82 e4        	sd	zero, 72(sp)
   10908: 82 e0        	sd	zero, 64(sp)
   1090a: 02 f8        	sd	zero, 48(sp)
   1090c: 93 7c 29 00  	andi	s9, s2, 2
   10910: 63 83 0c 00  	beqz	s9, 0x10916 <_vfprintf_r+0x1be>
   10914: 09 2c        	addiw	s8, s8, 2
   10916: 93 7a 49 08  	andi	s5, s2, 132
   1091a: 63 96 0a 00  	bnez	s5, 0x10926 <_vfprintf_r+0x1ce>
   1091e: 3b 07 8a 41  	subw	a4, s4, s8
   10922: e3 42 e0 04  	bgtz	a4, 0x11166 <_vfprintf_r+0xa0e>
   10926: 03 47 f1 0c  	lbu	a4, 207(sp)
   1092a: d2 6b        	ld	s7, 272(sp)
   1092c: 1d c3        	beqz	a4, 0x10952 <_vfprintf_r+0x1fa>
   1092e: 03 26 81 10  	lw	a2, 264(sp)
   10932: 93 05 f1 0c  	addi	a1, sp, 207
   10936: 0c e0        	sd	a1, 0(s0)
   10938: 05 26        	addiw	a2, a2, 1
   1093a: 85 45        	addi	a1, zero, 1
   1093c: 85 0b        	addi	s7, s7, 1
   1093e: 0c e4        	sd	a1, 8(s0)
   10940: 23 24 c1 10  	sw	a2, 264(sp)
   10944: 9b 05 06 00  	sext.w	a1, a2
   10948: 5e ea        	sd	s7, 272(sp)
   1094a: 1d 46        	addi	a2, zero, 7
   1094c: 41 04        	addi	s0, s0, 16
   1094e: 63 43 b6 20  	blt	a2, a1, 0x10b54 <_vfprintf_r+0x3fc>
   10952: 63 83 0c 02  	beqz	s9, 0x10978 <_vfprintf_r+0x220>
   10956: 03 26 81 10  	lw	a2, 264(sp)
   1095a: 89 45        	addi	a1, zero, 2
   1095c: 89 0b        	addi	s7, s7, 2
   1095e: 05 26        	addiw	a2, a2, 1
   10960: 9c 09        	addi	a5, sp, 208
   10962: 0c e4        	sd	a1, 8(s0)
   10964: 1c e0        	sd	a5, 0(s0)
   10966: 9b 05 06 00  	sext.w	a1, a2
   1096a: 23 24 c1 10  	sw	a2, 264(sp)
   1096e: 5e ea        	sd	s7, 272(sp)
   10970: 1d 46        	addi	a2, zero, 7
   10972: 41 04        	addi	s0, s0, 16
   10974: e3 45 b6 0a  	blt	a2, a1, 0x1121e <_vfprintf_r+0xac6>
   10978: 13 06 00 08  	addi	a2, zero, 128
   1097c: 63 80 ca 6a  	beq	s5, a2, 0x1101c <_vfprintf_r+0x8c4>
   10980: bb 0c b3 41  	subw	s9, t1, s11
   10984: 63 4a 90 73  	bgtz	s9, 0x110b8 <_vfprintf_r+0x960>
   10988: 13 76 09 10  	andi	a2, s2, 256
   1098c: 63 15 06 54  	bnez	a2, 0x10ed6 <_vfprintf_r+0x77e>
   10990: 03 26 81 10  	lw	a2, 264(sp)
   10994: 33 87 7d 01  	add	a4, s11, s7
   10998: 23 30 64 01  	sd	s6, 0(s0)
   1099c: 9b 06 16 00  	addiw	a3, a2, 1
   109a0: 23 34 b4 01  	sd	s11, 8(s0)
   109a4: 3a ea        	sd	a4, 272(sp)
   109a6: 23 24 d1 10  	sw	a3, 264(sp)
   109aa: 9d 47        	addi	a5, zero, 7
   109ac: 63 c5 d7 26  	blt	a5, a3, 0x10c16 <_vfprintf_r+0x4be>
   109b0: 41 04        	addi	s0, s0, 16
   109b2: 13 79 49 00  	andi	s2, s2, 4
   109b6: 63 06 09 00  	beqz	s2, 0x109c2 <_vfprintf_r+0x26a>
   109ba: 3b 09 8a 41  	subw	s2, s4, s8
   109be: e3 4c 20 07  	bgtz	s2, 0x11236 <_vfprintf_r+0xade>
   109c2: d2 87        	mv	a5, s4
   109c4: 63 53 8a 01  	bge	s4, s8, 0x109ca <_vfprintf_r+0x272>
   109c8: e2 87        	mv	a5, s8
   109ca: c2 66        	ld	a3, 16(sp)
   109cc: b5 9f        	addw	a5, a5, a3
   109ce: 3e e8        	sd	a5, 16(sp)
   109d0: 63 1d 07 76  	bnez	a4, 0x1114a <_vfprintf_r+0x9f2>
   109d4: e2 67        	ld	a5, 24(sp)
   109d6: 23 24 01 10  	sw	zero, 264(sp)
   109da: 89 c7        	beqz	a5, 0x109e4 <_vfprintf_r+0x28c>
   109dc: e2 65        	ld	a1, 24(sp)
   109de: 4e 85        	mv	a0, s3
   109e0: ef 20 40 0b  	jal	0x12a94 <_free_r>
   109e4: 00 03        	addi	s0, sp, 384
   109e6: 26 8b        	mv	s6, s1
   109e8: 03 47 0b 00  	lbu	a4, 0(s6)
   109ec: e3 13 07 e4  	bnez	a4, 0x10832 <_vfprintf_r+0xda>
   109f0: d2 67        	ld	a5, 272(sp)
   109f2: 99 c3        	beqz	a5, 0x109f8 <_vfprintf_r+0x2a0>
   109f4: 6f 10 00 2b  	j	0x11ca4 <_vfprintf_r+0x154c>
   109f8: 82 67        	ld	a5, 0(sp)
   109fa: 83 d7 07 01  	lhu	a5, 16(a5)
   109fe: 93 f7 07 04  	andi	a5, a5, 64
   10a02: 99 c3        	beqz	a5, 0x10a08 <_vfprintf_r+0x2b0>
   10a04: 6f 10 80 6e  	j	0x120ec <_vfprintf_r+0x1994>
   10a08: 83 30 81 26  	ld	ra, 616(sp)
   10a0c: 03 34 01 26  	ld	s0, 608(sp)
   10a10: 42 65        	ld	a0, 16(sp)
   10a12: 83 34 81 25  	ld	s1, 600(sp)
   10a16: 03 39 01 25  	ld	s2, 592(sp)
   10a1a: 83 39 81 24  	ld	s3, 584(sp)
   10a1e: 03 3a 01 24  	ld	s4, 576(sp)
   10a22: 83 3a 81 23  	ld	s5, 568(sp)
   10a26: 03 3b 01 23  	ld	s6, 560(sp)
   10a2a: 83 3b 81 22  	ld	s7, 552(sp)
   10a2e: 03 3c 01 22  	ld	s8, 544(sp)
   10a32: 83 3c 81 21  	ld	s9, 536(sp)
   10a36: 03 3d 01 21  	ld	s10, 528(sp)
   10a3a: 83 3d 81 20  	ld	s11, 520(sp)
   10a3e: 13 01 01 27  	addi	sp, sp, 624
   10a42: 82 80        	ret
   10a44: 13 69 09 01  	ori	s2, s2, 16
   10a48: 03 ce 04 00  	lbu	t3, 0(s1)
   10a4c: 01 29        	sext.w	s2, s2
   10a4e: a1 b5        	j	0x10896 <_vfprintf_r+0x13e>
   10a50: 82 65        	ld	a1, 0(sp)
   10a52: 46 65        	ld	a0, 80(sp)
   10a54: ef 10 b0 11  	jal	0x1236e <__swsetup_r>
   10a58: 19 c1        	beqz	a0, 0x10a5e <_vfprintf_r+0x306>
   10a5a: 6f 10 20 69  	j	0x120ec <_vfprintf_r+0x1994>
   10a5e: 82 67        	ld	a5, 0(sp)
   10a60: a9 46        	addi	a3, zero, 10
   10a62: 03 97 07 01  	lh	a4, 16(a5)
   10a66: 69 8b        	andi	a4, a4, 26
   10a68: e3 19 d7 d8  	bne	a4, a3, 0x107fa <_vfprintf_r+0xa2>
   10a6c: 82 67        	ld	a5, 0(sp)
   10a6e: 03 97 27 01  	lh	a4, 18(a5)
   10a72: e3 44 07 d8  	bltz	a4, 0x107fa <_vfprintf_r+0xa2>
   10a76: 82 76        	ld	a3, 32(sp)
   10a78: 46 65        	ld	a0, 80(sp)
   10a7a: 4a 86        	mv	a2, s2
   10a7c: be 85        	mv	a1, a5
   10a7e: ef 10 d0 05  	jal	0x122da <__sbprintf>
   10a82: 2a e8        	sd	a0, 16(sp)
   10a84: 51 b7        	j	0x10a08 <_vfprintf_r+0x2b0>
   10a86: 3b 89 64 41  	subw	s2, s1, s6
   10a8a: e3 11 09 dc  	bnez	s2, 0x1084c <_vfprintf_r+0xf4>
   10a8e: 03 c7 04 00  	lbu	a4, 0(s1)
   10a92: ed b3        	j	0x1087c <_vfprintf_r+0x124>
   10a94: 82 65        	ld	a1, 0(sp)
   10a96: 10 02        	addi	a2, sp, 256
   10a98: 4e 85        	mv	a0, s3
   10a9a: ef 80 60 0e  	jal	0x18b80 <__sprint_r>
   10a9e: 29 fd        	bnez	a0, 0x109f8 <_vfprintf_r+0x2a0>
   10aa0: 00 03        	addi	s0, sp, 384
   10aa2: f9 b3        	j	0x10870 <_vfprintf_r+0x118>
   10aa4: 13 77 89 00  	andi	a4, s2, 8
   10aa8: 56 83        	mv	t1, s5
   10aaa: e3 11 07 2c  	bnez	a4, 0x1156c <_vfprintf_r+0xe14>
   10aae: 82 77        	ld	a5, 32(sp)
   10ab0: 56 ec        	sd	s5, 24(sp)
   10ab2: 88 23        	fld	fa0, 0(a5)
   10ab4: 93 8a 87 00  	addi	s5, a5, 8
   10ab8: 56 f0        	sd	s5, 32(sp)
   10aba: ef c0 c0 30  	jal	0x1cdc6 <__extenddftf2>
   10abe: 62 63        	ld	t1, 24(sp)
   10ac0: 2a 87        	mv	a4, a0
   10ac2: 88 19        	addi	a0, sp, 240
   10ac4: 1a ec        	sd	t1, 24(sp)
   10ac6: ba f9        	sd	a4, 240(sp)
   10ac8: ae fd        	sd	a1, 248(sp)
   10aca: ef 40 70 1b  	jal	0x15480 <_ldcheck>
   10ace: aa cd        	sw	a0, 216(sp)
   10ad0: 09 47        	addi	a4, zero, 2
   10ad2: 62 63        	ld	t1, 24(sp)
   10ad4: e3 0d e5 50  	beq	a0, a4, 0x117ee <_vfprintf_r+0x1096>
   10ad8: 05 47        	addi	a4, zero, 1
   10ada: e3 04 e5 60  	beq	a0, a4, 0x118e2 <_vfprintf_r+0x118a>
   10ade: 13 07 10 06  	addi	a4, zero, 97
   10ae2: 63 14 ed 00  	bne	s10, a4, 0x10aea <_vfprintf_r+0x392>
   10ae6: 6f 10 80 4d  	j	0x11fbe <_vfprintf_r+0x1866>
   10aea: 13 07 10 04  	addi	a4, zero, 65
   10aee: 63 14 ed 00  	bne	s10, a4, 0x10af6 <_vfprintf_r+0x39e>
   10af2: 6f 10 80 1e  	j	0x11cda <_vfprintf_r+0x1582>
   10af6: 7d 57        	addi	a4, zero, -1
   10af8: 63 14 e3 00  	bne	t1, a4, 0x10b00 <_vfprintf_r+0x3a8>
   10afc: 6f 10 20 6d  	j	0x121ce <_vfprintf_r+0x1a76>
   10b00: 13 77 fd fd  	andi	a4, s10, -33
   10b04: 93 06 70 04  	addi	a3, zero, 71
   10b08: 63 14 d7 00  	bne	a4, a3, 0x10b10 <_vfprintf_r+0x3b8>
   10b0c: 6f 10 40 6b  	j	0x121c0 <_vfprintf_r+0x1a68>
   10b10: 13 67 09 10  	ori	a4, s2, 256
   10b14: ee 7c        	ld	s9, 248(sp)
   10b16: 9b 07 07 00  	sext.w	a5, a4
   10b1a: 3e e9        	sd	a5, 144(sp)
   10b1c: ca e0        	sd	s2, 64(sp)
   10b1e: 4e 7c        	ld	s8, 240(sp)
   10b20: 63 d4 0c 00  	bgez	s9, 0x10b28 <_vfprintf_r+0x3d0>
   10b24: 6f 10 c0 46  	j	0x11f90 <_vfprintf_r+0x1838>
   10b28: 02 ed        	sd	zero, 152(sp)
   10b2a: 02 ec        	sd	zero, 24(sp)
   10b2c: 1b 07 fd fb  	addiw	a4, s10, -65
   10b30: 1b 06 07 00  	sext.w	a2, a4
   10b34: 93 06 50 02  	addi	a3, zero, 37
   10b38: 63 f4 c6 00  	bgeu	a3, a2, 0x10b40 <_vfprintf_r+0x3e8>
   10b3c: 6f 00 70 79  	j	0x11ad2 <_vfprintf_r+0x137a>
   10b40: 93 16 07 02  	slli	a3, a4, 32
   10b44: 13 d7 e6 01  	srli	a4, a3, 30
   10b48: f9 66        	lui	a3, 30
   10b4a: 93 86 86 90  	addi	a3, a3, -1784
   10b4e: 36 97        	add	a4, a4, a3
   10b50: 18 43        	lw	a4, 0(a4)
   10b52: 02 87        	jr	a4
   10b54: 82 65        	ld	a1, 0(sp)
   10b56: 10 02        	addi	a2, sp, 256
   10b58: 4e 85        	mv	a0, s3
   10b5a: 1a e9        	sd	t1, 144(sp)
   10b5c: ef 80 40 02  	jal	0x18b80 <__sprint_r>
   10b60: 63 1c 05 5e  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   10b64: d2 6b        	ld	s7, 272(sp)
   10b66: 4a 63        	ld	t1, 144(sp)
   10b68: 00 03        	addi	s0, sp, 384
   10b6a: e5 b3        	j	0x10952 <_vfprintf_r+0x1fa>
   10b6c: e2 77        	ld	a5, 56(sp)
   10b6e: 85 46        	addi	a3, zero, 1
   10b70: 93 0c 04 01  	addi	s9, s0, 16
   10b74: e3 d6 f6 06  	bge	a3, a5, 0x113e0 <_vfprintf_r+0xc88>
   10b78: 03 26 81 10  	lw	a2, 264(sp)
   10b7c: 85 46        	addi	a3, zero, 1
   10b7e: 13 87 1b 00  	addi	a4, s7, 1
   10b82: 05 26        	addiw	a2, a2, 1
   10b84: 14 e4        	sd	a3, 8(s0)
   10b86: 23 24 c1 10  	sw	a2, 264(sp)
   10b8a: 9b 06 06 00  	sext.w	a3, a2
   10b8e: 23 30 64 01  	sd	s6, 0(s0)
   10b92: 3a ea        	sd	a4, 272(sp)
   10b94: 1d 46        	addi	a2, zero, 7
   10b96: e3 42 d6 32  	blt	a2, a3, 0x116ba <_vfprintf_r+0xf62>
   10b9a: e6 67        	ld	a5, 88(sp)
   10b9c: 1b 84 16 00  	addiw	s0, a3, 1
   10ba0: 23 24 81 10  	sw	s0, 264(sp)
   10ba4: 33 8d e7 00  	add	s10, a5, a4
   10ba8: 06 77        	ld	a4, 96(sp)
   10baa: 23 b4 fc 00  	sd	a5, 8(s9)
   10bae: 6a ea        	sd	s10, 272(sp)
   10bb0: 23 b0 ec 00  	sd	a4, 0(s9)
   10bb4: 1d 47        	addi	a4, zero, 7
   10bb6: c1 0c        	addi	s9, s9, 16
   10bb8: e3 4f 87 30  	blt	a4, s0, 0x116d6 <_vfprintf_r+0xf7e>
   10bbc: 4e 75        	ld	a0, 240(sp)
   10bbe: ee 75        	ld	a1, 248(sp)
   10bc0: 01 46        	mv	a2, zero
   10bc2: 81 46        	mv	a3, zero
   10bc4: ef a0 90 7a  	jal	0x1bb6c <__netf2>
   10bc8: e2 77        	ld	a5, 56(sp)
   10bca: 9b 8d f7 ff  	addiw	s11, a5, -1
   10bce: e3 0c 05 04  	beqz	a0, 0x11426 <_vfprintf_r+0xcce>
   10bd2: 93 07 1b 00  	addi	a5, s6, 1
   10bd6: 05 24        	addiw	s0, s0, 1
   10bd8: 6e 9d        	add	s10, s10, s11
   10bda: 23 b0 fc 00  	sd	a5, 0(s9)
   10bde: 23 b4 bc 01  	sd	s11, 8(s9)
   10be2: 6a ea        	sd	s10, 272(sp)
   10be4: 23 24 81 10  	sw	s0, 264(sp)
   10be8: 9d 47        	addi	a5, zero, 7
   10bea: e3 c0 87 02  	blt	a5, s0, 0x1140a <_vfprintf_r+0xcb2>
   10bee: c1 0c        	addi	s9, s9, 16
   10bf0: c6 76        	ld	a3, 112(sp)
   10bf2: 05 24        	addiw	s0, s0, 1
   10bf4: 9c 11        	addi	a5, sp, 224
   10bf6: 33 87 a6 01  	add	a4, a3, s10
   10bfa: 23 b0 fc 00  	sd	a5, 0(s9)
   10bfe: 23 b4 dc 00  	sd	a3, 8(s9)
   10c02: 23 24 81 10  	sw	s0, 264(sp)
   10c06: 9b 06 04 00  	sext.w	a3, s0
   10c0a: 3a ea        	sd	a4, 272(sp)
   10c0c: 9d 47        	addi	a5, zero, 7
   10c0e: 13 84 0c 01  	addi	s0, s9, 16
   10c12: e3 d0 d7 da  	bge	a5, a3, 0x109b2 <_vfprintf_r+0x25a>
   10c16: 82 65        	ld	a1, 0(sp)
   10c18: 10 02        	addi	a2, sp, 256
   10c1a: 4e 85        	mv	a0, s3
   10c1c: ef 70 50 76  	jal	0x18b80 <__sprint_r>
   10c20: 63 1c 05 52  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   10c24: 52 67        	ld	a4, 272(sp)
   10c26: 00 03        	addi	s0, sp, 384
   10c28: 69 b3        	j	0x109b2 <_vfprintf_r+0x25a>
   10c2a: 46 65        	ld	a0, 80(sp)
   10c2c: ef 10 50 53  	jal	0x12960 <__sinit>
   10c30: 59 b6        	j	0x107b6 <_vfprintf_r+0x5e>
   10c32: 02 77        	ld	a4, 32(sp)
   10c34: a3 07 01 0c  	sb	zero, 207(sp)
   10c38: 05 4c        	addi	s8, zero, 1
   10c3a: 1c 43        	lw	a5, 0(a4)
   10c3c: 21 07        	addi	a4, a4, 8
   10c3e: 3a f0        	sd	a4, 32(sp)
   10c40: 23 0c f1 10  	sb	a5, 280(sp)
   10c44: 85 4d        	addi	s11, zero, 1
   10c46: 13 0b 81 11  	addi	s6, sp, 280
   10c4a: 65 b9        	j	0x10902 <_vfprintf_r+0x1aa>
   10c4c: 82 77        	ld	a5, 32(sp)
   10c4e: a3 07 01 0c  	sb	zero, 207(sp)
   10c52: 56 83        	mv	t1, s5
   10c54: 03 bb 07 00  	ld	s6, 0(a5)
   10c58: 93 8a 87 00  	addi	s5, a5, 8
   10c5c: e3 02 0b 1e  	beqz	s6, 0x11640 <_vfprintf_r+0xee8>
   10c60: 7d 57        	addi	a4, zero, -1
   10c62: e3 05 e3 42  	beq	t1, a4, 0x1188c <_vfprintf_r+0x1134>
   10c66: 1a 86        	mv	a2, t1
   10c68: 81 45        	mv	a1, zero
   10c6a: 5a 85        	mv	a0, s6
   10c6c: 1a f0        	sd	t1, 32(sp)
   10c6e: ef 50 80 0e  	jal	0x15d56 <memchr>
   10c72: 2a ec        	sd	a0, 24(sp)
   10c74: 02 73        	ld	t1, 32(sp)
   10c76: 19 e1        	bnez	a0, 0x10c7c <_vfprintf_r+0x524>
   10c78: 6f 10 40 3a  	j	0x1201c <_vfprintf_r+0x18c4>
   10c7c: e2 67        	ld	a5, 24(sp)
   10c7e: bb 8d 67 41  	subw	s11, a5, s6
   10c82: 03 47 f1 0c  	lbu	a4, 207(sp)
   10c86: 13 c8 fd ff  	not	a6, s11
   10c8a: 13 58 f8 43  	srai	a6, a6, 63
   10c8e: 56 f0        	sd	s5, 32(sp)
   10c90: 02 ec        	sd	zero, 24(sp)
   10c92: 82 e4        	sd	zero, 72(sp)
   10c94: 82 e0        	sd	zero, 64(sp)
   10c96: 02 f8        	sd	zero, 48(sp)
   10c98: 33 7c b8 01  	and	s8, a6, s11
   10c9c: 01 43        	mv	t1, zero
   10c9e: e3 07 07 c6  	beqz	a4, 0x1090c <_vfprintf_r+0x1b4>
   10ca2: 05 2c        	addiw	s8, s8, 1
   10ca4: a5 b1        	j	0x1090c <_vfprintf_r+0x1b4>
   10ca6: 82 77        	ld	a5, 32(sp)
   10ca8: 03 aa 07 00  	lw	s4, 0(a5)
   10cac: a1 07        	addi	a5, a5, 8
   10cae: e3 55 0a 06  	bgez	s4, 0x11518 <_vfprintf_r+0xdc0>
   10cb2: 3b 0a 40 41  	negw	s4, s4
   10cb6: 3e f0        	sd	a5, 32(sp)
   10cb8: 13 69 49 00  	ori	s2, s2, 4
   10cbc: 03 ce 04 00  	lbu	t3, 0(s1)
   10cc0: 01 29        	sext.w	s2, s2
   10cc2: d1 be        	j	0x10896 <_vfprintf_r+0x13e>
   10cc4: 93 07 b0 02  	addi	a5, zero, 43
   10cc8: 03 ce 04 00  	lbu	t3, 0(s1)
   10ccc: a3 07 f1 0c  	sb	a5, 207(sp)
   10cd0: d9 b6        	j	0x10896 <_vfprintf_r+0x13e>
   10cd2: 13 69 09 08  	ori	s2, s2, 128
   10cd6: 03 ce 04 00  	lbu	t3, 0(s1)
   10cda: 01 29        	sext.w	s2, s2
   10cdc: 6d be        	j	0x10896 <_vfprintf_r+0x13e>
   10cde: 03 cd 04 00  	lbu	s10, 0(s1)
   10ce2: 93 06 a0 02  	addi	a3, zero, 42
   10ce6: 93 87 14 00  	addi	a5, s1, 1
   10cea: 63 14 dd 00  	bne	s10, a3, 0x10cf2 <_vfprintf_r+0x59a>
   10cee: 6f 10 60 5a  	j	0x12294 <_vfprintf_r+0x1b3c>
   10cf2: 1b 07 0d fd  	addiw	a4, s10, -48
   10cf6: 25 46        	addi	a2, zero, 9
   10cf8: be 84        	mv	s1, a5
   10cfa: 81 4a        	mv	s5, zero
   10cfc: a5 46        	addi	a3, zero, 9
   10cfe: e3 6f e6 b8  	bltu	a2, a4, 0x1089c <_vfprintf_r+0x144>
   10d02: 03 cd 04 00  	lbu	s10, 0(s1)
   10d06: 9b 97 2a 00  	slliw	a5, s5, 2
   10d0a: 3b 83 57 01  	addw	t1, a5, s5
   10d0e: 1b 13 13 00  	slliw	t1, t1, 1
   10d12: bb 0a e3 00  	addw	s5, t1, a4
   10d16: 1b 07 0d fd  	addiw	a4, s10, -48
   10d1a: 85 04        	addi	s1, s1, 1
   10d1c: e3 f3 e6 fe  	bgeu	a3, a4, 0x10d02 <_vfprintf_r+0x5aa>
   10d20: b5 be        	j	0x1089c <_vfprintf_r+0x144>
   10d22: 4e 85        	mv	a0, s3
   10d24: ef 40 40 7a  	jal	0x154c8 <_localeconv_r>
   10d28: 1c 65        	ld	a5, 8(a0)
   10d2a: 3e 85        	mv	a0, a5
   10d2c: be fc        	sd	a5, 120(sp)
   10d2e: ef 60 20 15  	jal	0x16e80 <strlen>
   10d32: aa 87        	mv	a5, a0
   10d34: 4e 85        	mv	a0, s3
   10d36: 3e 8c        	mv	s8, a5
   10d38: 3e e5        	sd	a5, 136(sp)
   10d3a: ef 40 e0 78  	jal	0x154c8 <_localeconv_r>
   10d3e: 1c 69        	ld	a5, 16(a0)
   10d40: 03 ce 04 00  	lbu	t3, 0(s1)
   10d44: 3e e1        	sd	a5, 128(sp)
   10d46: e3 08 0c b4  	beqz	s8, 0x10896 <_vfprintf_r+0x13e>
   10d4a: e3 86 07 b4  	beqz	a5, 0x10896 <_vfprintf_r+0x13e>
   10d4e: 83 c7 07 00  	lbu	a5, 0(a5)
   10d52: e3 82 07 b4  	beqz	a5, 0x10896 <_vfprintf_r+0x13e>
   10d56: 13 69 09 40  	ori	s2, s2, 1024
   10d5a: 01 29        	sext.w	s2, s2
   10d5c: 2d be        	j	0x10896 <_vfprintf_r+0x13e>
   10d5e: 13 69 19 00  	ori	s2, s2, 1
   10d62: 03 ce 04 00  	lbu	t3, 0(s1)
   10d66: 01 29        	sext.w	s2, s2
   10d68: 3d b6        	j	0x10896 <_vfprintf_r+0x13e>
   10d6a: 83 47 f1 0c  	lbu	a5, 207(sp)
   10d6e: 03 ce 04 00  	lbu	t3, 0(s1)
   10d72: e3 92 07 b2  	bnez	a5, 0x10896 <_vfprintf_r+0x13e>
   10d76: 93 07 00 02  	addi	a5, zero, 32
   10d7a: a3 07 f1 0c  	sb	a5, 207(sp)
   10d7e: 21 be        	j	0x10896 <_vfprintf_r+0x13e>
   10d80: 93 6c 09 01  	ori	s9, s2, 16
   10d84: 56 83        	mv	t1, s5
   10d86: 81 2c        	sext.w	s9, s9
   10d88: 82 76        	ld	a3, 32(sp)
   10d8a: 93 f7 0c 02  	andi	a5, s9, 32
   10d8e: 13 87 86 00  	addi	a4, a3, 8
   10d92: 63 87 07 74  	beqz	a5, 0x114e0 <_vfprintf_r+0xd88>
   10d96: 83 bb 06 00  	ld	s7, 0(a3)
   10d9a: 85 47        	addi	a5, zero, 1
   10d9c: 3a f0        	sd	a4, 32(sp)
   10d9e: 01 47        	mv	a4, zero
   10da0: a3 07 e1 0c  	sb	a4, 207(sp)
   10da4: 7d 57        	addi	a4, zero, -1
   10da6: 63 0a e3 1e  	beq	t1, a4, 0x10f9a <_vfprintf_r+0x842>
   10daa: 13 f9 fc f7  	andi	s2, s9, -129
   10dae: 01 29        	sext.w	s2, s2
   10db0: 63 94 0b 1e  	bnez	s7, 0x10f98 <_vfprintf_r+0x840>
   10db4: 63 17 03 22  	bnez	t1, 0x10fe2 <_vfprintf_r+0x88a>
   10db8: 63 94 07 74  	bnez	a5, 0x11500 <_vfprintf_r+0xda8>
   10dbc: 93 fd 1c 00  	andi	s11, s9, 1
   10dc0: 13 0b c1 17  	addi	s6, sp, 380
   10dc4: 63 9f 0d 74  	bnez	s11, 0x11522 <_vfprintf_r+0xdca>
   10dc8: 03 47 f1 0c  	lbu	a4, 207(sp)
   10dcc: 1a 88        	mv	a6, t1
   10dce: 63 53 b3 01  	bge	t1, s11, 0x10dd4 <_vfprintf_r+0x67c>
   10dd2: 6e 88        	mv	a6, s11
   10dd4: 02 ec        	sd	zero, 24(sp)
   10dd6: 82 e4        	sd	zero, 72(sp)
   10dd8: 82 e0        	sd	zero, 64(sp)
   10dda: 02 f8        	sd	zero, 48(sp)
   10ddc: 1b 0c 08 00  	sext.w	s8, a6
   10de0: e3 11 07 ec  	bnez	a4, 0x10ca2 <_vfprintf_r+0x54a>
   10de4: 25 b6        	j	0x1090c <_vfprintf_r+0x1b4>
   10de6: 13 69 89 00  	ori	s2, s2, 8
   10dea: 03 ce 04 00  	lbu	t3, 0(s1)
   10dee: 01 29        	sext.w	s2, s2
   10df0: 5d b4        	j	0x10896 <_vfprintf_r+0x13e>
   10df2: 13 69 09 01  	ori	s2, s2, 16
   10df6: 56 83        	mv	t1, s5
   10df8: 01 29        	sext.w	s2, s2
   10dfa: 82 76        	ld	a3, 32(sp)
   10dfc: 93 77 09 02  	andi	a5, s2, 32
   10e00: 13 87 86 00  	addi	a4, a3, 8
   10e04: 63 87 07 68  	beqz	a5, 0x11492 <_vfprintf_r+0xd3a>
   10e08: 9c 62        	ld	a5, 0(a3)
   10e0a: be 8b        	mv	s7, a5
   10e0c: 63 c4 07 6a  	bltz	a5, 0x114b4 <_vfprintf_r+0xd5c>
   10e10: fd 57        	addi	a5, zero, -1
   10e12: 63 08 f3 3e  	beq	t1, a5, 0x11202 <_vfprintf_r+0xaaa>
   10e16: 93 7c f9 f7  	andi	s9, s2, -129
   10e1a: 81 2c        	sext.w	s9, s9
   10e1c: e3 97 0b 1c  	bnez	s7, 0x117ea <_vfprintf_r+0x1092>
   10e20: 63 04 03 00  	beqz	t1, 0x10e28 <_vfprintf_r+0x6d0>
   10e24: 6f 10 00 37  	j	0x12194 <_vfprintf_r+0x1a3c>
   10e28: 81 4d        	mv	s11, zero
   10e2a: 3a f0        	sd	a4, 32(sp)
   10e2c: 66 89        	mv	s2, s9
   10e2e: 13 0b c1 17  	addi	s6, sp, 380
   10e32: 59 bf        	j	0x10dc8 <_vfprintf_r+0x670>
   10e34: 13 69 09 01  	ori	s2, s2, 16
   10e38: 56 83        	mv	t1, s5
   10e3a: 01 29        	sext.w	s2, s2
   10e3c: 82 76        	ld	a3, 32(sp)
   10e3e: 93 77 09 02  	andi	a5, s2, 32
   10e42: 13 87 86 00  	addi	a4, a3, 8
   10e46: 63 8f 07 66  	beqz	a5, 0x114c4 <_vfprintf_r+0xd6c>
   10e4a: 83 bb 06 00  	ld	s7, 0(a3)
   10e4e: 93 7c f9 bf  	andi	s9, s2, -1025
   10e52: 81 2c        	sext.w	s9, s9
   10e54: 3a f0        	sd	a4, 32(sp)
   10e56: 81 47        	mv	a5, zero
   10e58: 99 b7        	j	0x10d9e <_vfprintf_r+0x646>
   10e5a: 13 69 09 02  	ori	s2, s2, 32
   10e5e: 03 ce 04 00  	lbu	t3, 0(s1)
   10e62: 01 29        	sext.w	s2, s2
   10e64: 0d bc        	j	0x10896 <_vfprintf_r+0x13e>
   10e66: 02 77        	ld	a4, 32(sp)
   10e68: 93 77 09 02  	andi	a5, s2, 32
   10e6c: 18 63        	ld	a4, 0(a4)
   10e6e: 63 8f 07 6c  	beqz	a5, 0x1154c <_vfprintf_r+0xdf4>
   10e72: c2 67        	ld	a5, 16(sp)
   10e74: 1c e3        	sd	a5, 0(a4)
   10e76: 82 77        	ld	a5, 32(sp)
   10e78: 26 8b        	mv	s6, s1
   10e7a: a1 07        	addi	a5, a5, 8
   10e7c: 3e f0        	sd	a5, 32(sp)
   10e7e: ad b6        	j	0x109e8 <_vfprintf_r+0x290>
   10e80: 03 ce 04 00  	lbu	t3, 0(s1)
   10e84: 93 07 c0 06  	addi	a5, zero, 108
   10e88: 63 04 fe 7a  	beq	t3, a5, 0x11630 <_vfprintf_r+0xed8>
   10e8c: 13 69 09 01  	ori	s2, s2, 16
   10e90: 01 29        	sext.w	s2, s2
   10e92: 11 b4        	j	0x10896 <_vfprintf_r+0x13e>
   10e94: 03 ce 04 00  	lbu	t3, 0(s1)
   10e98: 93 07 80 06  	addi	a5, zero, 104
   10e9c: 63 0f fe 76  	beq	t3, a5, 0x1161a <_vfprintf_r+0xec2>
   10ea0: 13 69 09 04  	ori	s2, s2, 64
   10ea4: 01 29        	sext.w	s2, s2
   10ea6: c5 ba        	j	0x10896 <_vfprintf_r+0x13e>
   10ea8: 02 77        	ld	a4, 32(sp)
   10eaa: e1 77        	lui	a5, 1048568
   10eac: 93 c7 07 83  	xori	a5, a5, -2000
   10eb0: 23 18 f1 0c  	sh	a5, 208(sp)
   10eb4: 93 07 87 00  	addi	a5, a4, 8
   10eb8: 3e f0        	sd	a5, 32(sp)
   10eba: f5 67        	lui	a5, 29
   10ebc: 93 87 87 77  	addi	a5, a5, 1912
   10ec0: 93 6c 29 00  	ori	s9, s2, 2
   10ec4: be f4        	sd	a5, 104(sp)
   10ec6: 83 3b 07 00  	ld	s7, 0(a4)
   10eca: 56 83        	mv	t1, s5
   10ecc: 81 2c        	sext.w	s9, s9
   10ece: 89 47        	addi	a5, zero, 2
   10ed0: 13 0d 80 07  	addi	s10, zero, 120
   10ed4: e9 b5        	j	0x10d9e <_vfprintf_r+0x646>
   10ed6: 93 06 50 06  	addi	a3, zero, 101
   10eda: e3 d9 a6 c9  	bge	a3, s10, 0x10b6c <_vfprintf_r+0x414>
   10ede: 4e 75        	ld	a0, 240(sp)
   10ee0: ee 75        	ld	a1, 248(sp)
   10ee2: 01 46        	mv	a2, zero
   10ee4: 81 46        	mv	a3, zero
   10ee6: ef a0 70 48  	jal	0x1bb6c <__netf2>
   10eea: 63 16 05 3c  	bnez	a0, 0x112b6 <_vfprintf_r+0xb5e>
   10eee: 83 27 81 10  	lw	a5, 264(sp)
   10ef2: f5 66        	lui	a3, 29
   10ef4: 93 86 86 79  	addi	a3, a3, 1944
   10ef8: 85 27        	addiw	a5, a5, 1
   10efa: 14 e0        	sd	a3, 0(s0)
   10efc: 85 0b        	addi	s7, s7, 1
   10efe: 85 46        	addi	a3, zero, 1
   10f00: 14 e4        	sd	a3, 8(s0)
   10f02: 23 24 f1 10  	sw	a5, 264(sp)
   10f06: 9b 86 07 00  	sext.w	a3, a5
   10f0a: 5e ea        	sd	s7, 272(sp)
   10f0c: 9d 47        	addi	a5, zero, 7
   10f0e: 41 04        	addi	s0, s0, 16
   10f10: e3 c3 d7 0a  	blt	a5, a3, 0x117b6 <_vfprintf_r+0x105e>
   10f14: ee 47        	lw	a5, 216(sp)
   10f16: 62 77        	ld	a4, 56(sp)
   10f18: 63 d8 e7 5e  	bge	a5, a4, 0x11508 <_vfprintf_r+0xdb0>
   10f1c: e6 66        	ld	a3, 88(sp)
   10f1e: 83 27 81 10  	lw	a5, 264(sp)
   10f22: 06 76        	ld	a2, 96(sp)
   10f24: 33 87 76 01  	add	a4, a3, s7
   10f28: 85 27        	addiw	a5, a5, 1
   10f2a: 14 e4        	sd	a3, 8(s0)
   10f2c: 10 e0        	sd	a2, 0(s0)
   10f2e: 9b 86 07 00  	sext.w	a3, a5
   10f32: 23 24 f1 10  	sw	a5, 264(sp)
   10f36: 3a ea        	sd	a4, 272(sp)
   10f38: 9d 47        	addi	a5, zero, 7
   10f3a: 41 04        	addi	s0, s0, 16
   10f3c: 63 c5 d7 6c  	blt	a5, a3, 0x11606 <_vfprintf_r+0xeae>
   10f40: e2 77        	ld	a5, 56(sp)
   10f42: 9b 8c f7 ff  	addiw	s9, a5, -1
   10f46: e3 56 90 a7  	blez	s9, 0x109b2 <_vfprintf_r+0x25a>
   10f4a: 79 6f        	lui	t5, 30
   10f4c: c1 46        	addi	a3, zero, 16
   10f4e: 83 27 81 10  	lw	a5, 264(sp)
   10f52: 93 0b 0f 9a  	addi	s7, t5, -1632
   10f56: e3 db 96 07  	bge	a3, s9, 0x117cc <_vfprintf_r+0x1074>
   10f5a: 82 6a        	ld	s5, 0(sp)
   10f5c: 41 4d        	addi	s10, zero, 16
   10f5e: 9d 4d        	addi	s11, zero, 7
   10f60: 21 a0        	j	0x10f68 <_vfprintf_r+0x810>
   10f62: c1 3c        	addiw	s9, s9, -16
   10f64: e3 54 9d 07  	bge	s10, s9, 0x117cc <_vfprintf_r+0x1074>
   10f68: 85 27        	addiw	a5, a5, 1
   10f6a: 41 07        	addi	a4, a4, 16
   10f6c: 23 30 74 01  	sd	s7, 0(s0)
   10f70: 23 34 a4 01  	sd	s10, 8(s0)
   10f74: 3a ea        	sd	a4, 272(sp)
   10f76: 23 24 f1 10  	sw	a5, 264(sp)
   10f7a: 41 04        	addi	s0, s0, 16
   10f7c: e3 d3 fd fe  	bge	s11, a5, 0x10f62 <_vfprintf_r+0x80a>
   10f80: 10 02        	addi	a2, sp, 256
   10f82: d6 85        	mv	a1, s5
   10f84: 4e 85        	mv	a0, s3
   10f86: ef 70 b0 3f  	jal	0x18b80 <__sprint_r>
   10f8a: 63 17 05 1c  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   10f8e: 52 67        	ld	a4, 272(sp)
   10f90: 83 27 81 10  	lw	a5, 264(sp)
   10f94: 00 03        	addi	s0, sp, 384
   10f96: f1 b7        	j	0x10f62 <_vfprintf_r+0x80a>
   10f98: ca 8c        	mv	s9, s2
   10f9a: 05 47        	addi	a4, zero, 1
   10f9c: 63 85 e7 26  	beq	a5, a4, 0x11206 <_vfprintf_r+0xaae>
   10fa0: 09 47        	addi	a4, zero, 2
   10fa2: 63 89 e7 04  	beq	a5, a4, 0x10ff4 <_vfprintf_r+0x89c>
   10fa6: 93 0a c1 17  	addi	s5, sp, 380
   10faa: 56 8b        	mv	s6, s5
   10fac: 13 f7 7b 00  	andi	a4, s7, 7
   10fb0: 13 07 07 03  	addi	a4, a4, 48
   10fb4: a3 0f eb fe  	sb	a4, -1(s6)
   10fb8: 93 db 3b 00  	srli	s7, s7, 3
   10fbc: 5a 86        	mv	a2, s6
   10fbe: 7d 1b        	addi	s6, s6, -1
   10fc0: e3 96 0b fe  	bnez	s7, 0x10fac <_vfprintf_r+0x854>
   10fc4: 93 f5 1c 00  	andi	a1, s9, 1
   10fc8: b1 c5        	beqz	a1, 0x11014 <_vfprintf_r+0x8bc>
   10fca: 93 05 00 03  	addi	a1, zero, 48
   10fce: 63 03 b7 04  	beq	a4, a1, 0x11014 <_vfprintf_r+0x8bc>
   10fd2: 79 16        	addi	a2, a2, -2
   10fd4: a3 0f bb fe  	sb	a1, -1(s6)
   10fd8: bb 8d ca 40  	subw	s11, s5, a2
   10fdc: 66 89        	mv	s2, s9
   10fde: 32 8b        	mv	s6, a2
   10fe0: e5 b3        	j	0x10dc8 <_vfprintf_r+0x670>
   10fe2: 05 47        	addi	a4, zero, 1
   10fe4: 63 94 e7 00  	bne	a5, a4, 0x10fec <_vfprintf_r+0x894>
   10fe8: 6f 10 80 07  	j	0x12060 <_vfprintf_r+0x1908>
   10fec: 09 47        	addi	a4, zero, 2
   10fee: ca 8c        	mv	s9, s2
   10ff0: e3 9b e7 fa  	bne	a5, a4, 0x10fa6 <_vfprintf_r+0x84e>
   10ff4: 26 77        	ld	a4, 104(sp)
   10ff6: 93 0a c1 17  	addi	s5, sp, 380
   10ffa: 56 8b        	mv	s6, s5
   10ffc: 93 f7 fb 00  	andi	a5, s7, 15
   11000: ba 97        	add	a5, a5, a4
   11002: 83 c7 07 00  	lbu	a5, 0(a5)
   11006: 93 db 4b 00  	srli	s7, s7, 4
   1100a: 7d 1b        	addi	s6, s6, -1
   1100c: 23 00 fb 00  	sb	a5, 0(s6)
   11010: e3 96 0b fe  	bnez	s7, 0x10ffc <_vfprintf_r+0x8a4>
   11014: bb 8d 6a 41  	subw	s11, s5, s6
   11018: 66 89        	mv	s2, s9
   1101a: 7d b3        	j	0x10dc8 <_vfprintf_r+0x670>
   1101c: bb 0a 8a 41  	subw	s5, s4, s8
   11020: e3 50 50 97  	blez	s5, 0x10980 <_vfprintf_r+0x228>
   11024: 79 6f        	lui	t5, 30
   11026: c1 45        	addi	a1, zero, 16
   11028: 5e 87        	mv	a4, s7
   1102a: 03 26 81 10  	lw	a2, 264(sp)
   1102e: 93 0b 0f 9a  	addi	s7, t5, -1632
   11032: 63 d0 55 07  	bge	a1, s5, 0x11092 <_vfprintf_r+0x93a>
   11036: 26 e9        	sd	s1, 144(sp)
   11038: c1 4f        	addi	t6, zero, 16
   1103a: d6 84        	mv	s1, s5
   1103c: 9d 4c        	addi	s9, zero, 7
   1103e: d2 8a        	mv	s5, s4
   11040: 4a 8a        	mv	s4, s2
   11042: 5e 89        	mv	s2, s7
   11044: da 8b        	mv	s7, s6
   11046: 1a 8b        	mv	s6, t1
   11048: 21 a0        	j	0x11050 <_vfprintf_r+0x8f8>
   1104a: c1 34        	addiw	s1, s1, -16
   1104c: 63 dc 9f 02  	bge	t6, s1, 0x11084 <_vfprintf_r+0x92c>
   11050: 05 26        	addiw	a2, a2, 1
   11052: 41 07        	addi	a4, a4, 16
   11054: 23 30 24 01  	sd	s2, 0(s0)
   11058: 23 34 f4 01  	sd	t6, 8(s0)
   1105c: 3a ea        	sd	a4, 272(sp)
   1105e: 23 24 c1 10  	sw	a2, 264(sp)
   11062: 41 04        	addi	s0, s0, 16
   11064: e3 d3 cc fe  	bge	s9, a2, 0x1104a <_vfprintf_r+0x8f2>
   11068: 82 65        	ld	a1, 0(sp)
   1106a: 10 02        	addi	a2, sp, 256
   1106c: 4e 85        	mv	a0, s3
   1106e: ef 70 30 31  	jal	0x18b80 <__sprint_r>
   11072: 7d e1        	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11074: c1 4f        	addi	t6, zero, 16
   11076: c1 34        	addiw	s1, s1, -16
   11078: 52 67        	ld	a4, 272(sp)
   1107a: 03 26 81 10  	lw	a2, 264(sp)
   1107e: 00 03        	addi	s0, sp, 384
   11080: e3 c8 9f fc  	blt	t6, s1, 0x11050 <_vfprintf_r+0x8f8>
   11084: 5a 83        	mv	t1, s6
   11086: 5e 8b        	mv	s6, s7
   11088: ca 8b        	mv	s7, s2
   1108a: 52 89        	mv	s2, s4
   1108c: 56 8a        	mv	s4, s5
   1108e: a6 8a        	mv	s5, s1
   11090: ca 64        	ld	s1, 144(sp)
   11092: 9b 05 16 00  	addiw	a1, a2, 1
   11096: 56 97        	add	a4, a4, s5
   11098: 23 30 74 01  	sd	s7, 0(s0)
   1109c: 23 34 54 01  	sd	s5, 8(s0)
   110a0: 3a ea        	sd	a4, 272(sp)
   110a2: 23 24 b1 10  	sw	a1, 264(sp)
   110a6: 1d 46        	addi	a2, zero, 7
   110a8: 63 46 b6 4e  	blt	a2, a1, 0x11594 <_vfprintf_r+0xe3c>
   110ac: bb 0c b3 41  	subw	s9, t1, s11
   110b0: 41 04        	addi	s0, s0, 16
   110b2: ba 8b        	mv	s7, a4
   110b4: e3 5a 90 8d  	blez	s9, 0x10988 <_vfprintf_r+0x230>
   110b8: 79 6f        	lui	t5, 30
   110ba: c1 45        	addi	a1, zero, 16
   110bc: 5e 87        	mv	a4, s7
   110be: 03 26 81 10  	lw	a2, 264(sp)
   110c2: 93 0b 0f 9a  	addi	s7, t5, -1632
   110c6: 63 d1 95 07  	bge	a1, s9, 0x11128 <_vfprintf_r+0x9d0>
   110ca: a2 6b        	ld	s7, 8(sp)
   110cc: c1 48        	addi	a7, zero, 16
   110ce: 9d 4a        	addi	s5, zero, 7
   110d0: de 87        	mv	a5, s7
   110d2: da 8b        	mv	s7, s6
   110d4: 52 8b        	mv	s6, s4
   110d6: 4a 8a        	mv	s4, s2
   110d8: 26 89        	mv	s2, s1
   110da: e6 84        	mv	s1, s9
   110dc: be 8c        	mv	s9, a5
   110de: 21 a0        	j	0x110e6 <_vfprintf_r+0x98e>
   110e0: c1 34        	addiw	s1, s1, -16
   110e2: 63 dc 98 02  	bge	a7, s1, 0x1111a <_vfprintf_r+0x9c2>
   110e6: a2 66        	ld	a3, 8(sp)
   110e8: 05 26        	addiw	a2, a2, 1
   110ea: 41 07        	addi	a4, a4, 16
   110ec: 14 e0        	sd	a3, 0(s0)
   110ee: 23 34 14 01  	sd	a7, 8(s0)
   110f2: 3a ea        	sd	a4, 272(sp)
   110f4: 23 24 c1 10  	sw	a2, 264(sp)
   110f8: 41 04        	addi	s0, s0, 16
   110fa: e3 d3 ca fe  	bge	s5, a2, 0x110e0 <_vfprintf_r+0x988>
   110fe: 82 65        	ld	a1, 0(sp)
   11100: 10 02        	addi	a2, sp, 256
   11102: 4e 85        	mv	a0, s3
   11104: ef 70 d0 27  	jal	0x18b80 <__sprint_r>
   11108: 21 e9        	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   1110a: c1 48        	addi	a7, zero, 16
   1110c: c1 34        	addiw	s1, s1, -16
   1110e: 52 67        	ld	a4, 272(sp)
   11110: 03 26 81 10  	lw	a2, 264(sp)
   11114: 00 03        	addi	s0, sp, 384
   11116: e3 c8 98 fc  	blt	a7, s1, 0x110e6 <_vfprintf_r+0x98e>
   1111a: e6 87        	mv	a5, s9
   1111c: a6 8c        	mv	s9, s1
   1111e: ca 84        	mv	s1, s2
   11120: 52 89        	mv	s2, s4
   11122: 5a 8a        	mv	s4, s6
   11124: 5e 8b        	mv	s6, s7
   11126: be 8b        	mv	s7, a5
   11128: 9b 05 16 00  	addiw	a1, a2, 1
   1112c: 66 97        	add	a4, a4, s9
   1112e: 23 30 74 01  	sd	s7, 0(s0)
   11132: 23 34 94 01  	sd	s9, 8(s0)
   11136: 3a ea        	sd	a4, 272(sp)
   11138: 23 24 b1 10  	sw	a1, 264(sp)
   1113c: 1d 46        	addi	a2, zero, 7
   1113e: 63 4f b6 32  	blt	a2, a1, 0x1147c <_vfprintf_r+0xd24>
   11142: 41 04        	addi	s0, s0, 16
   11144: ba 8b        	mv	s7, a4
   11146: 6f f0 3f 84  	j	0x10988 <_vfprintf_r+0x230>
   1114a: 82 65        	ld	a1, 0(sp)
   1114c: 10 02        	addi	a2, sp, 256
   1114e: 4e 85        	mv	a0, s3
   11150: ef 70 10 23  	jal	0x18b80 <__sprint_r>
   11154: e3 00 05 88  	beqz	a0, 0x109d4 <_vfprintf_r+0x27c>
   11158: e2 65        	ld	a1, 24(sp)
   1115a: e3 8f 05 88  	beqz	a1, 0x109f8 <_vfprintf_r+0x2a0>
   1115e: 46 65        	ld	a0, 80(sp)
   11160: ef 10 50 13  	jal	0x12a94 <_free_r>
   11164: 51 b8        	j	0x109f8 <_vfprintf_r+0x2a0>
   11166: f9 68        	lui	a7, 30
   11168: 41 45        	addi	a0, zero, 16
   1116a: d2 65        	ld	a1, 272(sp)
   1116c: 03 26 81 10  	lw	a2, 264(sp)
   11170: 93 8b 08 9b  	addi	s7, a7, -1616
   11174: 63 5e e5 04  	bge	a0, a4, 0x111d0 <_vfprintf_r+0xa78>
   11178: 26 e9        	sd	s1, 144(sp)
   1117a: a2 87        	mv	a5, s0
   1117c: de 84        	mv	s1, s7
   1117e: 41 4f        	addi	t5, zero, 16
   11180: da 8b        	mv	s7, s6
   11182: 9d 43        	addi	t2, zero, 7
   11184: 3a 84        	mv	s0, a4
   11186: 1a 8b        	mv	s6, t1
   11188: 21 a0        	j	0x11190 <_vfprintf_r+0xa38>
   1118a: 41 34        	addiw	s0, s0, -16
   1118c: 63 5c 8f 02  	bge	t5, s0, 0x111c4 <_vfprintf_r+0xa6c>
   11190: 05 26        	addiw	a2, a2, 1
   11192: c1 05        	addi	a1, a1, 16
   11194: 84 e3        	sd	s1, 0(a5)
   11196: 23 b4 e7 01  	sd	t5, 8(a5)
   1119a: 2e ea        	sd	a1, 272(sp)
   1119c: 23 24 c1 10  	sw	a2, 264(sp)
   111a0: c1 07        	addi	a5, a5, 16
   111a2: e3 d4 c3 fe  	bge	t2, a2, 0x1118a <_vfprintf_r+0xa32>
   111a6: 82 65        	ld	a1, 0(sp)
   111a8: 10 02        	addi	a2, sp, 256
   111aa: 4e 85        	mv	a0, s3
   111ac: ef 70 50 1d  	jal	0x18b80 <__sprint_r>
   111b0: 45 f5        	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   111b2: 41 4f        	addi	t5, zero, 16
   111b4: 41 34        	addiw	s0, s0, -16
   111b6: d2 65        	ld	a1, 272(sp)
   111b8: 03 26 81 10  	lw	a2, 264(sp)
   111bc: 1c 03        	addi	a5, sp, 384
   111be: 9d 43        	addi	t2, zero, 7
   111c0: e3 48 8f fc  	blt	t5, s0, 0x11190 <_vfprintf_r+0xa38>
   111c4: 5a 83        	mv	t1, s6
   111c6: 5e 8b        	mv	s6, s7
   111c8: a6 8b        	mv	s7, s1
   111ca: ca 64        	ld	s1, 144(sp)
   111cc: 22 87        	mv	a4, s0
   111ce: 3e 84        	mv	s0, a5
   111d0: ba 95        	add	a1, a1, a4
   111d2: 05 26        	addiw	a2, a2, 1
   111d4: 18 e4        	sd	a4, 8(s0)
   111d6: 23 30 74 01  	sd	s7, 0(s0)
   111da: 2e ea        	sd	a1, 272(sp)
   111dc: 23 24 c1 10  	sw	a2, 264(sp)
   111e0: 9b 05 06 00  	sext.w	a1, a2
   111e4: 1d 47        	addi	a4, zero, 7
   111e6: 41 04        	addi	s0, s0, 16
   111e8: 63 5f b7 f2  	bge	a4, a1, 0x10926 <_vfprintf_r+0x1ce>
   111ec: 82 65        	ld	a1, 0(sp)
   111ee: 10 02        	addi	a2, sp, 256
   111f0: 4e 85        	mv	a0, s3
   111f2: 1a e9        	sd	t1, 144(sp)
   111f4: ef 70 d0 18  	jal	0x18b80 <__sprint_r>
   111f8: 25 f1        	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   111fa: 4a 63        	ld	t1, 144(sp)
   111fc: 00 03        	addi	s0, sp, 384
   111fe: 6f f0 8f f2  	j	0x10926 <_vfprintf_r+0x1ce>
   11202: ca 8c        	mv	s9, s2
   11204: 3a f0        	sd	a4, 32(sp)
   11206: a5 47        	addi	a5, zero, 9
   11208: 63 e5 77 4f  	bltu	a5, s7, 0x116f2 <_vfprintf_r+0xf9a>
   1120c: 9b 8b 0b 03  	addiw	s7, s7, 48
   11210: a3 0d 71 17  	sb	s7, 379(sp)
   11214: 66 89        	mv	s2, s9
   11216: 85 4d        	addi	s11, zero, 1
   11218: 13 0b b1 17  	addi	s6, sp, 379
   1121c: 75 b6        	j	0x10dc8 <_vfprintf_r+0x670>
   1121e: 82 65        	ld	a1, 0(sp)
   11220: 10 02        	addi	a2, sp, 256
   11222: 4e 85        	mv	a0, s3
   11224: 1a e9        	sd	t1, 144(sp)
   11226: ef 70 b0 15  	jal	0x18b80 <__sprint_r>
   1122a: 1d f5        	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   1122c: d2 6b        	ld	s7, 272(sp)
   1122e: 4a 63        	ld	t1, 144(sp)
   11230: 00 03        	addi	s0, sp, 384
   11232: 6f f0 6f f4  	j	0x10978 <_vfprintf_r+0x220>
   11236: f9 68        	lui	a7, 30
   11238: c1 46        	addi	a3, zero, 16
   1123a: 83 27 81 10  	lw	a5, 264(sp)
   1123e: 93 8b 08 9b  	addi	s7, a7, -1616
   11242: 63 d3 26 05  	bge	a3, s2, 0x11288 <_vfprintf_r+0xb30>
   11246: 82 6a        	ld	s5, 0(sp)
   11248: c1 4c        	addi	s9, zero, 16
   1124a: 1d 4d        	addi	s10, zero, 7
   1124c: 21 a0        	j	0x11254 <_vfprintf_r+0xafc>
   1124e: 41 39        	addiw	s2, s2, -16
   11250: 63 dc 2c 03  	bge	s9, s2, 0x11288 <_vfprintf_r+0xb30>
   11254: 85 27        	addiw	a5, a5, 1
   11256: 41 07        	addi	a4, a4, 16
   11258: 23 30 74 01  	sd	s7, 0(s0)
   1125c: 23 34 94 01  	sd	s9, 8(s0)
   11260: 3a ea        	sd	a4, 272(sp)
   11262: 23 24 f1 10  	sw	a5, 264(sp)
   11266: 41 04        	addi	s0, s0, 16
   11268: e3 53 fd fe  	bge	s10, a5, 0x1124e <_vfprintf_r+0xaf6>
   1126c: 10 02        	addi	a2, sp, 256
   1126e: d6 85        	mv	a1, s5
   11270: 4e 85        	mv	a0, s3
   11272: ef 70 f0 10  	jal	0x18b80 <__sprint_r>
   11276: e3 11 05 ee  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   1127a: 41 39        	addiw	s2, s2, -16
   1127c: 52 67        	ld	a4, 272(sp)
   1127e: 83 27 81 10  	lw	a5, 264(sp)
   11282: 00 03        	addi	s0, sp, 384
   11284: e3 c8 2c fd  	blt	s9, s2, 0x11254 <_vfprintf_r+0xafc>
   11288: 9b 86 17 00  	addiw	a3, a5, 1
   1128c: 4a 97        	add	a4, a4, s2
   1128e: 23 30 74 01  	sd	s7, 0(s0)
   11292: 23 34 24 01  	sd	s2, 8(s0)
   11296: 3a ea        	sd	a4, 272(sp)
   11298: 23 24 d1 10  	sw	a3, 264(sp)
   1129c: 9d 47        	addi	a5, zero, 7
   1129e: 63 d2 d7 f2  	bge	a5, a3, 0x109c2 <_vfprintf_r+0x26a>
   112a2: 82 65        	ld	a1, 0(sp)
   112a4: 10 02        	addi	a2, sp, 256
   112a6: 4e 85        	mv	a0, s3
   112a8: ef 70 90 0d  	jal	0x18b80 <__sprint_r>
   112ac: e3 16 05 ea  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   112b0: 52 67        	ld	a4, 272(sp)
   112b2: 6f f0 0f f1  	j	0x109c2 <_vfprintf_r+0x26a>
   112b6: 6e 46        	lw	a2, 216(sp)
   112b8: 63 58 c0 48  	blez	a2, 0x11748 <_vfprintf_r+0xff0>
   112bc: e2 77        	ld	a5, 56(sp)
   112be: 42 77        	ld	a4, 48(sp)
   112c0: b3 0a fb 00  	add	s5, s6, a5
   112c4: ba 86        	mv	a3, a4
   112c6: 63 c1 e7 2c  	blt	a5, a4, 0x11588 <_vfprintf_r+0xe30>
   112ca: 1b 8d 06 00  	sext.w	s10, a3
   112ce: 63 52 a0 03  	blez	s10, 0x112f2 <_vfprintf_r+0xb9a>
   112d2: 03 26 81 10  	lw	a2, 264(sp)
   112d6: ea 9b        	add	s7, s7, s10
   112d8: 23 30 64 01  	sd	s6, 0(s0)
   112dc: 9b 05 16 00  	addiw	a1, a2, 1
   112e0: 23 34 a4 01  	sd	s10, 8(s0)
   112e4: 5e ea        	sd	s7, 272(sp)
   112e6: 23 24 b1 10  	sw	a1, 264(sp)
   112ea: 1d 46        	addi	a2, zero, 7
   112ec: 41 04        	addi	s0, s0, 16
   112ee: e3 49 b6 14  	blt	a2, a1, 0x11c40 <_vfprintf_r+0x14e8>
   112f2: 13 46 fd ff  	not	a2, s10
   112f6: c2 77        	ld	a5, 48(sp)
   112f8: 7d 96        	srai	a2, a2, 63
   112fa: b3 76 cd 00  	and	a3, s10, a2
   112fe: 3b 8d d7 40  	subw	s10, a5, a3
   11302: 63 4c a0 35  	bgtz	s10, 0x1165a <_vfprintf_r+0xf02>
   11306: c2 77        	ld	a5, 48(sp)
   11308: 93 76 09 40  	andi	a3, s2, 1024
   1130c: 3e 9b        	add	s6, s6, a5
   1130e: 63 97 06 60  	bnez	a3, 0x1191c <_vfprintf_r+0x11c4>
   11312: ee 46        	lw	a3, 216(sp)
   11314: e2 77        	ld	a5, 56(sp)
   11316: 63 c6 f6 00  	blt	a3, a5, 0x11322 <_vfprintf_r+0xbca>
   1131a: 13 76 19 00  	andi	a2, s2, 1
   1131e: e3 0e 06 12  	beqz	a2, 0x11c5a <_vfprintf_r+0x1502>
   11322: 03 26 81 10  	lw	a2, 264(sp)
   11326: e6 67        	ld	a5, 88(sp)
   11328: 86 75        	ld	a1, 96(sp)
   1132a: 41 04        	addi	s0, s0, 16
   1132c: 33 87 77 01  	add	a4, a5, s7
   11330: 23 38 b4 fe  	sd	a1, -16(s0)
   11334: 9b 05 16 00  	addiw	a1, a2, 1
   11338: 23 3c f4 fe  	sd	a5, -8(s0)
   1133c: 3a ea        	sd	a4, 272(sp)
   1133e: 23 24 b1 10  	sw	a1, 264(sp)
   11342: 1d 46        	addi	a2, zero, 7
   11344: e3 48 b6 48  	blt	a2, a1, 0x11fd4 <_vfprintf_r+0x187c>
   11348: e2 77        	ld	a5, 56(sp)
   1134a: b3 8c 6a 41  	sub	s9, s5, s6
   1134e: 9b 85 0c 00  	sext.w	a1, s9
   11352: 3b 86 d7 40  	subw	a2, a5, a3
   11356: 63 53 b6 00  	bge	a2, a1, 0x1135c <_vfprintf_r+0xc04>
   1135a: b2 8c        	mv	s9, a2
   1135c: 81 2c        	sext.w	s9, s9
   1135e: 63 51 90 03  	blez	s9, 0x11380 <_vfprintf_r+0xc28>
   11362: 83 26 81 10  	lw	a3, 264(sp)
   11366: 66 97        	add	a4, a4, s9
   11368: 23 30 64 01  	sd	s6, 0(s0)
   1136c: 85 26        	addiw	a3, a3, 1
   1136e: 23 34 94 01  	sd	s9, 8(s0)
   11372: 3a ea        	sd	a4, 272(sp)
   11374: 23 24 d1 10  	sw	a3, 264(sp)
   11378: 9d 47        	addi	a5, zero, 7
   1137a: 41 04        	addi	s0, s0, 16
   1137c: e3 c3 d7 4c  	blt	a5, a3, 0x12042 <_vfprintf_r+0x18ea>
   11380: 93 c7 fc ff  	not	a5, s9
   11384: fd 97        	srai	a5, a5, 63
   11386: b3 fc fc 00  	and	s9, s9, a5
   1138a: bb 0c 96 41  	subw	s9, a2, s9
   1138e: 63 52 90 e3  	blez	s9, 0x109b2 <_vfprintf_r+0x25a>
   11392: 79 6f        	lui	t5, 30
   11394: c1 46        	addi	a3, zero, 16
   11396: 83 27 81 10  	lw	a5, 264(sp)
   1139a: 93 0b 0f 9a  	addi	s7, t5, -1632
   1139e: 63 d7 96 43  	bge	a3, s9, 0x117cc <_vfprintf_r+0x1074>
   113a2: 82 6a        	ld	s5, 0(sp)
   113a4: 41 4d        	addi	s10, zero, 16
   113a6: 9d 4d        	addi	s11, zero, 7
   113a8: 21 a0        	j	0x113b0 <_vfprintf_r+0xc58>
   113aa: c1 3c        	addiw	s9, s9, -16
   113ac: 63 50 9d 43  	bge	s10, s9, 0x117cc <_vfprintf_r+0x1074>
   113b0: 85 27        	addiw	a5, a5, 1
   113b2: 41 07        	addi	a4, a4, 16
   113b4: 23 30 74 01  	sd	s7, 0(s0)
   113b8: 23 34 a4 01  	sd	s10, 8(s0)
   113bc: 3a ea        	sd	a4, 272(sp)
   113be: 23 24 f1 10  	sw	a5, 264(sp)
   113c2: 41 04        	addi	s0, s0, 16
   113c4: e3 d3 fd fe  	bge	s11, a5, 0x113aa <_vfprintf_r+0xc52>
   113c8: 10 02        	addi	a2, sp, 256
   113ca: d6 85        	mv	a1, s5
   113cc: 4e 85        	mv	a0, s3
   113ce: ef 70 20 7b  	jal	0x18b80 <__sprint_r>
   113d2: e3 13 05 d8  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   113d6: 52 67        	ld	a4, 272(sp)
   113d8: 83 27 81 10  	lw	a5, 264(sp)
   113dc: 00 03        	addi	s0, sp, 384
   113de: f1 b7        	j	0x113aa <_vfprintf_r+0xc52>
   113e0: 13 76 19 00  	andi	a2, s2, 1
   113e4: 63 1a 06 f8  	bnez	a2, 0x10b78 <_vfprintf_r+0x420>
   113e8: 03 26 81 10  	lw	a2, 264(sp)
   113ec: 13 8d 1b 00  	addi	s10, s7, 1
   113f0: 23 30 64 01  	sd	s6, 0(s0)
   113f4: 9b 07 16 00  	addiw	a5, a2, 1
   113f8: 14 e4        	sd	a3, 8(s0)
   113fa: 23 24 f1 10  	sw	a5, 264(sp)
   113fe: 1b 84 07 00  	sext.w	s0, a5
   11402: 6a ea        	sd	s10, 272(sp)
   11404: 9d 47        	addi	a5, zero, 7
   11406: 63 d5 87 fe  	bge	a5, s0, 0x10bf0 <_vfprintf_r+0x498>
   1140a: 82 65        	ld	a1, 0(sp)
   1140c: 10 02        	addi	a2, sp, 256
   1140e: 4e 85        	mv	a0, s3
   11410: ef 70 00 77  	jal	0x18b80 <__sprint_r>
   11414: e3 12 05 d4  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11418: 52 6d        	ld	s10, 272(sp)
   1141a: 03 24 81 10  	lw	s0, 264(sp)
   1141e: 93 0c 01 18  	addi	s9, sp, 384
   11422: 6f f0 ef fc  	j	0x10bf0 <_vfprintf_r+0x498>
   11426: 62 77        	ld	a4, 56(sp)
   11428: 85 47        	addi	a5, zero, 1
   1142a: 63 d3 e7 fc  	bge	a5, a4, 0x10bf0 <_vfprintf_r+0x498>
   1142e: 79 6f        	lui	t5, 30
   11430: c5 47        	addi	a5, zero, 17
   11432: 93 0b 0f 9a  	addi	s7, t5, -1632
   11436: 63 de e7 42  	bge	a5, a4, 0x11872 <_vfprintf_r+0x111a>
   1143a: 82 6a        	ld	s5, 0(sp)
   1143c: 41 47        	addi	a4, zero, 16
   1143e: 1d 4b        	addi	s6, zero, 7
   11440: 21 a0        	j	0x11448 <_vfprintf_r+0xcf0>
   11442: c1 3d        	addiw	s11, s11, -16
   11444: 63 57 b7 43  	bge	a4, s11, 0x11872 <_vfprintf_r+0x111a>
   11448: 05 24        	addiw	s0, s0, 1
   1144a: 41 0d        	addi	s10, s10, 16
   1144c: 23 b0 7c 01  	sd	s7, 0(s9)
   11450: 23 b4 ec 00  	sd	a4, 8(s9)
   11454: 6a ea        	sd	s10, 272(sp)
   11456: 23 24 81 10  	sw	s0, 264(sp)
   1145a: c1 0c        	addi	s9, s9, 16
   1145c: e3 53 8b fe  	bge	s6, s0, 0x11442 <_vfprintf_r+0xcea>
   11460: 10 02        	addi	a2, sp, 256
   11462: d6 85        	mv	a1, s5
   11464: 4e 85        	mv	a0, s3
   11466: ef 70 a0 71  	jal	0x18b80 <__sprint_r>
   1146a: e3 17 05 ce  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   1146e: 52 6d        	ld	s10, 272(sp)
   11470: 03 24 81 10  	lw	s0, 264(sp)
   11474: 93 0c 01 18  	addi	s9, sp, 384
   11478: 41 47        	addi	a4, zero, 16
   1147a: e1 b7        	j	0x11442 <_vfprintf_r+0xcea>
   1147c: 82 65        	ld	a1, 0(sp)
   1147e: 10 02        	addi	a2, sp, 256
   11480: 4e 85        	mv	a0, s3
   11482: ef 70 e0 6f  	jal	0x18b80 <__sprint_r>
   11486: e3 19 05 cc  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   1148a: d2 6b        	ld	s7, 272(sp)
   1148c: 00 03        	addi	s0, sp, 384
   1148e: 6f f0 af cf  	j	0x10988 <_vfprintf_r+0x230>
   11492: 93 77 09 01  	andi	a5, s2, 16
   11496: dd e7        	bnez	a5, 0x11544 <_vfprintf_r+0xdec>
   11498: 82 76        	ld	a3, 32(sp)
   1149a: 93 77 09 04  	andi	a5, s2, 64
   1149e: 83 ab 06 00  	lw	s7, 0(a3)
   114a2: 63 8c 07 3e  	beqz	a5, 0x1189a <_vfprintf_r+0x1142>
   114a6: 9b 9b 0b 01  	slliw	s7, s7, 16
   114aa: 9b db 0b 41  	sraiw	s7, s7, 16
   114ae: de 87        	mv	a5, s7
   114b0: e3 d0 07 96  	bgez	a5, 0x10e10 <_vfprintf_r+0x6b8>
   114b4: 3a f0        	sd	a4, 32(sp)
   114b6: b3 0b 70 41  	neg	s7, s7
   114ba: ca 8c        	mv	s9, s2
   114bc: 13 07 d0 02  	addi	a4, zero, 45
   114c0: 85 47        	addi	a5, zero, 1
   114c2: f9 b8        	j	0x10da0 <_vfprintf_r+0x648>
   114c4: 93 77 09 01  	andi	a5, s2, 16
   114c8: b5 eb        	bnez	a5, 0x1153c <_vfprintf_r+0xde4>
   114ca: 82 76        	ld	a3, 32(sp)
   114cc: 93 77 09 04  	andi	a5, s2, 64
   114d0: 83 ab 06 00  	lw	s7, 0(a3)
   114d4: 63 88 07 3e  	beqz	a5, 0x118c4 <_vfprintf_r+0x116c>
   114d8: c2 1b        	slli	s7, s7, 48
   114da: 93 db 0b 03  	srli	s7, s7, 48
   114de: 85 ba        	j	0x10e4e <_vfprintf_r+0x6f6>
   114e0: 93 f7 0c 01  	andi	a5, s9, 16
   114e4: b1 e7        	bnez	a5, 0x11530 <_vfprintf_r+0xdd8>
   114e6: 82 76        	ld	a3, 32(sp)
   114e8: 93 f7 0c 04  	andi	a5, s9, 64
   114ec: 83 ab 06 00  	lw	s7, 0(a3)
   114f0: 63 80 07 3c  	beqz	a5, 0x118b0 <_vfprintf_r+0x1158>
   114f4: c2 1b        	slli	s7, s7, 48
   114f6: 93 db 0b 03  	srli	s7, s7, 48
   114fa: 3a f0        	sd	a4, 32(sp)
   114fc: 85 47        	addi	a5, zero, 1
   114fe: 45 b0        	j	0x10d9e <_vfprintf_r+0x646>
   11500: 81 4d        	mv	s11, zero
   11502: 13 0b c1 17  	addi	s6, sp, 380
   11506: c9 b0        	j	0x10dc8 <_vfprintf_r+0x670>
   11508: 93 77 19 00  	andi	a5, s2, 1
   1150c: 5e 87        	mv	a4, s7
   1150e: 63 82 07 ca  	beqz	a5, 0x109b2 <_vfprintf_r+0x25a>
   11512: 29 b4        	j	0x10f1c <_vfprintf_r+0x7c4>
   11514: 56 83        	mv	t1, s5
   11516: d5 b0        	j	0x10dfa <_vfprintf_r+0x6a2>
   11518: 03 ce 04 00  	lbu	t3, 0(s1)
   1151c: 3e f0        	sd	a5, 32(sp)
   1151e: 6f f0 8f b7  	j	0x10896 <_vfprintf_r+0x13e>
   11522: 93 07 00 03  	addi	a5, zero, 48
   11526: a3 0d f1 16  	sb	a5, 379(sp)
   1152a: 13 0b b1 17  	addi	s6, sp, 379
   1152e: 69 b8        	j	0x10dc8 <_vfprintf_r+0x670>
   11530: 82 77        	ld	a5, 32(sp)
   11532: 3a f0        	sd	a4, 32(sp)
   11534: 83 bb 07 00  	ld	s7, 0(a5)
   11538: 85 47        	addi	a5, zero, 1
   1153a: 95 b0        	j	0x10d9e <_vfprintf_r+0x646>
   1153c: 82 77        	ld	a5, 32(sp)
   1153e: 83 bb 07 00  	ld	s7, 0(a5)
   11542: 31 b2        	j	0x10e4e <_vfprintf_r+0x6f6>
   11544: 82 77        	ld	a5, 32(sp)
   11546: 9c 63        	ld	a5, 0(a5)
   11548: be 8b        	mv	s7, a5
   1154a: c9 b0        	j	0x10e0c <_vfprintf_r+0x6b4>
   1154c: 93 77 09 01  	andi	a5, s2, 16
   11550: e3 91 07 92  	bnez	a5, 0x10e72 <_vfprintf_r+0x71a>
   11554: 93 77 09 04  	andi	a5, s2, 64
   11558: 63 92 07 72  	bnez	a5, 0x11c7c <_vfprintf_r+0x1524>
   1155c: 13 79 09 20  	andi	s2, s2, 512
   11560: c2 67        	ld	a5, 16(sp)
   11562: e3 0c 09 44  	beqz	s2, 0x121ba <_vfprintf_r+0x1a62>
   11566: 23 00 f7 00  	sb	a5, 0(a4)
   1156a: 31 b2        	j	0x10e76 <_vfprintf_r+0x71e>
   1156c: 82 77        	ld	a5, 32(sp)
   1156e: 93 8a f7 00  	addi	s5, a5, 15
   11572: 93 fa 0a ff  	andi	s5, s5, -16
   11576: 93 87 0a 01  	addi	a5, s5, 16
   1157a: 03 b7 0a 00  	ld	a4, 0(s5)
   1157e: 83 b5 8a 00  	ld	a1, 8(s5)
   11582: 3e f0        	sd	a5, 32(sp)
   11584: 6f f0 ef d3  	j	0x10ac2 <_vfprintf_r+0x36a>
   11588: be 86        	mv	a3, a5
   1158a: 1b 8d 06 00  	sext.w	s10, a3
   1158e: e3 42 a0 d5  	bgtz	s10, 0x112d2 <_vfprintf_r+0xb7a>
   11592: 85 b3        	j	0x112f2 <_vfprintf_r+0xb9a>
   11594: 82 65        	ld	a1, 0(sp)
   11596: 10 02        	addi	a2, sp, 256
   11598: 4e 85        	mv	a0, s3
   1159a: 1a e9        	sd	t1, 144(sp)
   1159c: ef 70 40 5e  	jal	0x18b80 <__sprint_r>
   115a0: e3 1c 05 ba  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   115a4: d2 6b        	ld	s7, 272(sp)
   115a6: 4a 63        	ld	t1, 144(sp)
   115a8: 00 03        	addi	s0, sp, 384
   115aa: 6f f0 6f bd  	j	0x10980 <_vfprintf_r+0x228>
   115ae: f5 67        	lui	a5, 29
   115b0: 93 87 07 76  	addi	a5, a5, 1888
   115b4: 56 83        	mv	t1, s5
   115b6: be f4        	sd	a5, 104(sp)
   115b8: 82 76        	ld	a3, 32(sp)
   115ba: 93 77 09 02  	andi	a5, s2, 32
   115be: 13 87 86 00  	addi	a4, a3, 8
   115c2: ed c7        	beqz	a5, 0x116ac <_vfprintf_r+0xf54>
   115c4: 83 bb 06 00  	ld	s7, 0(a3)
   115c8: 93 77 19 00  	andi	a5, s2, 1
   115cc: 81 cf        	beqz	a5, 0x115e4 <_vfprintf_r+0xe8c>
   115ce: 63 8b 0b 00  	beqz	s7, 0x115e4 <_vfprintf_r+0xe8c>
   115d2: 13 69 29 00  	ori	s2, s2, 2
   115d6: 93 07 00 03  	addi	a5, zero, 48
   115da: 23 08 f1 0c  	sb	a5, 208(sp)
   115de: a3 08 a1 0d  	sb	s10, 209(sp)
   115e2: 01 29        	sext.w	s2, s2
   115e4: 93 7c f9 bf  	andi	s9, s2, -1025
   115e8: 81 2c        	sext.w	s9, s9
   115ea: 3a f0        	sd	a4, 32(sp)
   115ec: 89 47        	addi	a5, zero, 2
   115ee: 6f f0 0f fb  	j	0x10d9e <_vfprintf_r+0x646>
   115f2: 56 83        	mv	t1, s5
   115f4: ca 8c        	mv	s9, s2
   115f6: 6f f0 2f f9  	j	0x10d88 <_vfprintf_r+0x630>
   115fa: f5 67        	lui	a5, 29
   115fc: 93 87 87 77  	addi	a5, a5, 1912
   11600: 56 83        	mv	t1, s5
   11602: be f4        	sd	a5, 104(sp)
   11604: 55 bf        	j	0x115b8 <_vfprintf_r+0xe60>
   11606: 82 65        	ld	a1, 0(sp)
   11608: 10 02        	addi	a2, sp, 256
   1160a: 4e 85        	mv	a0, s3
   1160c: ef 70 40 57  	jal	0x18b80 <__sprint_r>
   11610: e3 14 05 b4  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11614: 52 67        	ld	a4, 272(sp)
   11616: 00 03        	addi	s0, sp, 384
   11618: 25 b2        	j	0x10f40 <_vfprintf_r+0x7e8>
   1161a: 13 69 09 20  	ori	s2, s2, 512
   1161e: 03 ce 14 00  	lbu	t3, 1(s1)
   11622: 01 29        	sext.w	s2, s2
   11624: 85 04        	addi	s1, s1, 1
   11626: 6f f0 0f a7  	j	0x10896 <_vfprintf_r+0x13e>
   1162a: 56 83        	mv	t1, s5
   1162c: 6f f0 1f 81  	j	0x10e3c <_vfprintf_r+0x6e4>
   11630: 13 69 09 02  	ori	s2, s2, 32
   11634: 03 ce 14 00  	lbu	t3, 1(s1)
   11638: 01 29        	sext.w	s2, s2
   1163a: 85 04        	addi	s1, s1, 1
   1163c: 6f f0 af a5  	j	0x10896 <_vfprintf_r+0x13e>
   11640: 99 47        	addi	a5, zero, 6
   11642: 9a 86        	mv	a3, t1
   11644: 63 e9 67 60  	bltu	a5, t1, 0x11c56 <_vfprintf_r+0x14fe>
   11648: 9b 8d 06 00  	sext.w	s11, a3
   1164c: f5 67        	lui	a5, 29
   1164e: 6e 8c        	mv	s8, s11
   11650: 56 f0        	sd	s5, 32(sp)
   11652: 13 8b 07 79  	addi	s6, a5, 1936
   11656: 6f f0 cf aa  	j	0x10902 <_vfprintf_r+0x1aa>
   1165a: 79 6f        	lui	t5, 30
   1165c: c1 45        	addi	a1, zero, 16
   1165e: 5e 87        	mv	a4, s7
   11660: 03 26 81 10  	lw	a2, 264(sp)
   11664: 93 0b 0f 9a  	addi	s7, t5, -1632
   11668: 63 da a5 29  	bge	a1, s10, 0x118fc <_vfprintf_r+0x11a4>
   1166c: 82 6c        	ld	s9, 0(sp)
   1166e: c1 48        	addi	a7, zero, 16
   11670: 9d 4d        	addi	s11, zero, 7
   11672: 21 a0        	j	0x1167a <_vfprintf_r+0xf22>
   11674: 41 3d        	addiw	s10, s10, -16
   11676: 63 d3 a8 29  	bge	a7, s10, 0x118fc <_vfprintf_r+0x11a4>
   1167a: 05 26        	addiw	a2, a2, 1
   1167c: 41 07        	addi	a4, a4, 16
   1167e: 23 30 74 01  	sd	s7, 0(s0)
   11682: 23 34 14 01  	sd	a7, 8(s0)
   11686: 3a ea        	sd	a4, 272(sp)
   11688: 23 24 c1 10  	sw	a2, 264(sp)
   1168c: 41 04        	addi	s0, s0, 16
   1168e: e3 d3 cd fe  	bge	s11, a2, 0x11674 <_vfprintf_r+0xf1c>
   11692: 10 02        	addi	a2, sp, 256
   11694: e6 85        	mv	a1, s9
   11696: 4e 85        	mv	a0, s3
   11698: ef 70 80 4e  	jal	0x18b80 <__sprint_r>
   1169c: e3 1e 05 aa  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   116a0: 52 67        	ld	a4, 272(sp)
   116a2: 03 26 81 10  	lw	a2, 264(sp)
   116a6: 00 03        	addi	s0, sp, 384
   116a8: c1 48        	addi	a7, zero, 16
   116aa: e9 b7        	j	0x11674 <_vfprintf_r+0xf1c>
   116ac: 93 77 09 01  	andi	a5, s2, 16
   116b0: e5 cb        	beqz	a5, 0x117a0 <_vfprintf_r+0x1048>
   116b2: 82 77        	ld	a5, 32(sp)
   116b4: 83 bb 07 00  	ld	s7, 0(a5)
   116b8: 01 bf        	j	0x115c8 <_vfprintf_r+0xe70>
   116ba: 82 65        	ld	a1, 0(sp)
   116bc: 10 02        	addi	a2, sp, 256
   116be: 4e 85        	mv	a0, s3
   116c0: ef 70 00 4c  	jal	0x18b80 <__sprint_r>
   116c4: e3 1a 05 a8  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   116c8: 52 67        	ld	a4, 272(sp)
   116ca: 83 26 81 10  	lw	a3, 264(sp)
   116ce: 93 0c 01 18  	addi	s9, sp, 384
   116d2: 6f f0 8f cc  	j	0x10b9a <_vfprintf_r+0x442>
   116d6: 82 65        	ld	a1, 0(sp)
   116d8: 10 02        	addi	a2, sp, 256
   116da: 4e 85        	mv	a0, s3
   116dc: ef 70 40 4a  	jal	0x18b80 <__sprint_r>
   116e0: e3 1c 05 a6  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   116e4: 52 6d        	ld	s10, 272(sp)
   116e6: 03 24 81 10  	lw	s0, 264(sp)
   116ea: 93 0c 01 18  	addi	s9, sp, 384
   116ee: 6f f0 ef cc  	j	0x10bbc <_vfprintf_r+0x464>
   116f2: 93 0a c1 17  	addi	s5, sp, 380
   116f6: 0a 68        	ld	a6, 128(sp)
   116f8: 93 f7 0c 40  	andi	a5, s9, 1024
   116fc: 81 46        	mv	a3, zero
   116fe: d6 85        	mv	a1, s5
   11700: 29 49        	addi	s2, zero, 10
   11702: a5 4d        	addi	s11, zero, 9
   11704: 13 0c f0 0f  	addi	s8, zero, 255
   11708: 39 a0        	j	0x11716 <_vfprintf_r+0xfbe>
   1170a: 33 d7 2b 03  	divu	a4, s7, s2
   1170e: 63 f6 7d 03  	bgeu	s11, s7, 0x1173a <_vfprintf_r+0xfe2>
   11712: ba 8b        	mv	s7, a4
   11714: da 85        	mv	a1, s6
   11716: 13 8b f5 ff  	addi	s6, a1, -1
   1171a: 85 26        	addiw	a3, a3, 1
   1171c: 33 f7 2b 03  	remu	a4, s7, s2
   11720: 1b 07 07 03  	addiw	a4, a4, 48
   11724: a3 8f e5 fe  	sb	a4, -1(a1)
   11728: ed d3        	beqz	a5, 0x1170a <_vfprintf_r+0xfb2>
   1172a: 03 47 08 00  	lbu	a4, 0(a6)
   1172e: e3 9e e6 fc  	bne	a3, a4, 0x1170a <_vfprintf_r+0xfb2>
   11732: e3 0c 87 fd  	beq	a4, s8, 0x1170a <_vfprintf_r+0xfb2>
   11736: 63 e7 7d 35  	bltu	s11, s7, 0x11a84 <_vfprintf_r+0x132c>
   1173a: 36 fc        	sd	a3, 56(sp)
   1173c: 42 e1        	sd	a6, 128(sp)
   1173e: bb 8d 6a 41  	subw	s11, s5, s6
   11742: 66 89        	mv	s2, s9
   11744: 6f f0 4f e8  	j	0x10dc8 <_vfprintf_r+0x670>
   11748: 83 26 81 10  	lw	a3, 264(sp)
   1174c: f5 65        	lui	a1, 29
   1174e: 93 85 85 79  	addi	a1, a1, 1944
   11752: 85 26        	addiw	a3, a3, 1
   11754: 0c e0        	sd	a1, 0(s0)
   11756: 13 87 1b 00  	addi	a4, s7, 1
   1175a: 85 45        	addi	a1, zero, 1
   1175c: 0c e4        	sd	a1, 8(s0)
   1175e: 23 24 d1 10  	sw	a3, 264(sp)
   11762: 9b 85 06 00  	sext.w	a1, a3
   11766: 3a ea        	sd	a4, 272(sp)
   11768: 9d 46        	addi	a3, zero, 7
   1176a: 41 04        	addi	s0, s0, 16
   1176c: 63 c4 b6 34  	blt	a3, a1, 0x11ab4 <_vfprintf_r+0x135c>
   11770: 55 ee        	bnez	a2, 0x1182c <_vfprintf_r+0x10d4>
   11772: e2 77        	ld	a5, 56(sp)
   11774: 93 76 19 00  	andi	a3, s2, 1
   11778: dd 8e        	or	a3, a3, a5
   1177a: 63 8c 06 a2  	beqz	a3, 0x109b2 <_vfprintf_r+0x25a>
   1177e: 03 26 81 10  	lw	a2, 264(sp)
   11782: e6 67        	ld	a5, 88(sp)
   11784: 86 76        	ld	a3, 96(sp)
   11786: 3e 97        	add	a4, a4, a5
   11788: 14 e0        	sd	a3, 0(s0)
   1178a: 9b 06 16 00  	addiw	a3, a2, 1
   1178e: 1c e4        	sd	a5, 8(s0)
   11790: 3a ea        	sd	a4, 272(sp)
   11792: 23 24 d1 10  	sw	a3, 264(sp)
   11796: 1d 46        	addi	a2, zero, 7
   11798: 63 44 d6 52  	blt	a2, a3, 0x11cc0 <_vfprintf_r+0x1568>
   1179c: 41 04        	addi	s0, s0, 16
   1179e: 5d a8        	j	0x11854 <_vfprintf_r+0x10fc>
   117a0: 82 76        	ld	a3, 32(sp)
   117a2: 93 77 09 04  	andi	a5, s2, 64
   117a6: 83 ab 06 00  	lw	s7, 0(a3)
   117aa: 63 85 07 12  	beqz	a5, 0x118d4 <_vfprintf_r+0x117c>
   117ae: c2 1b        	slli	s7, s7, 48
   117b0: 93 db 0b 03  	srli	s7, s7, 48
   117b4: 11 bd        	j	0x115c8 <_vfprintf_r+0xe70>
   117b6: 82 65        	ld	a1, 0(sp)
   117b8: 10 02        	addi	a2, sp, 256
   117ba: 4e 85        	mv	a0, s3
   117bc: ef 70 40 3c  	jal	0x18b80 <__sprint_r>
   117c0: e3 1c 05 98  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   117c4: d2 6b        	ld	s7, 272(sp)
   117c6: 00 03        	addi	s0, sp, 384
   117c8: 6f f0 cf f4  	j	0x10f14 <_vfprintf_r+0x7bc>
   117cc: 9b 86 17 00  	addiw	a3, a5, 1
   117d0: 23 30 74 01  	sd	s7, 0(s0)
   117d4: 23 34 94 01  	sd	s9, 8(s0)
   117d8: 66 97        	add	a4, a4, s9
   117da: 3a ea        	sd	a4, 272(sp)
   117dc: 23 24 d1 10  	sw	a3, 264(sp)
   117e0: 9d 47        	addi	a5, zero, 7
   117e2: 63 d7 d7 9c  	bge	a5, a3, 0x109b0 <_vfprintf_r+0x258>
   117e6: 6f f0 0f c3  	j	0x10c16 <_vfprintf_r+0x4be>
   117ea: 3a f0        	sd	a4, 32(sp)
   117ec: 29 bc        	j	0x11206 <_vfprintf_r+0xaae>
   117ee: 4e 75        	ld	a0, 240(sp)
   117f0: ee 75        	ld	a1, 248(sp)
   117f2: 01 46        	mv	a2, zero
   117f4: 81 46        	mv	a3, zero
   117f6: ef a0 00 4d  	jal	0x1bcc6 <__lttf2>
   117fa: 63 45 05 7c  	bltz	a0, 0x11fc4 <_vfprintf_r+0x186c>
   117fe: 03 47 f1 0c  	lbu	a4, 207(sp)
   11802: 93 07 70 04  	addi	a5, zero, 71
   11806: 63 d2 a7 2d  	bge	a5, s10, 0x11aca <_vfprintf_r+0x1372>
   1180a: f5 67        	lui	a5, 29
   1180c: 13 8b 87 74  	addi	s6, a5, 1864
   11810: 13 79 f9 f7  	andi	s2, s2, -129
   11814: 02 ec        	sd	zero, 24(sp)
   11816: 82 e4        	sd	zero, 72(sp)
   11818: 82 e0        	sd	zero, 64(sp)
   1181a: 02 f8        	sd	zero, 48(sp)
   1181c: 01 29        	sext.w	s2, s2
   1181e: 0d 4c        	addi	s8, zero, 3
   11820: 8d 4d        	addi	s11, zero, 3
   11822: 01 43        	mv	t1, zero
   11824: 63 1f 07 c6  	bnez	a4, 0x10ca2 <_vfprintf_r+0x54a>
   11828: 6f f0 4f 8e  	j	0x1090c <_vfprintf_r+0x1b4>
   1182c: 83 25 81 10  	lw	a1, 264(sp)
   11830: e6 67        	ld	a5, 88(sp)
   11832: 86 76        	ld	a3, 96(sp)
   11834: 41 04        	addi	s0, s0, 16
   11836: 3e 97        	add	a4, a4, a5
   11838: 23 38 d4 fe  	sd	a3, -16(s0)
   1183c: 9b 86 15 00  	addiw	a3, a1, 1
   11840: 23 3c f4 fe  	sd	a5, -8(s0)
   11844: 3a ea        	sd	a4, 272(sp)
   11846: 23 24 d1 10  	sw	a3, 264(sp)
   1184a: 9d 45        	addi	a1, zero, 7
   1184c: 63 ca d5 46  	blt	a1, a3, 0x11cc0 <_vfprintf_r+0x1568>
   11850: e3 47 06 04  	bltz	a2, 0x1209e <_vfprintf_r+0x1946>
   11854: e2 77        	ld	a5, 56(sp)
   11856: 1b 86 16 00  	addiw	a2, a3, 1
   1185a: 23 30 64 01  	sd	s6, 0(s0)
   1185e: 3e 97        	add	a4, a4, a5
   11860: 1c e4        	sd	a5, 8(s0)
   11862: 3a ea        	sd	a4, 272(sp)
   11864: 23 24 c1 10  	sw	a2, 264(sp)
   11868: 9d 47        	addi	a5, zero, 7
   1186a: 63 d3 c7 94  	bge	a5, a2, 0x109b0 <_vfprintf_r+0x258>
   1186e: 6f f0 8f ba  	j	0x10c16 <_vfprintf_r+0x4be>
   11872: 05 24        	addiw	s0, s0, 1
   11874: 6e 9d        	add	s10, s10, s11
   11876: 23 b0 7c 01  	sd	s7, 0(s9)
   1187a: 23 b4 bc 01  	sd	s11, 8(s9)
   1187e: 6a ea        	sd	s10, 272(sp)
   11880: 23 24 81 10  	sw	s0, 264(sp)
   11884: 9d 47        	addi	a5, zero, 7
   11886: 63 d4 87 b6  	bge	a5, s0, 0x10bee <_vfprintf_r+0x496>
   1188a: 41 b6        	j	0x1140a <_vfprintf_r+0xcb2>
   1188c: 5a 85        	mv	a0, s6
   1188e: ef 50 20 5f  	jal	0x16e80 <strlen>
   11892: 9b 0d 05 00  	sext.w	s11, a0
   11896: 6f f0 cf be  	j	0x10c82 <_vfprintf_r+0x52a>
   1189a: 93 77 09 20  	andi	a5, s2, 512
   1189e: 63 8b 07 3e  	beqz	a5, 0x11c94 <_vfprintf_r+0x153c>
   118a2: 9b 9b 8b 01  	slliw	s7, s7, 24
   118a6: 9b db 8b 41  	sraiw	s7, s7, 24
   118aa: de 87        	mv	a5, s7
   118ac: 6f f0 0f d6  	j	0x10e0c <_vfprintf_r+0x6b4>
   118b0: 93 f7 0c 20  	andi	a5, s9, 512
   118b4: 63 89 07 3c  	beqz	a5, 0x11c86 <_vfprintf_r+0x152e>
   118b8: 93 fb fb 0f  	andi	s7, s7, 255
   118bc: 3a f0        	sd	a4, 32(sp)
   118be: 85 47        	addi	a5, zero, 1
   118c0: 6f f0 ef cd  	j	0x10d9e <_vfprintf_r+0x646>
   118c4: 93 77 09 20  	andi	a5, s2, 512
   118c8: 63 89 07 3c  	beqz	a5, 0x11c9a <_vfprintf_r+0x1542>
   118cc: 93 fb fb 0f  	andi	s7, s7, 255
   118d0: 6f f0 ef d7  	j	0x10e4e <_vfprintf_r+0x6f6>
   118d4: 93 77 09 20  	andi	a5, s2, 512
   118d8: 63 8e 07 38  	beqz	a5, 0x11c74 <_vfprintf_r+0x151c>
   118dc: 93 fb fb 0f  	andi	s7, s7, 255
   118e0: e5 b1        	j	0x115c8 <_vfprintf_r+0xe70>
   118e2: ee 77        	ld	a5, 248(sp)
   118e4: 63 c7 07 3c  	bltz	a5, 0x11cb2 <_vfprintf_r+0x155a>
   118e8: 03 47 f1 0c  	lbu	a4, 207(sp)
   118ec: 93 07 70 04  	addi	a5, zero, 71
   118f0: 63 d4 a7 75  	bge	a5, s10, 0x12038 <_vfprintf_r+0x18e0>
   118f4: f5 67        	lui	a5, 29
   118f6: 13 8b 87 75  	addi	s6, a5, 1880
   118fa: 19 bf        	j	0x11810 <_vfprintf_r+0x10b8>
   118fc: 9b 05 16 00  	addiw	a1, a2, 1
   11900: 6a 97        	add	a4, a4, s10
   11902: 23 30 74 01  	sd	s7, 0(s0)
   11906: 23 34 a4 01  	sd	s10, 8(s0)
   1190a: 3a ea        	sd	a4, 272(sp)
   1190c: 23 24 b1 10  	sw	a1, 264(sp)
   11910: 9d 46        	addi	a3, zero, 7
   11912: 63 cf b6 54  	blt	a3, a1, 0x11e70 <_vfprintf_r+0x1718>
   11916: 41 04        	addi	s0, s0, 16
   11918: ba 8b        	mv	s7, a4
   1191a: f5 b2        	j	0x11306 <_vfprintf_r+0xbae>
   1191c: 26 e9        	sd	s1, 144(sp)
   1191e: 86 64        	ld	s1, 64(sp)
   11920: 79 6e        	lui	t3, 30
   11922: 52 f1        	sd	s4, 160(sp)
   11924: e2 e0        	sd	s8, 64(sp)
   11926: 4a ed        	sd	s2, 152(sp)
   11928: 5e 87        	mv	a4, s7
   1192a: 0a 6a        	ld	s4, 128(sp)
   1192c: 26 6c        	ld	s8, 72(sp)
   1192e: 82 6c        	ld	s9, 0(sp)
   11930: aa 6b        	ld	s7, 136(sp)
   11932: 1d 43        	addi	t1, zero, 7
   11934: 41 4d        	addi	s10, zero, 16
   11936: 93 0d 0e 9a  	addi	s11, t3, -1632
   1193a: c1 c0        	beqz	s1, 0x119ba <_vfprintf_r+0x1262>
   1193c: 63 07 0c 10  	beqz	s8, 0x11a4a <_vfprintf_r+0x12f2>
   11940: 7d 3c        	addiw	s8, s8, -1
   11942: 83 27 81 10  	lw	a5, 264(sp)
   11946: e6 76        	ld	a3, 120(sp)
   11948: 5e 97        	add	a4, a4, s7
   1194a: 23 34 74 01  	sd	s7, 8(s0)
   1194e: 14 e0        	sd	a3, 0(s0)
   11950: 9b 86 17 00  	addiw	a3, a5, 1
   11954: 3a ea        	sd	a4, 272(sp)
   11956: 23 24 d1 10  	sw	a3, 264(sp)
   1195a: 41 04        	addi	s0, s0, 16
   1195c: 63 4a d3 0e  	blt	t1, a3, 0x11a50 <_vfprintf_r+0x12f8>
   11960: 03 46 0a 00  	lbu	a2, 0(s4)
   11964: b3 87 6a 41  	sub	a5, s5, s6
   11968: 9b 85 07 00  	sext.w	a1, a5
   1196c: 9b 06 06 00  	sext.w	a3, a2
   11970: 63 53 b6 00  	bge	a2, a1, 0x11976 <_vfprintf_r+0x121e>
   11974: b6 87        	mv	a5, a3
   11976: 1b 89 07 00  	sext.w	s2, a5
   1197a: 63 55 20 03  	blez	s2, 0x119a4 <_vfprintf_r+0x124c>
   1197e: 83 26 81 10  	lw	a3, 264(sp)
   11982: 4a 97        	add	a4, a4, s2
   11984: 23 30 64 01  	sd	s6, 0(s0)
   11988: 1b 86 16 00  	addiw	a2, a3, 1
   1198c: 23 34 24 01  	sd	s2, 8(s0)
   11990: 3a ea        	sd	a4, 272(sp)
   11992: 23 24 c1 10  	sw	a2, 264(sp)
   11996: 63 48 c3 0c  	blt	t1, a2, 0x11a66 <_vfprintf_r+0x130e>
   1199a: 03 46 0a 00  	lbu	a2, 0(s4)
   1199e: 41 04        	addi	s0, s0, 16
   119a0: 9b 06 06 00  	sext.w	a3, a2
   119a4: 93 45 f9 ff  	not	a1, s2
   119a8: fd 95        	srai	a1, a1, 63
   119aa: b3 77 b9 00  	and	a5, s2, a1
   119ae: 3b 89 f6 40  	subw	s2, a3, a5
   119b2: 63 40 20 03  	bgtz	s2, 0x119d2 <_vfprintf_r+0x127a>
   119b6: 32 9b        	add	s6, s6, a2
   119b8: d1 f0        	bnez	s1, 0x1193c <_vfprintf_r+0x11e4>
   119ba: e3 13 0c f8  	bnez	s8, 0x11940 <_vfprintf_r+0x11e8>
   119be: 52 e1        	sd	s4, 128(sp)
   119c0: ca 64        	ld	s1, 144(sp)
   119c2: 6a 69        	ld	s2, 152(sp)
   119c4: 0a 7a        	ld	s4, 160(sp)
   119c6: 06 6c        	ld	s8, 64(sp)
   119c8: ba 8b        	mv	s7, a4
   119ca: e3 f4 6a 95  	bgeu	s5, s6, 0x11312 <_vfprintf_r+0xbba>
   119ce: 56 8b        	mv	s6, s5
   119d0: 89 b2        	j	0x11312 <_vfprintf_r+0xbba>
   119d2: f9 65        	lui	a1, 30
   119d4: 83 27 81 10  	lw	a5, 264(sp)
   119d8: 93 85 05 9a  	addi	a1, a1, -1632
   119dc: 63 57 2d 05  	bge	s10, s2, 0x11a2a <_vfprintf_r+0x12d2>
   119e0: 26 f8        	sd	s1, 48(sp)
   119e2: ca 84        	mv	s1, s2
   119e4: 6e 89        	mv	s2, s11
   119e6: 21 a0        	j	0x119ee <_vfprintf_r+0x1296>
   119e8: c1 34        	addiw	s1, s1, -16
   119ea: 63 5d 9d 02  	bge	s10, s1, 0x11a24 <_vfprintf_r+0x12cc>
   119ee: 85 27        	addiw	a5, a5, 1
   119f0: 41 07        	addi	a4, a4, 16
   119f2: 23 30 b4 01  	sd	s11, 0(s0)
   119f6: 23 34 a4 01  	sd	s10, 8(s0)
   119fa: 3a ea        	sd	a4, 272(sp)
   119fc: 23 24 f1 10  	sw	a5, 264(sp)
   11a00: 41 04        	addi	s0, s0, 16
   11a02: e3 53 f3 fe  	bge	t1, a5, 0x119e8 <_vfprintf_r+0x1290>
   11a06: 10 02        	addi	a2, sp, 256
   11a08: e6 85        	mv	a1, s9
   11a0a: 4e 85        	mv	a0, s3
   11a0c: ef 70 40 17  	jal	0x18b80 <__sprint_r>
   11a10: 63 14 05 f4  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11a14: c1 34        	addiw	s1, s1, -16
   11a16: 52 67        	ld	a4, 272(sp)
   11a18: 83 27 81 10  	lw	a5, 264(sp)
   11a1c: 00 03        	addi	s0, sp, 384
   11a1e: 1d 43        	addi	t1, zero, 7
   11a20: e3 47 9d fc  	blt	s10, s1, 0x119ee <_vfprintf_r+0x1296>
   11a24: ca 85        	mv	a1, s2
   11a26: 26 89        	mv	s2, s1
   11a28: c2 74        	ld	s1, 48(sp)
   11a2a: 9b 86 17 00  	addiw	a3, a5, 1
   11a2e: 4a 97        	add	a4, a4, s2
   11a30: 0c e0        	sd	a1, 0(s0)
   11a32: 23 34 24 01  	sd	s2, 8(s0)
   11a36: 3a ea        	sd	a4, 272(sp)
   11a38: 23 24 d1 10  	sw	a3, 264(sp)
   11a3c: 63 43 d3 56  	blt	t1, a3, 0x11fa2 <_vfprintf_r+0x184a>
   11a40: 03 46 0a 00  	lbu	a2, 0(s4)
   11a44: 41 04        	addi	s0, s0, 16
   11a46: 32 9b        	add	s6, s6, a2
   11a48: 85 bf        	j	0x119b8 <_vfprintf_r+0x1260>
   11a4a: 7d 1a        	addi	s4, s4, -1
   11a4c: fd 34        	addiw	s1, s1, -1
   11a4e: d5 bd        	j	0x11942 <_vfprintf_r+0x11ea>
   11a50: 10 02        	addi	a2, sp, 256
   11a52: e6 85        	mv	a1, s9
   11a54: 4e 85        	mv	a0, s3
   11a56: ef 70 a0 12  	jal	0x18b80 <__sprint_r>
   11a5a: 63 1f 05 ee  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11a5e: 52 67        	ld	a4, 272(sp)
   11a60: 00 03        	addi	s0, sp, 384
   11a62: 1d 43        	addi	t1, zero, 7
   11a64: f5 bd        	j	0x11960 <_vfprintf_r+0x1208>
   11a66: 10 02        	addi	a2, sp, 256
   11a68: e6 85        	mv	a1, s9
   11a6a: 4e 85        	mv	a0, s3
   11a6c: ef 70 40 11  	jal	0x18b80 <__sprint_r>
   11a70: 63 14 05 ee  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11a74: 03 46 0a 00  	lbu	a2, 0(s4)
   11a78: 52 67        	ld	a4, 272(sp)
   11a7a: 00 03        	addi	s0, sp, 384
   11a7c: 9b 06 06 00  	sext.w	a3, a2
   11a80: 1d 43        	addi	t1, zero, 7
   11a82: 0d b7        	j	0x119a4 <_vfprintf_r+0x124c>
   11a84: 3e fc        	sd	a5, 56(sp)
   11a86: aa 67        	ld	a5, 136(sp)
   11a88: e6 75        	ld	a1, 120(sp)
   11a8a: 1a f8        	sd	t1, 48(sp)
   11a8c: 33 0b fb 40  	sub	s6, s6, a5
   11a90: 3e 86        	mv	a2, a5
   11a92: 5a 85        	mv	a0, s6
   11a94: 42 ec        	sd	a6, 24(sp)
   11a96: ef 50 a0 47  	jal	0x16f10 <strncpy>
   11a9a: 62 68        	ld	a6, 24(sp)
   11a9c: 29 47        	addi	a4, zero, 10
   11a9e: 33 d7 eb 02  	divu	a4, s7, a4
   11aa2: 83 46 18 00  	lbu	a3, 1(a6)
   11aa6: 42 73        	ld	t1, 48(sp)
   11aa8: e2 77        	ld	a5, 56(sp)
   11aaa: b3 36 d0 00  	snez	a3, a3
   11aae: 36 98        	add	a6, a6, a3
   11ab0: 81 46        	mv	a3, zero
   11ab2: 85 b1        	j	0x11712 <_vfprintf_r+0xfba>
   11ab4: 82 65        	ld	a1, 0(sp)
   11ab6: 10 02        	addi	a2, sp, 256
   11ab8: 4e 85        	mv	a0, s3
   11aba: ef 70 60 0c  	jal	0x18b80 <__sprint_r>
   11abe: 63 1d 05 e8  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11ac2: 6e 46        	lw	a2, 216(sp)
   11ac4: 52 67        	ld	a4, 272(sp)
   11ac6: 00 03        	addi	s0, sp, 384
   11ac8: 65 b1        	j	0x11770 <_vfprintf_r+0x1018>
   11aca: f5 67        	lui	a5, 29
   11acc: 13 8b 07 74  	addi	s6, a5, 1856
   11ad0: 81 b3        	j	0x11810 <_vfprintf_r+0x10b8>
   11ad2: 1a 89        	mv	s2, t1
   11ad4: 89 46        	addi	a3, zero, 2
   11ad6: 4a 87        	mv	a4, s2
   11ad8: 93 08 81 0e  	addi	a7, sp, 232
   11adc: 13 08 c1 0d  	addi	a6, sp, 220
   11ae0: bc 09        	addi	a5, sp, 216
   11ae2: e2 85        	mv	a1, s8
   11ae4: 66 86        	mv	a2, s9
   11ae6: 4e 85        	mv	a0, s3
   11ae8: 1a f8        	sd	t1, 48(sp)
   11aea: ef 20 30 07  	jal	0x1435c <_ldtoa_r>
   11aee: 13 07 70 06  	addi	a4, zero, 103
   11af2: 42 73        	ld	t1, 48(sp)
   11af4: 2a 8b        	mv	s6, a0
   11af6: 63 04 ed 48  	beq	s10, a4, 0x11f7e <_vfprintf_r+0x1826>
   11afa: 13 07 70 04  	addi	a4, zero, 71
   11afe: 63 00 ed 48  	beq	s10, a4, 0x11f7e <_vfprintf_r+0x1826>
   11b02: 93 7d fd fd  	andi	s11, s10, -33
   11b06: 13 07 60 04  	addi	a4, zero, 70
   11b0a: b3 0a 25 01  	add	s5, a0, s2
   11b0e: 63 8f ed 4c  	beq	s11, a4, 0x11fec <_vfprintf_r+0x1894>
   11b12: 62 85        	mv	a0, s8
   11b14: 01 46        	mv	a2, zero
   11b16: 81 46        	mv	a3, zero
   11b18: e6 85        	mv	a1, s9
   11b1a: 1a f8        	sd	t1, 48(sp)
   11b1c: ef a0 00 05  	jal	0x1bb6c <__netf2>
   11b20: 42 73        	ld	t1, 48(sp)
   11b22: 56 8c        	mv	s8, s5
   11b24: 11 cd        	beqz	a0, 0x11b40 <_vfprintf_r+0x13e8>
   11b26: 2e 7c        	ld	s8, 232(sp)
   11b28: 63 7c 5c 01  	bgeu	s8, s5, 0x11b40 <_vfprintf_r+0x13e8>
   11b2c: 93 06 00 03  	addi	a3, zero, 48
   11b30: 93 07 1c 00  	addi	a5, s8, 1
   11b34: be f5        	sd	a5, 232(sp)
   11b36: 23 00 dc 00  	sb	a3, 0(s8)
   11b3a: 2e 7c        	ld	s8, 232(sp)
   11b3c: e3 6a 5c ff  	bltu	s8, s5, 0x11b30 <_vfprintf_r+0x13d8>
   11b40: bb 07 6c 41  	subw	a5, s8, s6
   11b44: 3e fc        	sd	a5, 56(sp)
   11b46: 6e 46        	lw	a2, 216(sp)
   11b48: 13 07 70 04  	addi	a4, zero, 71
   11b4c: 32 f8        	sd	a2, 48(sp)
   11b4e: 63 8b ed 38  	beq	s11, a4, 0x11ee4 <_vfprintf_r+0x178c>
   11b52: 13 07 60 04  	addi	a4, zero, 70
   11b56: 63 88 ed 50  	beq	s11, a4, 0x12066 <_vfprintf_r+0x190e>
   11b5a: 1b 07 f6 ff  	addiw	a4, a2, -1
   11b5e: ba cd        	sw	a4, 216(sp)
   11b60: 93 06 10 06  	addi	a3, zero, 97
   11b64: 63 0a dd 6e  	beq	s10, a3, 0x12258 <_vfprintf_r+0x1b00>
   11b68: 93 05 10 04  	addi	a1, zero, 65
   11b6c: 93 06 00 05  	addi	a3, zero, 80
   11b70: 63 14 bd 38  	bne	s10, a1, 0x11ef8 <_vfprintf_r+0x17a0>
   11b74: 85 45        	addi	a1, zero, 1
   11b76: 23 00 d1 0e  	sb	a3, 224(sp)
   11b7a: 93 06 b0 02  	addi	a3, zero, 43
   11b7e: 63 56 07 00  	bgez	a4, 0x11b8a <_vfprintf_r+0x1432>
   11b82: 05 47        	addi	a4, zero, 1
   11b84: 11 9f        	subw	a4, a4, a2
   11b86: 93 06 d0 02  	addi	a3, zero, 45
   11b8a: a3 00 d1 0e  	sb	a3, 225(sp)
   11b8e: a5 46        	addi	a3, zero, 9
   11b90: 63 d8 e6 58  	bge	a3, a4, 0x12120 <_vfprintf_r+0x19c8>
   11b94: 13 08 f1 0e  	addi	a6, sp, 239
   11b98: c2 88        	mv	a7, a6
   11b9a: 29 43        	addi	t1, zero, 10
   11b9c: 13 0f 30 06  	addi	t5, zero, 99
   11ba0: bb 67 67 02  	remw	a5, a4, t1
   11ba4: 46 86        	mv	a2, a7
   11ba6: ba 86        	mv	a3, a4
   11ba8: fd 18        	addi	a7, a7, -1
   11baa: 9b 87 07 03  	addiw	a5, a5, 48
   11bae: a3 0f f6 fe  	sb	a5, -1(a2)
   11bb2: 3b 47 67 02  	divw	a4, a4, t1
   11bb6: e3 45 df fe  	blt	t5, a3, 0x11ba0 <_vfprintf_r+0x1448>
   11bba: 1b 05 07 03  	addiw	a0, a4, 48
   11bbe: a3 8f a8 fe  	sb	a0, -1(a7)
   11bc2: 13 07 e6 ff  	addi	a4, a2, -2
   11bc6: 63 74 07 6d  	bgeu	a4, a6, 0x1228e <_vfprintf_r+0x1b36>
   11bca: 93 06 21 0e  	addi	a3, sp, 226
   11bce: 83 47 07 00  	lbu	a5, 0(a4)
   11bd2: 05 07        	addi	a4, a4, 1
   11bd4: 85 06        	addi	a3, a3, 1
   11bd6: a3 8f f6 fe  	sb	a5, -1(a3)
   11bda: e3 1a 07 ff  	bne	a4, a6, 0x11bce <_vfprintf_r+0x1476>
   11bde: 13 07 11 0f  	addi	a4, sp, 241
   11be2: 93 07 21 0e  	addi	a5, sp, 226
   11be6: 11 8f        	sub	a4, a4, a2
   11be8: 3e 97        	add	a4, a4, a5
   11bea: 94 11        	addi	a3, sp, 224
   11bec: bb 07 d7 40  	subw	a5, a4, a3
   11bf0: be f8        	sd	a5, 112(sp)
   11bf2: 46 77        	ld	a4, 112(sp)
   11bf4: e2 77        	ld	a5, 56(sp)
   11bf6: 05 46        	addi	a2, zero, 1
   11bf8: bb 8d e7 00  	addw	s11, a5, a4
   11bfc: 6e 87        	mv	a4, s11
   11bfe: 63 5b f6 5c  	bge	a2, a5, 0x121d4 <_vfprintf_r+0x1a7c>
   11c02: e6 67        	ld	a5, 88(sp)
   11c04: bb 0d f7 00  	addw	s11, a4, a5
   11c08: 86 67        	ld	a5, 64(sp)
   11c0a: 13 c8 fd ff  	not	a6, s11
   11c0e: 13 58 f8 43  	srai	a6, a6, 63
   11c12: 13 f9 f7 bf  	andi	s2, a5, -1025
   11c16: 01 29        	sext.w	s2, s2
   11c18: 33 f8 0d 01  	and	a6, s11, a6
   11c1c: 13 69 09 10  	ori	s2, s2, 256
   11c20: 1b 0c 08 00  	sext.w	s8, a6
   11c24: 82 e4        	sd	zero, 72(sp)
   11c26: 82 e0        	sd	zero, 64(sp)
   11c28: 02 f8        	sd	zero, 48(sp)
   11c2a: ea 67        	ld	a5, 152(sp)
   11c2c: 63 82 07 30  	beqz	a5, 0x11f30 <_vfprintf_r+0x17d8>
   11c30: 13 07 d0 02  	addi	a4, zero, 45
   11c34: a3 07 e1 0c  	sb	a4, 207(sp)
   11c38: 01 43        	mv	t1, zero
   11c3a: 05 2c        	addiw	s8, s8, 1
   11c3c: 6f e0 1f cd  	j	0x1090c <_vfprintf_r+0x1b4>
   11c40: 82 65        	ld	a1, 0(sp)
   11c42: 10 02        	addi	a2, sp, 256
   11c44: 4e 85        	mv	a0, s3
   11c46: ef 60 b0 73  	jal	0x18b80 <__sprint_r>
   11c4a: 63 17 05 d0  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11c4e: d2 6b        	ld	s7, 272(sp)
   11c50: 00 03        	addi	s0, sp, 384
   11c52: 6f f0 0f ea  	j	0x112f2 <_vfprintf_r+0xb9a>
   11c56: 99 46        	addi	a3, zero, 6
   11c58: c5 ba        	j	0x11648 <_vfprintf_r+0xef0>
   11c5a: b3 8c 6a 41  	sub	s9, s5, s6
   11c5e: 3b 86 d7 40  	subw	a2, a5, a3
   11c62: 9b 87 0c 00  	sext.w	a5, s9
   11c66: 5e 87        	mv	a4, s7
   11c68: 63 53 f6 00  	bge	a2, a5, 0x11c6e <_vfprintf_r+0x1516>
   11c6c: b2 8c        	mv	s9, a2
   11c6e: 81 2c        	sext.w	s9, s9
   11c70: 6f f0 0f f1  	j	0x11380 <_vfprintf_r+0xc28>
   11c74: 82 1b        	slli	s7, s7, 32
   11c76: 93 db 0b 02  	srli	s7, s7, 32
   11c7a: b9 b2        	j	0x115c8 <_vfprintf_r+0xe70>
   11c7c: c2 67        	ld	a5, 16(sp)
   11c7e: 23 10 f7 00  	sh	a5, 0(a4)
   11c82: 6f f0 4f 9f  	j	0x10e76 <_vfprintf_r+0x71e>
   11c86: 82 1b        	slli	s7, s7, 32
   11c88: 93 db 0b 02  	srli	s7, s7, 32
   11c8c: 3a f0        	sd	a4, 32(sp)
   11c8e: 85 47        	addi	a5, zero, 1
   11c90: 6f f0 ef 90  	j	0x10d9e <_vfprintf_r+0x646>
   11c94: de 87        	mv	a5, s7
   11c96: 6f f0 6f 97  	j	0x10e0c <_vfprintf_r+0x6b4>
   11c9a: 82 1b        	slli	s7, s7, 32
   11c9c: 93 db 0b 02  	srli	s7, s7, 32
   11ca0: 6f f0 ef 9a  	j	0x10e4e <_vfprintf_r+0x6f6>
   11ca4: 82 65        	ld	a1, 0(sp)
   11ca6: 46 65        	ld	a0, 80(sp)
   11ca8: 10 02        	addi	a2, sp, 256
   11caa: ef 60 70 6d  	jal	0x18b80 <__sprint_r>
   11cae: 6f e0 bf d4  	j	0x109f8 <_vfprintf_r+0x2a0>
   11cb2: 93 07 d0 02  	addi	a5, zero, 45
   11cb6: a3 07 f1 0c  	sb	a5, 207(sp)
   11cba: 13 07 d0 02  	addi	a4, zero, 45
   11cbe: 3d b1        	j	0x118ec <_vfprintf_r+0x1194>
   11cc0: 82 65        	ld	a1, 0(sp)
   11cc2: 10 02        	addi	a2, sp, 256
   11cc4: 4e 85        	mv	a0, s3
   11cc6: ef 60 b0 6b  	jal	0x18b80 <__sprint_r>
   11cca: 63 17 05 c8  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11cce: 6e 46        	lw	a2, 216(sp)
   11cd0: 52 67        	ld	a4, 272(sp)
   11cd2: 83 26 81 10  	lw	a3, 264(sp)
   11cd6: 00 03        	addi	s0, sp, 384
   11cd8: a5 be        	j	0x11850 <_vfprintf_r+0x10f8>
   11cda: 13 07 80 05  	addi	a4, zero, 88
   11cde: 93 67 29 00  	ori	a5, s2, 2
   11ce2: 93 06 00 03  	addi	a3, zero, 48
   11ce6: 81 27        	sext.w	a5, a5
   11ce8: a3 08 e1 0c  	sb	a4, 209(sp)
   11cec: 23 08 d1 0c  	sb	a3, 208(sp)
   11cf0: 13 07 30 06  	addi	a4, zero, 99
   11cf4: be e0        	sd	a5, 64(sp)
   11cf6: 02 ec        	sd	zero, 24(sp)
   11cf8: 13 0b 81 11  	addi	s6, sp, 280
   11cfc: 63 40 67 48  	blt	a4, t1, 0x1217c <_vfprintf_r+0x1a24>
   11d00: 13 69 29 10  	ori	s2, s2, 258
   11d04: ee 7c        	ld	s9, 248(sp)
   11d06: 9b 07 09 00  	sext.w	a5, s2
   11d0a: 3e e9        	sd	a5, 144(sp)
   11d0c: 02 ed        	sd	zero, 152(sp)
   11d0e: 4e 7c        	ld	s8, 240(sp)
   11d10: 63 c1 0c 28  	bltz	s9, 0x11f92 <_vfprintf_r+0x183a>
   11d14: 13 07 10 06  	addi	a4, zero, 97
   11d18: 63 04 ed 00  	beq	s10, a4, 0x11d20 <_vfprintf_r+0x15c8>
   11d1c: 6f e0 1f e1  	j	0x10b2c <_vfprintf_r+0x3d4>
   11d20: e6 85        	mv	a1, s9
   11d22: 62 85        	mv	a0, s8
   11d24: 1a f8        	sd	t1, 48(sp)
   11d26: ef b0 40 16  	jal	0x1ce8a <__trunctfdf2>
   11d2a: a8 09        	addi	a0, sp, 216
   11d2c: ef 40 f0 5d  	jal	0x16b0a <frexp>
   11d30: ef b0 60 09  	jal	0x1cdc6 <__extenddftf2>
   11d34: 79 67        	lui	a4, 30
   11d36: 83 36 87 9c  	ld	a3, -1592(a4)
   11d3a: 01 46        	mv	a2, zero
   11d3c: ef a0 a0 04  	jal	0x1bd86 <__multf3>
   11d40: 01 46        	mv	a2, zero
   11d42: 81 46        	mv	a3, zero
   11d44: aa 8d        	mv	s11, a0
   11d46: 2e 89        	mv	s2, a1
   11d48: ef 90 50 62  	jal	0x1bb6c <__netf2>
   11d4c: 42 73        	ld	t1, 48(sp)
   11d4e: 19 e1        	bnez	a0, 0x11d54 <_vfprintf_r+0x15fc>
   11d50: 05 47        	addi	a4, zero, 1
   11d52: ba cd        	sw	a4, 216(sp)
   11d54: 75 67        	lui	a4, 29
   11d56: 93 07 87 77  	addi	a5, a4, 1912
   11d5a: 3e fc        	sd	a5, 56(sp)
   11d5c: f9 66        	lui	a3, 30
   11d5e: 83 b7 86 9d  	ld	a5, -1576(a3)
   11d62: 13 17 03 02  	slli	a4, t1, 32
   11d66: 01 93        	srli	a4, a4, 32
   11d68: 05 07        	addi	a4, a4, 1
   11d6a: 3e f8        	sd	a5, 48(sp)
   11d6c: 1b 06 03 00  	sext.w	a2, t1
   11d70: b3 07 eb 00  	add	a5, s6, a4
   11d74: a6 e4        	sd	s1, 72(sp)
   11d76: 52 f1        	sd	s4, 160(sp)
   11d78: 22 f9        	sd	s0, 176(sp)
   11d7a: 1a f5        	sd	t1, 168(sp)
   11d7c: 62 74        	ld	s0, 56(sp)
   11d7e: be 84        	mv	s1, a5
   11d80: 5a 8a        	mv	s4, s6
   11d82: 32 fd        	sd	a2, 184(sp)
   11d84: ce 8b        	mv	s7, s3
   11d86: da 8c        	mv	s9, s6
   11d88: 39 a0        	j	0x11d96 <_vfprintf_r+0x163e>
   11d8a: 01 46        	mv	a2, zero
   11d8c: 81 46        	mv	a3, zero
   11d8e: ef 90 f0 5d  	jal	0x1bb6c <__netf2>
   11d92: 63 04 05 40  	beqz	a0, 0x1219a <_vfprintf_r+0x1a42>
   11d96: c2 76        	ld	a3, 48(sp)
   11d98: 01 46        	mv	a2, zero
   11d9a: 6e 85        	mv	a0, s11
   11d9c: ca 85        	mv	a1, s2
   11d9e: ef 90 90 7e  	jal	0x1bd86 <__multf3>
   11da2: 2e 89        	mv	s2, a1
   11da4: aa 89        	mv	s3, a0
   11da6: ef a0 b0 73  	jal	0x1cce0 <__fixtfsi>
   11daa: 1b 0c 05 00  	sext.w	s8, a0
   11dae: 62 85        	mv	a0, s8
   11db0: ef a0 10 7c  	jal	0x1cd70 <__floatsitf>
   11db4: ae 86        	mv	a3, a1
   11db6: 2a 86        	mv	a2, a0
   11db8: ca 85        	mv	a1, s2
   11dba: 4e 85        	mv	a0, s3
   11dbc: ef a0 c0 5e  	jal	0x1c3a8 <__subtf3>
   11dc0: 33 07 84 01  	add	a4, s0, s8
   11dc4: 83 46 07 00  	lbu	a3, 0(a4)
   11dc8: d2 89        	mv	s3, s4
   11dca: 05 0a        	addi	s4, s4, 1
   11dcc: a3 0f da fe  	sb	a3, -1(s4)
   11dd0: aa 8a        	mv	s5, a0
   11dd2: aa 8d        	mv	s11, a0
   11dd4: 2e 89        	mv	s2, a1
   11dd6: e3 9a 44 fb  	bne	s1, s4, 0x11d8a <_vfprintf_r+0x1632>
   11dda: d2 8a        	mv	s5, s4
   11ddc: a6 64        	ld	s1, 72(sp)
   11dde: 2a 73        	ld	t1, 168(sp)
   11de0: 4a 74        	ld	s0, 176(sp)
   11de2: 0a 7a        	ld	s4, 160(sp)
   11de4: 4e f8        	sd	s3, 48(sp)
   11de6: 66 8b        	mv	s6, s9
   11de8: de 89        	mv	s3, s7
   11dea: ae 8c        	mv	s9, a1
   11dec: aa 8b        	mv	s7, a0
   11dee: fd 5d        	addi	s11, zero, -1
   11df0: f9 66        	lui	a3, 30
   11df2: 03 b9 86 9e  	ld	s2, -1560(a3)
   11df6: 01 46        	mv	a2, zero
   11df8: 5e 85        	mv	a0, s7
   11dfa: ca 86        	mv	a3, s2
   11dfc: e6 85        	mv	a1, s9
   11dfe: 9a e4        	sd	t1, 72(sp)
   11e00: ef 90 70 60  	jal	0x1bc06 <__gttf2>
   11e04: 26 63        	ld	t1, 72(sp)
   11e06: 63 4f a0 00  	bgtz	a0, 0x11e24 <_vfprintf_r+0x16cc>
   11e0a: 01 46        	mv	a2, zero
   11e0c: ca 86        	mv	a3, s2
   11e0e: 5e 85        	mv	a0, s7
   11e10: e6 85        	mv	a1, s9
   11e12: ef 90 b0 55  	jal	0x1bb6c <__netf2>
   11e16: 26 63        	ld	t1, 72(sp)
   11e18: 63 14 05 0e  	bnez	a0, 0x11f00 <_vfprintf_r+0x17a8>
   11e1c: 13 7c 1c 00  	andi	s8, s8, 1
   11e20: 63 00 0c 0e  	beqz	s8, 0x11f00 <_vfprintf_r+0x17a8>
   11e24: c2 77        	ld	a5, 48(sp)
   11e26: d6 86        	mv	a3, s5
   11e28: be f5        	sd	a5, 232(sp)
   11e2a: e2 77        	ld	a5, 56(sp)
   11e2c: 03 c6 fa ff  	lbu	a2, -1(s5)
   11e30: 83 c5 f7 00  	lbu	a1, 15(a5)
   11e34: 63 9e c5 00  	bne	a1, a2, 0x11e50 <_vfprintf_r+0x16f8>
   11e38: 13 05 00 03  	addi	a0, zero, 48
   11e3c: a3 8f a6 fe  	sb	a0, -1(a3)
   11e40: ae 76        	ld	a3, 232(sp)
   11e42: 93 87 f6 ff  	addi	a5, a3, -1
   11e46: be f5        	sd	a5, 232(sp)
   11e48: 03 c6 f6 ff  	lbu	a2, -1(a3)
   11e4c: e3 88 c5 fe  	beq	a1, a2, 0x11e3c <_vfprintf_r+0x16e4>
   11e50: 9b 05 16 00  	addiw	a1, a2, 1
   11e54: 13 05 90 03  	addi	a0, zero, 57
   11e58: 93 f5 f5 0f  	andi	a1, a1, 255
   11e5c: 63 0c a6 02  	beq	a2, a0, 0x11e94 <_vfprintf_r+0x173c>
   11e60: a3 8f b6 fe  	sb	a1, -1(a3)
   11e64: bb 87 6a 41  	subw	a5, s5, s6
   11e68: 3e fc        	sd	a5, 56(sp)
   11e6a: 93 7d fd fd  	andi	s11, s10, -33
   11e6e: e1 b9        	j	0x11b46 <_vfprintf_r+0x13ee>
   11e70: 82 65        	ld	a1, 0(sp)
   11e72: 10 02        	addi	a2, sp, 256
   11e74: 4e 85        	mv	a0, s3
   11e76: ef 60 b0 50  	jal	0x18b80 <__sprint_r>
   11e7a: 63 1f 05 ac  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11e7e: d2 6b        	ld	s7, 272(sp)
   11e80: 00 03        	addi	s0, sp, 384
   11e82: 6f f0 4f c8  	j	0x11306 <_vfprintf_r+0xbae>
   11e86: 1b 09 13 00  	addiw	s2, t1, 1
   11e8a: 89 46        	addi	a3, zero, 2
   11e8c: a9 b1        	j	0x11ad6 <_vfprintf_r+0x137e>
   11e8e: 1a 89        	mv	s2, t1
   11e90: 8d 46        	addi	a3, zero, 3
   11e92: 91 b1        	j	0x11ad6 <_vfprintf_r+0x137e>
   11e94: e2 77        	ld	a5, 56(sp)
   11e96: 83 c5 a7 00  	lbu	a1, 10(a5)
   11e9a: d9 b7        	j	0x11e60 <_vfprintf_r+0x1708>
   11e9c: e6 85        	mv	a1, s9
   11e9e: 62 85        	mv	a0, s8
   11ea0: 1a f8        	sd	t1, 48(sp)
   11ea2: ef a0 90 7e  	jal	0x1ce8a <__trunctfdf2>
   11ea6: a8 09        	addi	a0, sp, 216
   11ea8: ef 40 30 46  	jal	0x16b0a <frexp>
   11eac: ef a0 b0 71  	jal	0x1cdc6 <__extenddftf2>
   11eb0: 79 67        	lui	a4, 30
   11eb2: 83 36 87 9c  	ld	a3, -1592(a4)
   11eb6: 01 46        	mv	a2, zero
   11eb8: ef 90 f0 6c  	jal	0x1bd86 <__multf3>
   11ebc: 01 46        	mv	a2, zero
   11ebe: 81 46        	mv	a3, zero
   11ec0: aa 8d        	mv	s11, a0
   11ec2: 2e 89        	mv	s2, a1
   11ec4: ef 90 90 4a  	jal	0x1bb6c <__netf2>
   11ec8: 42 73        	ld	t1, 48(sp)
   11eca: 19 e1        	bnez	a0, 0x11ed0 <_vfprintf_r+0x1778>
   11ecc: 05 47        	addi	a4, zero, 1
   11ece: ba cd        	sw	a4, 216(sp)
   11ed0: 75 67        	lui	a4, 29
   11ed2: 93 07 07 76  	addi	a5, a4, 1888
   11ed6: 3e fc        	sd	a5, 56(sp)
   11ed8: 51 b5        	j	0x11d5c <_vfprintf_r+0x1604>
   11eda: 2e 7c        	ld	s8, 232(sp)
   11edc: 6e 46        	lw	a2, 216(sp)
   11ede: bb 07 6c 41  	subw	a5, s8, s6
   11ee2: 3e fc        	sd	a5, 56(sp)
   11ee4: 75 57        	addi	a4, zero, -3
   11ee6: 32 f8        	sd	a2, 48(sp)
   11ee8: 63 44 e6 00  	blt	a2, a4, 0x11ef0 <_vfprintf_r+0x1798>
   11eec: 63 5a c3 04  	bge	t1, a2, 0x11f40 <_vfprintf_r+0x17e8>
   11ef0: 1b 07 f6 ff  	addiw	a4, a2, -1
   11ef4: 79 3d        	addiw	s10, s10, -2
   11ef6: ba cd        	sw	a4, 216(sp)
   11ef8: 93 76 fd 0f  	andi	a3, s10, 255
   11efc: 81 45        	mv	a1, zero
   11efe: a5 b9        	j	0x11b76 <_vfprintf_r+0x141e>
   11f00: e3 c2 0d f6  	bltz	s11, 0x11e64 <_vfprintf_r+0x170c>
   11f04: 85 2d        	addiw	s11, s11, 1
   11f06: d7 75 00 04  	<unknown>
   11f0a: 93 96 0d 02  	slli	a3, s11, 32
   11f0e: 93 05 00 03  	addi	a1, zero, 48
   11f12: 81 92        	srli	a3, a3, 32
   11f14: 56 86        	mv	a2, s5
   11f16: 73 25 20 c2  	csrr	a0, vlenb
   11f1a: 57 cc 05 5e  	<unknown>
   11f1e: d7 f7 06 04  	<unknown>
   11f22: 27 0c 06 02  	<unknown>
   11f26: 9d 8e        	sub	a3, a3, a5
   11f28: 2a 96        	add	a2, a2, a0
   11f2a: f5 fa        	bnez	a3, 0x11f1e <_vfprintf_r+0x17c6>
   11f2c: ee 9a        	add	s5, s5, s11
   11f2e: 1d bf        	j	0x11e64 <_vfprintf_r+0x170c>
   11f30: 03 47 f1 0c  	lbu	a4, 207(sp)
   11f34: 01 43        	mv	t1, zero
   11f36: 19 c3        	beqz	a4, 0x11f3c <_vfprintf_r+0x17e4>
   11f38: 6f e0 bf d6  	j	0x10ca2 <_vfprintf_r+0x54a>
   11f3c: 6f e0 1f 9d  	j	0x1090c <_vfprintf_r+0x1b4>
   11f40: e2 77        	ld	a5, 56(sp)
   11f42: 63 49 f6 1a  	blt	a2, a5, 0x120f4 <_vfprintf_r+0x199c>
   11f46: 86 67        	ld	a5, 64(sp)
   11f48: b2 8d        	mv	s11, a2
   11f4a: 13 f7 17 00  	andi	a4, a5, 1
   11f4e: 01 c7        	beqz	a4, 0x11f56 <_vfprintf_r+0x17fe>
   11f50: e6 67        	ld	a5, 88(sp)
   11f52: bb 8d c7 00  	addw	s11, a5, a2
   11f56: 86 67        	ld	a5, 64(sp)
   11f58: 13 f7 07 40  	andi	a4, a5, 1024
   11f5c: 19 c3        	beqz	a4, 0x11f62 <_vfprintf_r+0x180a>
   11f5e: 63 41 c0 28  	bgtz	a2, 0x121e0 <_vfprintf_r+0x1a88>
   11f62: 13 c8 fd ff  	not	a6, s11
   11f66: 13 58 f8 43  	srai	a6, a6, 63
   11f6a: 4a 69        	ld	s2, 144(sp)
   11f6c: 33 f8 0d 01  	and	a6, s11, a6
   11f70: 1b 0c 08 00  	sext.w	s8, a6
   11f74: 13 0d 70 06  	addi	s10, zero, 103
   11f78: 82 e4        	sd	zero, 72(sp)
   11f7a: 82 e0        	sd	zero, 64(sp)
   11f7c: 7d b1        	j	0x11c2a <_vfprintf_r+0x14d2>
   11f7e: 86 67        	ld	a5, 64(sp)
   11f80: 13 f7 17 00  	andi	a4, a5, 1
   11f84: 39 db        	beqz	a4, 0x11eda <_vfprintf_r+0x1782>
   11f86: b3 0a 2b 01  	add	s5, s6, s2
   11f8a: 93 0d 70 04  	addi	s11, zero, 71
   11f8e: 51 b6        	j	0x11b12 <_vfprintf_r+0x13ba>
   11f90: 02 ec        	sd	zero, 24(sp)
   11f92: 7d 57        	addi	a4, zero, -1
   11f94: 7e 17        	slli	a4, a4, 63
   11f96: 93 07 d0 02  	addi	a5, zero, 45
   11f9a: b3 cc ec 00  	xor	s9, s9, a4
   11f9e: 3e ed        	sd	a5, 152(sp)
   11fa0: 95 bb        	j	0x11d14 <_vfprintf_r+0x15bc>
   11fa2: 10 02        	addi	a2, sp, 256
   11fa4: e6 85        	mv	a1, s9
   11fa6: 4e 85        	mv	a0, s3
   11fa8: ef 60 90 3d  	jal	0x18b80 <__sprint_r>
   11fac: 63 16 05 9a  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11fb0: 03 46 0a 00  	lbu	a2, 0(s4)
   11fb4: 52 67        	ld	a4, 272(sp)
   11fb6: 00 03        	addi	s0, sp, 384
   11fb8: 1d 43        	addi	t1, zero, 7
   11fba: 32 9b        	add	s6, s6, a2
   11fbc: f5 ba        	j	0x119b8 <_vfprintf_r+0x1260>
   11fbe: 13 07 80 07  	addi	a4, zero, 120
   11fc2: 31 bb        	j	0x11cde <_vfprintf_r+0x1586>
   11fc4: 93 07 d0 02  	addi	a5, zero, 45
   11fc8: a3 07 f1 0c  	sb	a5, 207(sp)
   11fcc: 13 07 d0 02  	addi	a4, zero, 45
   11fd0: 6f f0 3f 83  	j	0x11802 <_vfprintf_r+0x10aa>
   11fd4: 82 65        	ld	a1, 0(sp)
   11fd6: 10 02        	addi	a2, sp, 256
   11fd8: 4e 85        	mv	a0, s3
   11fda: ef 60 70 3a  	jal	0x18b80 <__sprint_r>
   11fde: 63 1d 05 96  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   11fe2: ee 46        	lw	a3, 216(sp)
   11fe4: 52 67        	ld	a4, 272(sp)
   11fe6: 00 03        	addi	s0, sp, 384
   11fe8: 6f f0 0f b6  	j	0x11348 <_vfprintf_r+0xbf0>
   11fec: 83 46 05 00  	lbu	a3, 0(a0)
   11ff0: 13 07 00 03  	addi	a4, zero, 48
   11ff4: 63 85 e6 00  	beq	a3, a4, 0x11ffe <_vfprintf_r+0x18a6>
   11ff8: ee 46        	lw	a3, 216(sp)
   11ffa: b6 9a        	add	s5, s5, a3
   11ffc: 19 be        	j	0x11b12 <_vfprintf_r+0x13ba>
   11ffe: 01 46        	mv	a2, zero
   12000: 81 46        	mv	a3, zero
   12002: 62 85        	mv	a0, s8
   12004: e6 85        	mv	a1, s9
   12006: 1a f8        	sd	t1, 48(sp)
   12008: ef 90 50 36  	jal	0x1bb6c <__netf2>
   1200c: 42 73        	ld	t1, 48(sp)
   1200e: 6d d5        	beqz	a0, 0x11ff8 <_vfprintf_r+0x18a0>
   12010: 05 47        	addi	a4, zero, 1
   12012: bb 06 27 41  	subw	a3, a4, s2
   12016: b6 cd        	sw	a3, 216(sp)
   12018: b6 9a        	add	s5, s5, a3
   1201a: e5 bc        	j	0x11b12 <_vfprintf_r+0x13ba>
   1201c: 03 47 f1 0c  	lbu	a4, 207(sp)
   12020: 56 f0        	sd	s5, 32(sp)
   12022: 82 e4        	sd	zero, 72(sp)
   12024: 82 e0        	sd	zero, 64(sp)
   12026: 02 f8        	sd	zero, 48(sp)
   12028: 1a 8c        	mv	s8, t1
   1202a: 9a 8d        	mv	s11, t1
   1202c: 01 43        	mv	t1, zero
   1202e: 19 c3        	beqz	a4, 0x12034 <_vfprintf_r+0x18dc>
   12030: 6f e0 3f c7  	j	0x10ca2 <_vfprintf_r+0x54a>
   12034: 6f e0 9f 8d  	j	0x1090c <_vfprintf_r+0x1b4>
   12038: f5 67        	lui	a5, 29
   1203a: 13 8b 07 75  	addi	s6, a5, 1872
   1203e: 6f f0 2f fd  	j	0x11810 <_vfprintf_r+0x10b8>
   12042: 82 65        	ld	a1, 0(sp)
   12044: 10 02        	addi	a2, sp, 256
   12046: 4e 85        	mv	a0, s3
   12048: ef 60 90 33  	jal	0x18b80 <__sprint_r>
   1204c: 63 16 05 90  	bnez	a0, 0x11158 <_vfprintf_r+0xa00>
   12050: 6e 46        	lw	a2, 216(sp)
   12052: e2 77        	ld	a5, 56(sp)
   12054: 52 67        	ld	a4, 272(sp)
   12056: 00 03        	addi	s0, sp, 384
   12058: 3b 86 c7 40  	subw	a2, a5, a2
   1205c: 6f f0 4f b2  	j	0x11380 <_vfprintf_r+0xc28>
   12060: ca 8c        	mv	s9, s2
   12062: 6f f0 af 9a  	j	0x1120c <_vfprintf_r+0xab4>
   12066: 86 67        	ld	a5, 64(sp)
   12068: 13 f7 17 00  	andi	a4, a5, 1
   1206c: c2 77        	ld	a5, 48(sp)
   1206e: 33 67 e3 00  	or	a4, t1, a4
   12072: 63 56 f0 1e  	blez	a5, 0x1225e <_vfprintf_r+0x1b06>
   12076: 63 19 07 1c  	bnez	a4, 0x12248 <_vfprintf_r+0x1af0>
   1207a: c2 7d        	ld	s11, 48(sp)
   1207c: 13 0d 60 06  	addi	s10, zero, 102
   12080: 86 67        	ld	a5, 64(sp)
   12082: 13 f7 07 40  	andi	a4, a5, 1024
   12086: 63 1f 07 14  	bnez	a4, 0x121e4 <_vfprintf_r+0x1a8c>
   1208a: 13 c8 fd ff  	not	a6, s11
   1208e: 13 58 f8 43  	srai	a6, a6, 63
   12092: 33 f8 0d 01  	and	a6, s11, a6
   12096: 4a 69        	ld	s2, 144(sp)
   12098: 1b 0c 08 00  	sext.w	s8, a6
   1209c: f1 bd        	j	0x11f78 <_vfprintf_r+0x1820>
   1209e: 79 6f        	lui	t5, 30
   120a0: c1 55        	addi	a1, zero, -16
   120a2: bb 0c c0 40  	negw	s9, a2
   120a6: 93 0b 0f 9a  	addi	s7, t5, -1632
   120aa: 41 4d        	addi	s10, zero, 16
   120ac: 9d 4d        	addi	s11, zero, 7
   120ae: 63 46 b6 00  	blt	a2, a1, 0x120ba <_vfprintf_r+0x1962>
   120b2: 59 a8        	j	0x12148 <_vfprintf_r+0x19f0>
   120b4: c1 3c        	addiw	s9, s9, -16
   120b6: 63 59 9d 09  	bge	s10, s9, 0x12148 <_vfprintf_r+0x19f0>
   120ba: 85 26        	addiw	a3, a3, 1
   120bc: 41 07        	addi	a4, a4, 16
   120be: 23 30 74 01  	sd	s7, 0(s0)
   120c2: 23 34 a4 01  	sd	s10, 8(s0)
   120c6: 3a ea        	sd	a4, 272(sp)
   120c8: 23 24 d1 10  	sw	a3, 264(sp)
   120cc: 41 04        	addi	s0, s0, 16
   120ce: e3 d3 dd fe  	bge	s11, a3, 0x120b4 <_vfprintf_r+0x195c>
   120d2: 82 65        	ld	a1, 0(sp)
   120d4: 10 02        	addi	a2, sp, 256
   120d6: 4e 85        	mv	a0, s3
   120d8: ef 60 90 2a  	jal	0x18b80 <__sprint_r>
   120dc: 19 c1        	beqz	a0, 0x120e2 <_vfprintf_r+0x198a>
   120de: 6f f0 af 87  	j	0x11158 <_vfprintf_r+0xa00>
   120e2: 52 67        	ld	a4, 272(sp)
   120e4: 83 26 81 10  	lw	a3, 264(sp)
   120e8: 00 03        	addi	s0, sp, 384
   120ea: e9 b7        	j	0x120b4 <_vfprintf_r+0x195c>
   120ec: fd 57        	addi	a5, zero, -1
   120ee: 3e e8        	sd	a5, 16(sp)
   120f0: 6f e0 9f 91  	j	0x10a08 <_vfprintf_r+0x2b0>
   120f4: e6 67        	ld	a5, 88(sp)
   120f6: 62 77        	ld	a4, 56(sp)
   120f8: 13 0d 70 06  	addi	s10, zero, 103
   120fc: bb 8d e7 00  	addw	s11, a5, a4
   12100: c2 77        	ld	a5, 48(sp)
   12102: e3 4f f0 f6  	bgtz	a5, 0x12080 <_vfprintf_r+0x1928>
   12106: 3b 87 fd 40  	subw	a4, s11, a5
   1210a: 05 27        	addiw	a4, a4, 1
   1210c: 9b 0d 07 00  	sext.w	s11, a4
   12110: 13 c6 fd ff  	not	a2, s11
   12114: 7d 96        	srai	a2, a2, 63
   12116: 71 8f        	and	a4, a4, a2
   12118: 4a 69        	ld	s2, 144(sp)
   1211a: 1b 0c 07 00  	sext.w	s8, a4
   1211e: a9 bd        	j	0x11f78 <_vfprintf_r+0x1820>
   12120: 93 06 21 0e  	addi	a3, sp, 226
   12124: 99 e5        	bnez	a1, 0x12132 <_vfprintf_r+0x19da>
   12126: 93 06 00 03  	addi	a3, zero, 48
   1212a: 23 01 d1 0e  	sb	a3, 226(sp)
   1212e: 93 06 31 0e  	addi	a3, sp, 227
   12132: 1c 04        	addi	a5, sp, 512
   12134: 33 86 f6 40  	sub	a2, a3, a5
   12138: 1b 07 07 03  	addiw	a4, a4, 48
   1213c: 9b 07 16 12  	addiw	a5, a2, 289
   12140: 23 80 e6 00  	sb	a4, 0(a3)
   12144: be f8        	sd	a5, 112(sp)
   12146: 75 b4        	j	0x11bf2 <_vfprintf_r+0x149a>
   12148: 85 26        	addiw	a3, a3, 1
   1214a: 66 97        	add	a4, a4, s9
   1214c: 23 30 74 01  	sd	s7, 0(s0)
   12150: 23 34 94 01  	sd	s9, 8(s0)
   12154: 3a ea        	sd	a4, 272(sp)
   12156: 23 24 d1 10  	sw	a3, 264(sp)
   1215a: 1d 46        	addi	a2, zero, 7
   1215c: 63 50 d6 e4  	bge	a2, a3, 0x1179c <_vfprintf_r+0x1044>
   12160: 82 65        	ld	a1, 0(sp)
   12162: 10 02        	addi	a2, sp, 256
   12164: 4e 85        	mv	a0, s3
   12166: ef 60 b0 21  	jal	0x18b80 <__sprint_r>
   1216a: 19 c1        	beqz	a0, 0x12170 <_vfprintf_r+0x1a18>
   1216c: 6f e0 df fe  	j	0x11158 <_vfprintf_r+0xa00>
   12170: 52 67        	ld	a4, 272(sp)
   12172: 83 26 81 10  	lw	a3, 264(sp)
   12176: 00 03        	addi	s0, sp, 384
   12178: 6f f0 cf ed  	j	0x11854 <_vfprintf_r+0x10fc>
   1217c: 9b 05 13 00  	addiw	a1, t1, 1
   12180: 4e 85        	mv	a0, s3
   12182: 1a ec        	sd	t1, 24(sp)
   12184: ef 30 60 59  	jal	0x1571a <_malloc_r>
   12188: 62 63        	ld	t1, 24(sp)
   1218a: 2a 8b        	mv	s6, a0
   1218c: 63 06 05 12  	beqz	a0, 0x122b8 <_vfprintf_r+0x1b60>
   12190: 2a ec        	sd	a0, 24(sp)
   12192: bd b6        	j	0x11d00 <_vfprintf_r+0x15a8>
   12194: 3a f0        	sd	a4, 32(sp)
   12196: 6f f0 6f 87  	j	0x1120c <_vfprintf_r+0xab4>
   1219a: 6a 76        	ld	a2, 184(sp)
   1219c: 66 8b        	mv	s6, s9
   1219e: 4e f8        	sd	s3, 48(sp)
   121a0: 3b 06 cb 00  	addw	a2, s6, a2
   121a4: de 89        	mv	s3, s7
   121a6: d6 8b        	mv	s7, s5
   121a8: d2 8a        	mv	s5, s4
   121aa: a6 64        	ld	s1, 72(sp)
   121ac: 2a 73        	ld	t1, 168(sp)
   121ae: 4a 74        	ld	s0, 176(sp)
   121b0: 0a 7a        	ld	s4, 160(sp)
   121b2: ca 8c        	mv	s9, s2
   121b4: bb 0d 56 41  	subw	s11, a2, s5
   121b8: 25 b9        	j	0x11df0 <_vfprintf_r+0x1698>
   121ba: 1c c3        	sw	a5, 0(a4)
   121bc: 6f e0 bf cb  	j	0x10e76 <_vfprintf_r+0x71e>
   121c0: 63 04 03 00  	beqz	t1, 0x121c8 <_vfprintf_r+0x1a70>
   121c4: 6f e0 df 94  	j	0x10b10 <_vfprintf_r+0x3b8>
   121c8: 05 43        	addi	t1, zero, 1
   121ca: 6f e0 7f 94  	j	0x10b10 <_vfprintf_r+0x3b8>
   121ce: 19 43        	addi	t1, zero, 6
   121d0: 6f e0 1f 94  	j	0x10b10 <_vfprintf_r+0x3b8>
   121d4: 86 67        	ld	a5, 64(sp)
   121d6: 13 f6 17 00  	andi	a2, a5, 1
   121da: e3 07 06 a2  	beqz	a2, 0x11c08 <_vfprintf_r+0x14b0>
   121de: 15 b4        	j	0x11c02 <_vfprintf_r+0x14aa>
   121e0: 13 0d 70 06  	addi	s10, zero, 103
   121e4: 8a 66        	ld	a3, 128(sp)
   121e6: 13 06 f0 0f  	addi	a2, zero, 255
   121ea: 03 c7 06 00  	lbu	a4, 0(a3)
   121ee: 63 02 c7 0c  	beq	a4, a2, 0x122b2 <_vfprintf_r+0x1b5a>
   121f2: c2 77        	ld	a5, 48(sp)
   121f4: 01 48        	mv	a6, zero
   121f6: 01 45        	mv	a0, zero
   121f8: 93 05 f0 0f  	addi	a1, zero, 255
   121fc: 1b 06 07 00  	sext.w	a2, a4
   12200: 63 5a f7 00  	bge	a4, a5, 0x12214 <_vfprintf_r+0x1abc>
   12204: 03 c7 16 00  	lbu	a4, 1(a3)
   12208: 91 9f        	subw	a5, a5, a2
   1220a: 1d cb        	beqz	a4, 0x12240 <_vfprintf_r+0x1ae8>
   1220c: 05 25        	addiw	a0, a0, 1
   1220e: 85 06        	addi	a3, a3, 1
   12210: e3 16 b7 fe  	bne	a4, a1, 0x121fc <_vfprintf_r+0x1aa4>
   12214: 3e f8        	sd	a5, 48(sp)
   12216: 36 e1        	sd	a3, 128(sp)
   12218: aa e0        	sd	a0, 64(sp)
   1221a: c2 e4        	sd	a6, 72(sp)
   1221c: a6 67        	ld	a5, 72(sp)
   1221e: 06 67        	ld	a4, 64(sp)
   12220: 4a 69        	ld	s2, 144(sp)
   12222: 3d 9f        	addw	a4, a4, a5
   12224: aa 67        	ld	a5, 136(sp)
   12226: 3b 07 f7 02  	mulw	a4, a4, a5
   1222a: 3b 07 b7 01  	addw	a4, a4, s11
   1222e: 9b 0d 07 00  	sext.w	s11, a4
   12232: 13 c6 fd ff  	not	a2, s11
   12236: 7d 96        	srai	a2, a2, 63
   12238: 71 8f        	and	a4, a4, a2
   1223a: 1b 0c 07 00  	sext.w	s8, a4
   1223e: f5 b2        	j	0x11c2a <_vfprintf_r+0x14d2>
   12240: 03 c7 06 00  	lbu	a4, 0(a3)
   12244: 05 28        	addiw	a6, a6, 1
   12246: e9 b7        	j	0x12210 <_vfprintf_r+0x1ab8>
   12248: 66 67        	ld	a4, 88(sp)
   1224a: 13 0d 60 06  	addi	s10, zero, 102
   1224e: bb 06 f7 00  	addw	a3, a4, a5
   12252: bb 8d 66 00  	addw	s11, a3, t1
   12256: 2d b5        	j	0x12080 <_vfprintf_r+0x1928>
   12258: 93 06 00 07  	addi	a3, zero, 112
   1225c: 21 ba        	j	0x11b74 <_vfprintf_r+0x141c>
   1225e: 19 e7        	bnez	a4, 0x1226c <_vfprintf_r+0x1b14>
   12260: 4a 69        	ld	s2, 144(sp)
   12262: 05 4c        	addi	s8, zero, 1
   12264: 13 0d 60 06  	addi	s10, zero, 102
   12268: 85 4d        	addi	s11, zero, 1
   1226a: 39 b3        	j	0x11f78 <_vfprintf_r+0x1820>
   1226c: e6 67        	ld	a5, 88(sp)
   1226e: 4a 69        	ld	s2, 144(sp)
   12270: 13 0d 60 06  	addi	s10, zero, 102
   12274: 1b 87 17 00  	addiw	a4, a5, 1
   12278: 3b 07 67 00  	addw	a4, a4, t1
   1227c: 9b 0d 07 00  	sext.w	s11, a4
   12280: 13 c6 fd ff  	not	a2, s11
   12284: 7d 96        	srai	a2, a2, 63
   12286: 71 8f        	and	a4, a4, a2
   12288: 1b 0c 07 00  	sext.w	s8, a4
   1228c: f5 b1        	j	0x11f78 <_vfprintf_r+0x1820>
   1228e: 89 47        	addi	a5, zero, 2
   12290: be f8        	sd	a5, 112(sp)
   12292: 85 b2        	j	0x11bf2 <_vfprintf_r+0x149a>
   12294: 82 76        	ld	a3, 32(sp)
   12296: 98 42        	lw	a4, 0(a3)
   12298: a1 06        	addi	a3, a3, 8
   1229a: 3a 83        	mv	t1, a4
   1229c: 63 53 07 00  	bgez	a4, 0x122a2 <_vfprintf_r+0x1b4a>
   122a0: 7d 53        	addi	t1, zero, -1
   122a2: 03 ce 14 00  	lbu	t3, 1(s1)
   122a6: 9b 0a 03 00  	sext.w	s5, t1
   122aa: 36 f0        	sd	a3, 32(sp)
   122ac: be 84        	mv	s1, a5
   122ae: 6f e0 8f de  	j	0x10896 <_vfprintf_r+0x13e>
   122b2: 82 e4        	sd	zero, 72(sp)
   122b4: 82 e0        	sd	zero, 64(sp)
   122b6: 9d b7        	j	0x1221c <_vfprintf_r+0x1ac4>
   122b8: 02 67        	ld	a4, 0(sp)
   122ba: 83 57 07 01  	lhu	a5, 16(a4)
   122be: 93 e7 07 04  	ori	a5, a5, 64
   122c2: 23 18 f7 00  	sh	a5, 16(a4)
   122c6: 6f e0 2f f3  	j	0x109f8 <_vfprintf_r+0x2a0>

00000000000122ca <vfprintf>:
   122ca: aa 87        	mv	a5, a0
   122cc: 03 b5 81 85  	ld	a0, -1960(gp)
   122d0: b2 86        	mv	a3, a2
   122d2: 2e 86        	mv	a2, a1
   122d4: be 85        	mv	a1, a5
   122d6: 6f e0 2f c8  	j	0x10758 <_vfprintf_r>

00000000000122da <__sbprintf>:
   122da: 83 d7 05 01  	lhu	a5, 16(a1)
   122de: 03 ae c5 0a  	lw	t3, 172(a1)
   122e2: 03 d3 25 01  	lhu	t1, 18(a1)
   122e6: 83 b8 05 03  	ld	a7, 48(a1)
   122ea: 03 b8 05 04  	ld	a6, 64(a1)
   122ee: 13 01 01 b3  	addi	sp, sp, -1232
   122f2: f5 9b        	andi	a5, a5, -3
   122f4: 13 07 00 40  	addi	a4, zero, 1024
   122f8: 23 30 81 4c  	sd	s0, 1216(sp)
   122fc: 23 18 f1 00  	sh	a5, 16(sp)
   12300: 2e 84        	mv	s0, a1
   12302: 1c 19        	addi	a5, sp, 176
   12304: 8a 85        	mv	a1, sp
   12306: 23 3c 91 4a  	sd	s1, 1208(sp)
   1230a: 23 38 21 4b  	sd	s2, 1200(sp)
   1230e: 23 34 11 4c  	sd	ra, 1224(sp)
   12312: 2a 89        	mv	s2, a0
   12314: 72 d7        	sw	t3, 172(sp)
   12316: 23 19 61 00  	sh	t1, 18(sp)
   1231a: 46 f8        	sd	a7, 48(sp)
   1231c: c2 e0        	sd	a6, 64(sp)
   1231e: 3e e0        	sd	a5, 0(sp)
   12320: 3e ec        	sd	a5, 24(sp)
   12322: 3a c6        	sw	a4, 12(sp)
   12324: 3a d0        	sw	a4, 32(sp)
   12326: 02 d4        	sw	zero, 40(sp)
   12328: ef e0 0f c3  	jal	0x10758 <_vfprintf_r>
   1232c: aa 84        	mv	s1, a0
   1232e: 63 59 05 02  	bgez	a0, 0x12360 <__sbprintf+0x86>
   12332: 83 57 01 01  	lhu	a5, 16(sp)
   12336: 93 f7 07 04  	andi	a5, a5, 64
   1233a: 99 c7        	beqz	a5, 0x12348 <__sbprintf+0x6e>
   1233c: 83 57 04 01  	lhu	a5, 16(s0)
   12340: 93 e7 07 04  	ori	a5, a5, 64
   12344: 23 18 f4 00  	sh	a5, 16(s0)
   12348: 83 30 81 4c  	ld	ra, 1224(sp)
   1234c: 03 34 01 4c  	ld	s0, 1216(sp)
   12350: 03 39 01 4b  	ld	s2, 1200(sp)
   12354: 26 85        	mv	a0, s1
   12356: 83 34 81 4b  	ld	s1, 1208(sp)
   1235a: 13 01 01 4d  	addi	sp, sp, 1232
   1235e: 82 80        	ret
   12360: 8a 85        	mv	a1, sp
   12362: 4a 85        	mv	a0, s2
   12364: ef 00 a0 32  	jal	0x1268e <_fflush_r>
   12368: 69 d5        	beqz	a0, 0x12332 <__sbprintf+0x58>
   1236a: fd 54        	addi	s1, zero, -1
   1236c: d9 b7        	j	0x12332 <__sbprintf+0x58>

000000000001236e <__swsetup_r>:
   1236e: 83 b7 81 85  	ld	a5, -1960(gp)
   12372: 01 11        	addi	sp, sp, -32
   12374: 22 e8        	sd	s0, 16(sp)
   12376: 26 e4        	sd	s1, 8(sp)
   12378: 06 ec        	sd	ra, 24(sp)
   1237a: aa 84        	mv	s1, a0
   1237c: 2e 84        	mv	s0, a1
   1237e: 99 c3        	beqz	a5, 0x12384 <__swsetup_r+0x16>
   12380: b8 4b        	lw	a4, 80(a5)
   12382: 29 cb        	beqz	a4, 0x123d4 <__swsetup_r+0x66>
   12384: 83 17 04 01  	lh	a5, 16(s0)
   12388: 13 f7 87 00  	andi	a4, a5, 8
   1238c: 21 cf        	beqz	a4, 0x123e4 <__swsetup_r+0x76>
   1238e: 18 6c        	ld	a4, 24(s0)
   12390: 35 c7        	beqz	a4, 0x123fc <__swsetup_r+0x8e>
   12392: 93 f6 17 00  	andi	a3, a5, 1
   12396: 91 ce        	beqz	a3, 0x123b2 <__swsetup_r+0x44>
   12398: 14 50        	lw	a3, 32(s0)
   1239a: 23 26 04 00  	sw	zero, 12(s0)
   1239e: 01 45        	mv	a0, zero
   123a0: bb 06 d0 40  	negw	a3, a3
   123a4: 14 d4        	sw	a3, 40(s0)
   123a6: 11 cf        	beqz	a4, 0x123c2 <__swsetup_r+0x54>
   123a8: e2 60        	ld	ra, 24(sp)
   123aa: 42 64        	ld	s0, 16(sp)
   123ac: a2 64        	ld	s1, 8(sp)
   123ae: 05 61        	addi	sp, sp, 32
   123b0: 82 80        	ret
   123b2: 93 f6 27 00  	andi	a3, a5, 2
   123b6: 01 46        	mv	a2, zero
   123b8: 91 e2        	bnez	a3, 0x123bc <__swsetup_r+0x4e>
   123ba: 10 50        	lw	a2, 32(s0)
   123bc: 50 c4        	sw	a2, 12(s0)
   123be: 01 45        	mv	a0, zero
   123c0: 65 f7        	bnez	a4, 0x123a8 <__swsetup_r+0x3a>
   123c2: 13 f7 07 08  	andi	a4, a5, 128
   123c6: 6d d3        	beqz	a4, 0x123a8 <__swsetup_r+0x3a>
   123c8: 93 e7 07 04  	ori	a5, a5, 64
   123cc: 23 18 f4 00  	sh	a5, 16(s0)
   123d0: 7d 55        	addi	a0, zero, -1
   123d2: d9 bf        	j	0x123a8 <__swsetup_r+0x3a>
   123d4: 3e 85        	mv	a0, a5
   123d6: ef 00 a0 58  	jal	0x12960 <__sinit>
   123da: 83 17 04 01  	lh	a5, 16(s0)
   123de: 13 f7 87 00  	andi	a4, a5, 8
   123e2: 55 f7        	bnez	a4, 0x1238e <__swsetup_r+0x20>
   123e4: 13 f7 07 01  	andi	a4, a5, 16
   123e8: 21 cf        	beqz	a4, 0x12440 <__swsetup_r+0xd2>
   123ea: 13 f7 47 00  	andi	a4, a5, 4
   123ee: 0d e7        	bnez	a4, 0x12418 <__swsetup_r+0xaa>
   123f0: 18 6c        	ld	a4, 24(s0)
   123f2: 93 e7 87 00  	ori	a5, a5, 8
   123f6: 23 18 f4 00  	sh	a5, 16(s0)
   123fa: 41 ff        	bnez	a4, 0x12392 <__swsetup_r+0x24>
   123fc: 93 f6 07 28  	andi	a3, a5, 640
   12400: 13 06 00 20  	addi	a2, zero, 512
   12404: e3 87 c6 f8  	beq	a3, a2, 0x12392 <__swsetup_r+0x24>
   12408: a2 85        	mv	a1, s0
   1240a: 26 85        	mv	a0, s1
   1240c: ef 30 20 17  	jal	0x1557e <__smakebuf_r>
   12410: 83 17 04 01  	lh	a5, 16(s0)
   12414: 18 6c        	ld	a4, 24(s0)
   12416: b5 bf        	j	0x12392 <__swsetup_r+0x24>
   12418: 2c 6c        	ld	a1, 88(s0)
   1241a: 81 cd        	beqz	a1, 0x12432 <__swsetup_r+0xc4>
   1241c: 13 07 44 07  	addi	a4, s0, 116
   12420: 63 87 e5 00  	beq	a1, a4, 0x1242e <__swsetup_r+0xc0>
   12424: 26 85        	mv	a0, s1
   12426: ef 00 e0 66  	jal	0x12a94 <_free_r>
   1242a: 83 17 04 01  	lh	a5, 16(s0)
   1242e: 23 3c 04 04  	sd	zero, 88(s0)
   12432: 18 6c        	ld	a4, 24(s0)
   12434: 93 f7 b7 fd  	andi	a5, a5, -37
   12438: 23 24 04 00  	sw	zero, 8(s0)
   1243c: 18 e0        	sd	a4, 0(s0)
   1243e: 55 bf        	j	0x123f2 <__swsetup_r+0x84>
   12440: 25 47        	addi	a4, zero, 9
   12442: 98 c0        	sw	a4, 0(s1)
   12444: 93 e7 07 04  	ori	a5, a5, 64
   12448: 51 b7        	j	0x123cc <__swsetup_r+0x5e>

000000000001244a <__call_exitprocs>:
   1244a: 5d 71        	addi	sp, sp, -80
   1244c: 52 f0        	sd	s4, 32(sp)
   1244e: 03 ba 81 83  	ld	s4, -1992(gp)
   12452: 4a f8        	sd	s2, 48(sp)
   12454: 86 e4        	sd	ra, 72(sp)
   12456: 03 39 8a 1f  	ld	s2, 504(s4)
   1245a: a2 e0        	sd	s0, 64(sp)
   1245c: 26 fc        	sd	s1, 56(sp)
   1245e: 4e f4        	sd	s3, 40(sp)
   12460: 56 ec        	sd	s5, 24(sp)
   12462: 5a e8        	sd	s6, 16(sp)
   12464: 5e e4        	sd	s7, 8(sp)
   12466: 62 e0        	sd	s8, 0(sp)
   12468: 63 08 09 02  	beqz	s2, 0x12498 <__call_exitprocs+0x4e>
   1246c: 2a 8b        	mv	s6, a0
   1246e: ae 8b        	mv	s7, a1
   12470: 85 4a        	addi	s5, zero, 1
   12472: fd 59        	addi	s3, zero, -1
   12474: 83 24 89 00  	lw	s1, 8(s2)
   12478: 1b 84 f4 ff  	addiw	s0, s1, -1
   1247c: 63 4e 04 00  	bltz	s0, 0x12498 <__call_exitprocs+0x4e>
   12480: 8e 04        	slli	s1, s1, 3
   12482: ca 94        	add	s1, s1, s2
   12484: 63 86 0b 02  	beqz	s7, 0x124b0 <__call_exitprocs+0x66>
   12488: 83 b7 84 20  	ld	a5, 520(s1)
   1248c: 63 82 77 03  	beq	a5, s7, 0x124b0 <__call_exitprocs+0x66>
   12490: 7d 34        	addiw	s0, s0, -1
   12492: e1 14        	addi	s1, s1, -8
   12494: e3 18 34 ff  	bne	s0, s3, 0x12484 <__call_exitprocs+0x3a>
   12498: a6 60        	ld	ra, 72(sp)
   1249a: 06 64        	ld	s0, 64(sp)
   1249c: e2 74        	ld	s1, 56(sp)
   1249e: 42 79        	ld	s2, 48(sp)
   124a0: a2 79        	ld	s3, 40(sp)
   124a2: 02 7a        	ld	s4, 32(sp)
   124a4: e2 6a        	ld	s5, 24(sp)
   124a6: 42 6b        	ld	s6, 16(sp)
   124a8: a2 6b        	ld	s7, 8(sp)
   124aa: 02 6c        	ld	s8, 0(sp)
   124ac: 61 61        	addi	sp, sp, 80
   124ae: 82 80        	ret
   124b0: 83 27 89 00  	lw	a5, 8(s2)
   124b4: 98 64        	ld	a4, 8(s1)
   124b6: fd 37        	addiw	a5, a5, -1
   124b8: 63 84 87 04  	beq	a5, s0, 0x12500 <__call_exitprocs+0xb6>
   124bc: 23 b4 04 00  	sd	zero, 8(s1)
   124c0: 61 db        	beqz	a4, 0x12490 <__call_exitprocs+0x46>
   124c2: 83 27 09 31  	lw	a5, 784(s2)
   124c6: bb 96 8a 00  	sllw	a3, s5, s0
   124ca: 03 2c 89 00  	lw	s8, 8(s2)
   124ce: f5 8f        	and	a5, a5, a3
   124d0: 81 27        	sext.w	a5, a5
   124d2: 89 ef        	bnez	a5, 0x124ec <__call_exitprocs+0xa2>
   124d4: 02 97        	jalr	a4
   124d6: 03 27 89 00  	lw	a4, 8(s2)
   124da: 83 37 8a 1f  	ld	a5, 504(s4)
   124de: 63 14 87 01  	bne	a4, s8, 0x124e6 <__call_exitprocs+0x9c>
   124e2: e3 87 27 fb  	beq	a5, s2, 0x12490 <__call_exitprocs+0x46>
   124e6: cd db        	beqz	a5, 0x12498 <__call_exitprocs+0x4e>
   124e8: 3e 89        	mv	s2, a5
   124ea: 69 b7        	j	0x12474 <__call_exitprocs+0x2a>
   124ec: 83 27 49 31  	lw	a5, 788(s2)
   124f0: 83 b5 84 10  	ld	a1, 264(s1)
   124f4: f5 8f        	and	a5, a5, a3
   124f6: 81 27        	sext.w	a5, a5
   124f8: 99 e7        	bnez	a5, 0x12506 <__call_exitprocs+0xbc>
   124fa: 5a 85        	mv	a0, s6
   124fc: 02 97        	jalr	a4
   124fe: e1 bf        	j	0x124d6 <__call_exitprocs+0x8c>
   12500: 23 24 89 00  	sw	s0, 8(s2)
   12504: 75 bf        	j	0x124c0 <__call_exitprocs+0x76>
   12506: 2e 85        	mv	a0, a1
   12508: 02 97        	jalr	a4
   1250a: f1 b7        	j	0x124d6 <__call_exitprocs+0x8c>

000000000001250c <__sflush_r>:
   1250c: 83 97 05 01  	lh	a5, 16(a1)
   12510: 79 71        	addi	sp, sp, -48
   12512: 22 f0        	sd	s0, 32(sp)
   12514: 4e e4        	sd	s3, 8(sp)
   12516: 06 f4        	sd	ra, 40(sp)
   12518: 26 ec        	sd	s1, 24(sp)
   1251a: 4a e8        	sd	s2, 16(sp)
   1251c: 13 f7 87 00  	andi	a4, a5, 8
   12520: 2e 84        	mv	s0, a1
   12522: aa 89        	mv	s3, a0
   12524: 69 e3        	bnez	a4, 0x125e6 <__sflush_r+0xda>
   12526: 05 67        	lui	a4, 1
   12528: 13 07 07 80  	addi	a4, a4, -2048
   1252c: 94 45        	lw	a3, 8(a1)
   1252e: 5d 8f        	or	a4, a4, a5
   12530: 23 98 e5 00  	sh	a4, 16(a1)
   12534: 63 54 d0 10  	blez	a3, 0x1263c <__sflush_r+0x130>
   12538: 03 38 84 04  	ld	a6, 72(s0)
   1253c: 63 0d 08 08  	beqz	a6, 0x125d6 <__sflush_r+0xca>
   12540: b1 83        	srli	a5, a5, 12
   12542: 83 a4 09 00  	lw	s1, 0(s3)
   12546: 85 8b        	andi	a5, a5, 1
   12548: 23 a0 09 00  	sw	zero, 0(s3)
   1254c: 0c 78        	ld	a1, 48(s0)
   1254e: e5 ef        	bnez	a5, 0x12646 <__sflush_r+0x13a>
   12550: 01 46        	mv	a2, zero
   12552: 85 46        	addi	a3, zero, 1
   12554: 4e 85        	mv	a0, s3
   12556: 02 98        	jalr	a6
   12558: fd 57        	addi	a5, zero, -1
   1255a: 2a 86        	mv	a2, a0
   1255c: 63 08 f5 10  	beq	a0, a5, 0x1266c <__sflush_r+0x160>
   12560: 03 17 04 01  	lh	a4, 16(s0)
   12564: 03 38 84 04  	ld	a6, 72(s0)
   12568: 0c 78        	ld	a1, 48(s0)
   1256a: 11 8b        	andi	a4, a4, 4
   1256c: 19 c7        	beqz	a4, 0x1257a <__sflush_r+0x6e>
   1256e: 18 44        	lw	a4, 8(s0)
   12570: 3c 6c        	ld	a5, 88(s0)
   12572: 19 8e        	sub	a2, a2, a4
   12574: 99 c3        	beqz	a5, 0x1257a <__sflush_r+0x6e>
   12576: 3c 58        	lw	a5, 112(s0)
   12578: 1d 8e        	sub	a2, a2, a5
   1257a: 81 46        	mv	a3, zero
   1257c: 4e 85        	mv	a0, s3
   1257e: 02 98        	jalr	a6
   12580: fd 57        	addi	a5, zero, -1
   12582: 63 14 f5 0c  	bne	a0, a5, 0x1264a <__sflush_r+0x13e>
   12586: 83 a6 09 00  	lw	a3, 0(s3)
   1258a: f5 47        	addi	a5, zero, 29
   1258c: 63 ea d7 08  	bltu	a5, a3, 0x12620 <__sflush_r+0x114>
   12590: 37 07 c0 df  	lui	a4, 916480
   12594: 79 17        	addi	a4, a4, -2
   12596: 33 57 d7 40  	sra	a4, a4, a3
   1259a: 05 8b        	andi	a4, a4, 1
   1259c: 83 17 04 01  	lh	a5, 16(s0)
   125a0: 51 e3        	bnez	a4, 0x12624 <__sflush_r+0x118>
   125a2: 10 6c        	ld	a2, 24(s0)
   125a4: 7d 77        	lui	a4, 1048575
   125a6: 13 07 f7 7f  	addi	a4, a4, 2047
   125aa: 7d 8f        	and	a4, a4, a5
   125ac: b1 83        	srli	a5, a5, 12
   125ae: 23 18 e4 00  	sh	a4, 16(s0)
   125b2: 23 24 04 00  	sw	zero, 8(s0)
   125b6: 10 e0        	sd	a2, 0(s0)
   125b8: 85 8b        	andi	a5, a5, 1
   125ba: f9 e7        	bnez	a5, 0x12688 <__sflush_r+0x17c>
   125bc: 2c 6c        	ld	a1, 88(s0)
   125be: 23 a0 99 00  	sw	s1, 0(s3)
   125c2: 91 c9        	beqz	a1, 0x125d6 <__sflush_r+0xca>
   125c4: 93 07 44 07  	addi	a5, s0, 116
   125c8: 63 85 f5 00  	beq	a1, a5, 0x125d2 <__sflush_r+0xc6>
   125cc: 4e 85        	mv	a0, s3
   125ce: ef 00 60 4c  	jal	0x12a94 <_free_r>
   125d2: 23 3c 04 04  	sd	zero, 88(s0)
   125d6: 01 45        	mv	a0, zero
   125d8: a2 70        	ld	ra, 40(sp)
   125da: 02 74        	ld	s0, 32(sp)
   125dc: e2 64        	ld	s1, 24(sp)
   125de: 42 69        	ld	s2, 16(sp)
   125e0: a2 69        	ld	s3, 8(sp)
   125e2: 45 61        	addi	sp, sp, 48
   125e4: 82 80        	ret
   125e6: 03 b9 85 01  	ld	s2, 24(a1)
   125ea: e3 06 09 fe  	beqz	s2, 0x125d6 <__sflush_r+0xca>
   125ee: 84 61        	ld	s1, 0(a1)
   125f0: 8d 8b        	andi	a5, a5, 3
   125f2: 23 b0 25 01  	sd	s2, 0(a1)
   125f6: bb 84 24 41  	subw	s1, s1, s2
   125fa: 01 47        	mv	a4, zero
   125fc: 91 e3        	bnez	a5, 0x12600 <__sflush_r+0xf4>
   125fe: 98 51        	lw	a4, 32(a1)
   12600: 58 c4        	sw	a4, 12(s0)
   12602: 63 46 90 00  	bgtz	s1, 0x1260e <__sflush_r+0x102>
   12606: c1 bf        	j	0x125d6 <__sflush_r+0xca>
   12608: 2a 99        	add	s2, s2, a0
   1260a: e3 56 90 fc  	blez	s1, 0x125d6 <__sflush_r+0xca>
   1260e: 3c 60        	ld	a5, 64(s0)
   12610: 0c 78        	ld	a1, 48(s0)
   12612: a6 86        	mv	a3, s1
   12614: 4a 86        	mv	a2, s2
   12616: 4e 85        	mv	a0, s3
   12618: 82 97        	jalr	a5
   1261a: 89 9c        	subw	s1, s1, a0
   1261c: e3 46 a0 fe  	bgtz	a0, 0x12608 <__sflush_r+0xfc>
   12620: 83 17 04 01  	lh	a5, 16(s0)
   12624: 93 e7 07 04  	ori	a5, a5, 64
   12628: a2 70        	ld	ra, 40(sp)
   1262a: 23 18 f4 00  	sh	a5, 16(s0)
   1262e: 02 74        	ld	s0, 32(sp)
   12630: e2 64        	ld	s1, 24(sp)
   12632: 42 69        	ld	s2, 16(sp)
   12634: a2 69        	ld	s3, 8(sp)
   12636: 7d 55        	addi	a0, zero, -1
   12638: 45 61        	addi	sp, sp, 48
   1263a: 82 80        	ret
   1263c: b4 59        	lw	a3, 112(a1)
   1263e: e3 4d d0 ee  	bgtz	a3, 0x12538 <__sflush_r+0x2c>
   12642: 01 45        	mv	a0, zero
   12644: 51 bf        	j	0x125d8 <__sflush_r+0xcc>
   12646: 50 68        	ld	a2, 144(s0)
   12648: 0d b7        	j	0x1256a <__sflush_r+0x5e>
   1264a: 83 17 04 01  	lh	a5, 16(s0)
   1264e: 14 6c        	ld	a3, 24(s0)
   12650: 7d 77        	lui	a4, 1048575
   12652: 13 07 f7 7f  	addi	a4, a4, 2047
   12656: 7d 8f        	and	a4, a4, a5
   12658: b1 83        	srli	a5, a5, 12
   1265a: 23 18 e4 00  	sh	a4, 16(s0)
   1265e: 23 24 04 00  	sw	zero, 8(s0)
   12662: 14 e0        	sd	a3, 0(s0)
   12664: 85 8b        	andi	a5, a5, 1
   12666: b9 db        	beqz	a5, 0x125bc <__sflush_r+0xb0>
   12668: 48 e8        	sd	a0, 144(s0)
   1266a: 89 bf        	j	0x125bc <__sflush_r+0xb0>
   1266c: 83 a7 09 00  	lw	a5, 0(s3)
   12670: e3 88 07 ee  	beqz	a5, 0x12560 <__sflush_r+0x54>
   12674: 75 47        	addi	a4, zero, 29
   12676: 63 85 e7 00  	beq	a5, a4, 0x12680 <__sflush_r+0x174>
   1267a: 59 47        	addi	a4, zero, 22
   1267c: e3 92 e7 fa  	bne	a5, a4, 0x12620 <__sflush_r+0x114>
   12680: 23 a0 99 00  	sw	s1, 0(s3)
   12684: 01 45        	mv	a0, zero
   12686: 89 bf        	j	0x125d8 <__sflush_r+0xcc>
   12688: 95 fa        	bnez	a3, 0x125bc <__sflush_r+0xb0>
   1268a: 48 e8        	sd	a0, 144(s0)
   1268c: 05 bf        	j	0x125bc <__sflush_r+0xb0>

000000000001268e <_fflush_r>:
   1268e: 01 11        	addi	sp, sp, -32
   12690: 22 e8        	sd	s0, 16(sp)
   12692: 06 ec        	sd	ra, 24(sp)
   12694: 2a 84        	mv	s0, a0
   12696: 19 c1        	beqz	a0, 0x1269c <_fflush_r+0xe>
   12698: 3c 49        	lw	a5, 80(a0)
   1269a: 89 cb        	beqz	a5, 0x126ac <_fflush_r+0x1e>
   1269c: 83 97 05 01  	lh	a5, 16(a1)
   126a0: 89 ef        	bnez	a5, 0x126ba <_fflush_r+0x2c>
   126a2: e2 60        	ld	ra, 24(sp)
   126a4: 42 64        	ld	s0, 16(sp)
   126a6: 01 45        	mv	a0, zero
   126a8: 05 61        	addi	sp, sp, 32
   126aa: 82 80        	ret
   126ac: 2e e4        	sd	a1, 8(sp)
   126ae: ef 00 20 2b  	jal	0x12960 <__sinit>
   126b2: a2 65        	ld	a1, 8(sp)
   126b4: 83 97 05 01  	lh	a5, 16(a1)
   126b8: ed d7        	beqz	a5, 0x126a2 <_fflush_r+0x14>
   126ba: 22 85        	mv	a0, s0
   126bc: 42 64        	ld	s0, 16(sp)
   126be: e2 60        	ld	ra, 24(sp)
   126c0: 05 61        	addi	sp, sp, 32
   126c2: a9 b5        	j	0x1250c <__sflush_r>

00000000000126c4 <fflush>:
   126c4: 1d cd        	beqz	a0, 0x12702 <fflush+0x3e>
   126c6: 01 11        	addi	sp, sp, -32
   126c8: 22 e8        	sd	s0, 16(sp)
   126ca: 2a 84        	mv	s0, a0
   126cc: 03 b5 81 85  	ld	a0, -1960(gp)
   126d0: 06 ec        	sd	ra, 24(sp)
   126d2: 19 c1        	beqz	a0, 0x126d8 <fflush+0x14>
   126d4: 3c 49        	lw	a5, 80(a0)
   126d6: 91 cf        	beqz	a5, 0x126f2 <fflush+0x2e>
   126d8: 83 17 04 01  	lh	a5, 16(s0)
   126dc: 91 e7        	bnez	a5, 0x126e8 <fflush+0x24>
   126de: e2 60        	ld	ra, 24(sp)
   126e0: 42 64        	ld	s0, 16(sp)
   126e2: 01 45        	mv	a0, zero
   126e4: 05 61        	addi	sp, sp, 32
   126e6: 82 80        	ret
   126e8: a2 85        	mv	a1, s0
   126ea: 42 64        	ld	s0, 16(sp)
   126ec: e2 60        	ld	ra, 24(sp)
   126ee: 05 61        	addi	sp, sp, 32
   126f0: 31 bd        	j	0x1250c <__sflush_r>
   126f2: 2a e4        	sd	a0, 8(sp)
   126f4: ef 00 c0 26  	jal	0x12960 <__sinit>
   126f8: 83 17 04 01  	lh	a5, 16(s0)
   126fc: 22 65        	ld	a0, 8(sp)
   126fe: e5 d3        	beqz	a5, 0x126de <fflush+0x1a>
   12700: e5 b7        	j	0x126e8 <fflush+0x24>
   12702: 03 b5 81 83  	ld	a0, -1992(gp)
   12706: c9 65        	lui	a1, 18
   12708: 93 85 e5 68  	addi	a1, a1, 1678
   1270c: 89 ad        	j	0x12d5e <_fwalk_reent>

000000000001270e <__fp_lock>:
   1270e: 01 45        	mv	a0, zero
   12710: 82 80        	ret

0000000000012712 <_cleanup_r>:
   12712: e9 65        	lui	a1, 26
   12714: 93 85 25 aa  	addi	a1, a1, -1374
   12718: 99 a5        	j	0x12d5e <_fwalk_reent>

000000000001271a <__fp_unlock>:
   1271a: 01 45        	mv	a0, zero
   1271c: 82 80        	ret

000000000001271e <__sinit.part.0>:
   1271e: 39 71        	addi	sp, sp, -64
   12720: c9 67        	lui	a5, 18
   12722: 06 fc        	sd	ra, 56(sp)
   12724: 22 f8        	sd	s0, 48(sp)
   12726: 26 f4        	sd	s1, 40(sp)
   12728: 4a f0        	sd	s2, 32(sp)
   1272a: 4e ec        	sd	s3, 24(sp)
   1272c: 52 e8        	sd	s4, 16(sp)
   1272e: 56 e4        	sd	s5, 8(sp)
   12730: 5a e0        	sd	s6, 0(sp)
   12732: 00 65        	ld	s0, 8(a0)
   12734: 93 87 27 71  	addi	a5, a5, 1810
   12738: 3c ed        	sd	a5, 88(a0)
   1273a: 13 07 85 53  	addi	a4, a0, 1336
   1273e: 8d 47        	addi	a5, zero, 3
   12740: 23 38 e5 52  	sd	a4, 1328(a0)
   12744: 23 24 f5 52  	sw	a5, 1320(a0)
   12748: 23 30 05 52  	sd	zero, 1312(a0)
   1274c: 91 47        	addi	a5, zero, 4
   1274e: 2a 89        	mv	s2, a0
   12750: 1c c8        	sw	a5, 16(s0)
   12752: 21 46        	addi	a2, zero, 8
   12754: 81 45        	mv	a1, zero
   12756: 23 26 04 0a  	sw	zero, 172(s0)
   1275a: 23 30 04 00  	sd	zero, 0(s0)
   1275e: 23 34 04 00  	sd	zero, 8(s0)
   12762: 23 3c 04 00  	sd	zero, 24(s0)
   12766: 23 20 04 02  	sw	zero, 32(s0)
   1276a: 23 24 04 02  	sw	zero, 40(s0)
   1276e: 13 05 44 0a  	addi	a0, s0, 164
   12772: ef d0 ff ee  	jal	0x10660 <memset>
   12776: 5d 6b        	lui	s6, 23
   12778: 83 34 09 01  	ld	s1, 16(s2)
   1277c: dd 6a        	lui	s5, 23
   1277e: 5d 6a        	lui	s4, 23
   12780: dd 69        	lui	s3, 23
   12782: 13 0b 0b c1  	addi	s6, s6, -1008
   12786: 93 8a ca c4  	addi	s5, s5, -948
   1278a: 13 0a 0a ca  	addi	s4, s4, -864
   1278e: 93 89 29 cf  	addi	s3, s3, -782
   12792: c1 67        	lui	a5, 16
   12794: 23 3c 64 03  	sd	s6, 56(s0)
   12798: 23 30 54 05  	sd	s5, 64(s0)
   1279c: 23 34 44 05  	sd	s4, 72(s0)
   127a0: 23 38 34 05  	sd	s3, 80(s0)
   127a4: 00 f8        	sd	s0, 48(s0)
   127a6: a5 07        	addi	a5, a5, 9
   127a8: 9c c8        	sw	a5, 16(s1)
   127aa: 21 46        	addi	a2, zero, 8
   127ac: 81 45        	mv	a1, zero
   127ae: 23 a6 04 0a  	sw	zero, 172(s1)
   127b2: 23 b0 04 00  	sd	zero, 0(s1)
   127b6: 23 b4 04 00  	sd	zero, 8(s1)
   127ba: 23 bc 04 00  	sd	zero, 24(s1)
   127be: 23 a0 04 02  	sw	zero, 32(s1)
   127c2: 23 a4 04 02  	sw	zero, 40(s1)
   127c6: 13 85 44 0a  	addi	a0, s1, 164
   127ca: ef d0 7f e9  	jal	0x10660 <memset>
   127ce: 03 34 89 01  	ld	s0, 24(s2)
   127d2: b7 07 02 00  	lui	a5, 32
   127d6: 23 bc 64 03  	sd	s6, 56(s1)
   127da: 23 b0 54 05  	sd	s5, 64(s1)
   127de: 23 b4 44 05  	sd	s4, 72(s1)
   127e2: 23 b8 34 05  	sd	s3, 80(s1)
   127e6: 84 f8        	sd	s1, 48(s1)
   127e8: c9 07        	addi	a5, a5, 18
   127ea: 1c c8        	sw	a5, 16(s0)
   127ec: 23 26 04 0a  	sw	zero, 172(s0)
   127f0: 23 30 04 00  	sd	zero, 0(s0)
   127f4: 23 34 04 00  	sd	zero, 8(s0)
   127f8: 23 3c 04 00  	sd	zero, 24(s0)
   127fc: 23 20 04 02  	sw	zero, 32(s0)
   12800: 23 24 04 02  	sw	zero, 40(s0)
   12804: 13 05 44 0a  	addi	a0, s0, 164
   12808: 21 46        	addi	a2, zero, 8
   1280a: 81 45        	mv	a1, zero
   1280c: ef d0 5f e5  	jal	0x10660 <memset>
   12810: e2 70        	ld	ra, 56(sp)
   12812: 23 3c 64 03  	sd	s6, 56(s0)
   12816: 23 30 54 05  	sd	s5, 64(s0)
   1281a: 23 34 44 05  	sd	s4, 72(s0)
   1281e: 23 38 34 05  	sd	s3, 80(s0)
   12822: 00 f8        	sd	s0, 48(s0)
   12824: 42 74        	ld	s0, 48(sp)
   12826: 85 47        	addi	a5, zero, 1
   12828: 23 28 f9 04  	sw	a5, 80(s2)
   1282c: a2 74        	ld	s1, 40(sp)
   1282e: 02 79        	ld	s2, 32(sp)
   12830: e2 69        	ld	s3, 24(sp)
   12832: 42 6a        	ld	s4, 16(sp)
   12834: a2 6a        	ld	s5, 8(sp)
   12836: 02 6b        	ld	s6, 0(sp)
   12838: 21 61        	addi	sp, sp, 64
   1283a: 82 80        	ret

000000000001283c <__sfmoreglue>:
   1283c: 01 11        	addi	sp, sp, -32
   1283e: 26 e4        	sd	s1, 8(sp)
   12840: 93 07 00 0b  	addi	a5, zero, 176
   12844: 9b 84 f5 ff  	addiw	s1, a1, -1
   12848: b3 84 f4 02  	mul	s1, s1, a5
   1284c: 4a e0        	sd	s2, 0(sp)
   1284e: 2e 89        	mv	s2, a1
   12850: 22 e8        	sd	s0, 16(sp)
   12852: 06 ec        	sd	ra, 24(sp)
   12854: 93 85 84 0c  	addi	a1, s1, 200
   12858: ef 20 30 6c  	jal	0x1571a <_malloc_r>
   1285c: 2a 84        	mv	s0, a0
   1285e: 01 cd        	beqz	a0, 0x12876 <__sfmoreglue+0x3a>
   12860: 61 05        	addi	a0, a0, 24
   12862: 23 30 04 00  	sd	zero, 0(s0)
   12866: 23 24 24 01  	sw	s2, 8(s0)
   1286a: 08 e8        	sd	a0, 16(s0)
   1286c: 13 86 04 0b  	addi	a2, s1, 176
   12870: 81 45        	mv	a1, zero
   12872: ef d0 ff de  	jal	0x10660 <memset>
   12876: e2 60        	ld	ra, 24(sp)
   12878: 22 85        	mv	a0, s0
   1287a: 42 64        	ld	s0, 16(sp)
   1287c: a2 64        	ld	s1, 8(sp)
   1287e: 02 69        	ld	s2, 0(sp)
   12880: 05 61        	addi	sp, sp, 32
   12882: 82 80        	ret

0000000000012884 <__sfp>:
   12884: 79 71        	addi	sp, sp, -48
   12886: 26 ec        	sd	s1, 24(sp)
   12888: 83 b4 81 83  	ld	s1, -1992(gp)
   1288c: 4a e8        	sd	s2, 16(sp)
   1288e: 06 f4        	sd	ra, 40(sp)
   12890: bc 48        	lw	a5, 80(s1)
   12892: 22 f0        	sd	s0, 32(sp)
   12894: 4e e4        	sd	s3, 8(sp)
   12896: 2a 89        	mv	s2, a0
   12898: bd cf        	beqz	a5, 0x12916 <__sfp+0x92>
   1289a: 93 84 04 52  	addi	s1, s1, 1312
   1289e: 93 09 00 0b  	addi	s3, zero, 176
   128a2: 98 44        	lw	a4, 8(s1)
   128a4: 80 68        	ld	s0, 16(s1)
   128a6: 63 54 e0 06  	blez	a4, 0x1290e <__sfp+0x8a>
   128aa: 02 17        	slli	a4, a4, 32
   128ac: 01 93        	srli	a4, a4, 32
   128ae: 33 07 37 03  	mul	a4, a4, s3
   128b2: 22 97        	add	a4, a4, s0
   128b4: 29 a0        	j	0x128be <__sfp+0x3a>
   128b6: 13 04 04 0b  	addi	s0, s0, 176
   128ba: 63 0a e4 04  	beq	s0, a4, 0x1290e <__sfp+0x8a>
   128be: 83 17 04 01  	lh	a5, 16(s0)
   128c2: f5 fb        	bnez	a5, 0x128b6 <__sfp+0x32>
   128c4: c1 77        	lui	a5, 1048560
   128c6: 85 07        	addi	a5, a5, 1
   128c8: 23 26 04 0a  	sw	zero, 172(s0)
   128cc: 23 30 04 00  	sd	zero, 0(s0)
   128d0: 23 34 04 00  	sd	zero, 8(s0)
   128d4: 1c c8        	sw	a5, 16(s0)
   128d6: 23 3c 04 00  	sd	zero, 24(s0)
   128da: 23 20 04 02  	sw	zero, 32(s0)
   128de: 23 24 04 02  	sw	zero, 40(s0)
   128e2: 21 46        	addi	a2, zero, 8
   128e4: 81 45        	mv	a1, zero
   128e6: 13 05 44 0a  	addi	a0, s0, 164
   128ea: ef d0 7f d7  	jal	0x10660 <memset>
   128ee: 23 3c 04 04  	sd	zero, 88(s0)
   128f2: 23 20 04 06  	sw	zero, 96(s0)
   128f6: 23 3c 04 06  	sd	zero, 120(s0)
   128fa: 23 20 04 08  	sw	zero, 128(s0)
   128fe: a2 70        	ld	ra, 40(sp)
   12900: 22 85        	mv	a0, s0
   12902: 02 74        	ld	s0, 32(sp)
   12904: e2 64        	ld	s1, 24(sp)
   12906: 42 69        	ld	s2, 16(sp)
   12908: a2 69        	ld	s3, 8(sp)
   1290a: 45 61        	addi	sp, sp, 48
   1290c: 82 80        	ret
   1290e: 80 60        	ld	s0, 0(s1)
   12910: 19 c4        	beqz	s0, 0x1291e <__sfp+0x9a>
   12912: a2 84        	mv	s1, s0
   12914: 79 b7        	j	0x128a2 <__sfp+0x1e>
   12916: 26 85        	mv	a0, s1
   12918: ef f0 7f e0  	jal	0x1271e <__sinit.part.0>
   1291c: bd bf        	j	0x1289a <__sfp+0x16>
   1291e: 93 05 80 2d  	addi	a1, zero, 728
   12922: 4a 85        	mv	a0, s2
   12924: ef 20 70 5f  	jal	0x1571a <_malloc_r>
   12928: 2a 84        	mv	s0, a0
   1292a: 19 cd        	beqz	a0, 0x12948 <__sfp+0xc4>
   1292c: 61 05        	addi	a0, a0, 24
   1292e: 91 47        	addi	a5, zero, 4
   12930: 23 30 04 00  	sd	zero, 0(s0)
   12934: 1c c4        	sw	a5, 8(s0)
   12936: 08 e8        	sd	a0, 16(s0)
   12938: 13 06 00 2c  	addi	a2, zero, 704
   1293c: 81 45        	mv	a1, zero
   1293e: ef d0 3f d2  	jal	0x10660 <memset>
   12942: 80 e0        	sd	s0, 0(s1)
   12944: a2 84        	mv	s1, s0
   12946: b1 bf        	j	0x128a2 <__sfp+0x1e>
   12948: 23 b0 04 00  	sd	zero, 0(s1)
   1294c: b1 47        	addi	a5, zero, 12
   1294e: 23 20 f9 00  	sw	a5, 0(s2)
   12952: 75 b7        	j	0x128fe <__sfp+0x7a>

0000000000012954 <_cleanup>:
   12954: 03 b5 81 83  	ld	a0, -1992(gp)
   12958: e9 65        	lui	a1, 26
   1295a: 93 85 25 aa  	addi	a1, a1, -1374
   1295e: 01 a1        	j	0x12d5e <_fwalk_reent>

0000000000012960 <__sinit>:
   12960: 3c 49        	lw	a5, 80(a0)
   12962: 91 c3        	beqz	a5, 0x12966 <__sinit+0x6>
   12964: 82 80        	ret
   12966: 65 bb        	j	0x1271e <__sinit.part.0>

0000000000012968 <__sfp_lock_acquire>:
   12968: 82 80        	ret

000000000001296a <__sfp_lock_release>:
   1296a: 82 80        	ret

000000000001296c <__sinit_lock_acquire>:
   1296c: 82 80        	ret

000000000001296e <__sinit_lock_release>:
   1296e: 82 80        	ret

0000000000012970 <__fp_lock_all>:
   12970: 03 b5 81 85  	ld	a0, -1960(gp)
   12974: c9 65        	lui	a1, 18
   12976: 93 85 e5 70  	addi	a1, a1, 1806
   1297a: a5 a6        	j	0x12ce2 <_fwalk>

000000000001297c <__fp_unlock_all>:
   1297c: 03 b5 81 85  	ld	a0, -1960(gp)
   12980: c9 65        	lui	a1, 18
   12982: 93 85 a5 71  	addi	a1, a1, 1818
   12986: b1 ae        	j	0x12ce2 <_fwalk>

0000000000012988 <__libc_fini_array>:
   12988: 01 11        	addi	sp, sp, -32
   1298a: 22 e8        	sd	s0, 16(sp)
   1298c: b7 f7 01 00  	lui	a5, 31
   12990: 37 f4 01 00  	lui	s0, 31
   12994: 93 87 87 64  	addi	a5, a5, 1608
   12998: 13 04 04 65  	addi	s0, s0, 1616
   1299c: 1d 8c        	sub	s0, s0, a5
   1299e: 26 e4        	sd	s1, 8(sp)
   129a0: 06 ec        	sd	ra, 24(sp)
   129a2: 93 54 34 40  	srai	s1, s0, 3
   129a6: 81 c8        	beqz	s1, 0x129b6 <__libc_fini_array+0x2e>
   129a8: 61 14        	addi	s0, s0, -8
   129aa: 3e 94        	add	s0, s0, a5
   129ac: 1c 60        	ld	a5, 0(s0)
   129ae: fd 14        	addi	s1, s1, -1
   129b0: 61 14        	addi	s0, s0, -8
   129b2: 82 97        	jalr	a5
   129b4: e5 fc        	bnez	s1, 0x129ac <__libc_fini_array+0x24>
   129b6: e2 60        	ld	ra, 24(sp)
   129b8: 42 64        	ld	s0, 16(sp)
   129ba: a2 64        	ld	s1, 8(sp)
   129bc: 05 61        	addi	sp, sp, 32
   129be: 82 80        	ret

00000000000129c0 <_malloc_trim_r>:
   129c0: 79 71        	addi	sp, sp, -48
   129c2: 4e e4        	sd	s3, 8(sp)
   129c4: b7 09 02 00  	lui	s3, 32
   129c8: 22 f0        	sd	s0, 32(sp)
   129ca: 26 ec        	sd	s1, 24(sp)
   129cc: 4a e8        	sd	s2, 16(sp)
   129ce: 52 e0        	sd	s4, 0(sp)
   129d0: 06 f4        	sd	ra, 40(sp)
   129d2: 2e 8a        	mv	s4, a1
   129d4: 2a 89        	mv	s2, a0
   129d6: 93 89 09 f4  	addi	s3, s3, -192
   129da: ef 30 20 41  	jal	0x15dec <__malloc_lock>
   129de: 03 b7 09 01  	ld	a4, 16(s3)
   129e2: 85 67        	lui	a5, 1
   129e4: 13 84 f7 fd  	addi	s0, a5, -33
   129e8: 04 67        	ld	s1, 8(a4)
   129ea: f1 98        	andi	s1, s1, -4
   129ec: 26 94        	add	s0, s0, s1
   129ee: 33 04 44 41  	sub	s0, s0, s4
   129f2: 31 80        	srli	s0, s0, 12
   129f4: 7d 14        	addi	s0, s0, -1
   129f6: 32 04        	slli	s0, s0, 12
   129f8: 63 4b f4 00  	blt	s0, a5, 0x12a0e <_malloc_trim_r+0x4e>
   129fc: 81 45        	mv	a1, zero
   129fe: 4a 85        	mv	a0, s2
   12a00: ef 40 40 0d  	jal	0x16ad4 <_sbrk_r>
   12a04: 83 b7 09 01  	ld	a5, 16(s3)
   12a08: a6 97        	add	a5, a5, s1
   12a0a: 63 0e f5 00  	beq	a0, a5, 0x12a26 <_malloc_trim_r+0x66>
   12a0e: 4a 85        	mv	a0, s2
   12a10: ef 30 e0 3d  	jal	0x15dee <__malloc_unlock>
   12a14: a2 70        	ld	ra, 40(sp)
   12a16: 02 74        	ld	s0, 32(sp)
   12a18: e2 64        	ld	s1, 24(sp)
   12a1a: 42 69        	ld	s2, 16(sp)
   12a1c: a2 69        	ld	s3, 8(sp)
   12a1e: 02 6a        	ld	s4, 0(sp)
   12a20: 01 45        	mv	a0, zero
   12a22: 45 61        	addi	sp, sp, 48
   12a24: 82 80        	ret
   12a26: b3 05 80 40  	neg	a1, s0
   12a2a: 4a 85        	mv	a0, s2
   12a2c: ef 40 80 0a  	jal	0x16ad4 <_sbrk_r>
   12a30: fd 57        	addi	a5, zero, -1
   12a32: 63 0b f5 02  	beq	a0, a5, 0x12a68 <_malloc_trim_r+0xa8>
   12a36: b7 37 02 00  	lui	a5, 35
   12a3a: 93 87 07 a9  	addi	a5, a5, -1392
   12a3e: 98 43        	lw	a4, 0(a5)
   12a40: 83 b6 09 01  	ld	a3, 16(s3)
   12a44: 81 8c        	sub	s1, s1, s0
   12a46: 93 e4 14 00  	ori	s1, s1, 1
   12a4a: 01 9f        	subw	a4, a4, s0
   12a4c: 4a 85        	mv	a0, s2
   12a4e: 84 e6        	sd	s1, 8(a3)
   12a50: 98 c3        	sw	a4, 0(a5)
   12a52: ef 30 c0 39  	jal	0x15dee <__malloc_unlock>
   12a56: a2 70        	ld	ra, 40(sp)
   12a58: 02 74        	ld	s0, 32(sp)
   12a5a: e2 64        	ld	s1, 24(sp)
   12a5c: 42 69        	ld	s2, 16(sp)
   12a5e: a2 69        	ld	s3, 8(sp)
   12a60: 02 6a        	ld	s4, 0(sp)
   12a62: 05 45        	addi	a0, zero, 1
   12a64: 45 61        	addi	sp, sp, 48
   12a66: 82 80        	ret
   12a68: 81 45        	mv	a1, zero
   12a6a: 4a 85        	mv	a0, s2
   12a6c: ef 40 80 06  	jal	0x16ad4 <_sbrk_r>
   12a70: 03 b7 09 01  	ld	a4, 16(s3)
   12a74: fd 46        	addi	a3, zero, 31
   12a76: b3 07 e5 40  	sub	a5, a0, a4
   12a7a: e3 da f6 f8  	bge	a3, a5, 0x12a0e <_malloc_trim_r+0x4e>
   12a7e: 83 b6 01 86  	ld	a3, -1952(gp)
   12a82: 93 e7 17 00  	ori	a5, a5, 1
   12a86: 1c e7        	sd	a5, 8(a4)
   12a88: 15 8d        	sub	a0, a0, a3
   12a8a: b7 36 02 00  	lui	a3, 35
   12a8e: 23 a8 a6 a8  	sw	a0, -1392(a3)
   12a92: b5 bf        	j	0x12a0e <_malloc_trim_r+0x4e>

0000000000012a94 <_free_r>:
   12a94: 63 85 05 12  	beqz	a1, 0x12bbe <_free_r+0x12a>
   12a98: 01 11        	addi	sp, sp, -32
   12a9a: 22 e8        	sd	s0, 16(sp)
   12a9c: 26 e4        	sd	s1, 8(sp)
   12a9e: 2e 84        	mv	s0, a1
   12aa0: aa 84        	mv	s1, a0
   12aa2: 06 ec        	sd	ra, 24(sp)
   12aa4: ef 30 80 34  	jal	0x15dec <__malloc_lock>
   12aa8: 03 35 84 ff  	ld	a0, -8(s0)
   12aac: 13 07 04 ff  	addi	a4, s0, -16
   12ab0: b7 05 02 00  	lui	a1, 32
   12ab4: 93 77 e5 ff  	andi	a5, a0, -2
   12ab8: 33 06 f7 00  	add	a2, a4, a5
   12abc: 93 85 05 f4  	addi	a1, a1, -192
   12ac0: 14 66        	ld	a3, 8(a2)
   12ac2: 03 b8 05 01  	ld	a6, 16(a1)
   12ac6: f1 9a        	andi	a3, a3, -4
   12ac8: 63 0e c8 12  	beq	a6, a2, 0x12c04 <_free_r+0x170>
   12acc: 14 e6        	sd	a3, 8(a2)
   12ace: 05 89        	andi	a0, a0, 1
   12ad0: 33 08 d6 00  	add	a6, a2, a3
   12ad4: 49 e1        	bnez	a0, 0x12b56 <_free_r+0xc2>
   12ad6: 03 33 04 ff  	ld	t1, -16(s0)
   12ada: 03 38 88 00  	ld	a6, 8(a6)
   12ade: 37 05 02 00  	lui	a0, 32
   12ae2: 33 07 67 40  	sub	a4, a4, t1
   12ae6: 83 38 07 01  	ld	a7, 16(a4)
   12aea: 13 05 05 f5  	addi	a0, a0, -176
   12aee: 9a 97        	add	a5, a5, t1
   12af0: 13 78 18 00  	andi	a6, a6, 1
   12af4: 63 8b a8 0e  	beq	a7, a0, 0x12bea <_free_r+0x156>
   12af8: 03 33 87 01  	ld	t1, 24(a4)
   12afc: 23 bc 68 00  	sd	t1, 24(a7)
   12b00: 23 38 13 01  	sd	a7, 16(t1)
   12b04: 63 09 08 12  	beqz	a6, 0x12c36 <_free_r+0x1a2>
   12b08: 93 e6 17 00  	ori	a3, a5, 1
   12b0c: 14 e7        	sd	a3, 8(a4)
   12b0e: 1c e2        	sd	a5, 0(a2)
   12b10: 93 06 f0 1f  	addi	a3, zero, 511
   12b14: 63 ee f6 04  	bltu	a3, a5, 0x12b70 <_free_r+0xdc>
   12b18: 8d 83        	srli	a5, a5, 3
   12b1a: 9b 86 17 00  	addiw	a3, a5, 1
   12b1e: 9b 96 16 00  	slliw	a3, a3, 1
   12b22: 8e 06        	slli	a3, a3, 3
   12b24: 03 b8 85 00  	ld	a6, 8(a1)
   12b28: ae 96        	add	a3, a3, a1
   12b2a: 88 62        	ld	a0, 0(a3)
   12b2c: 1b d6 27 40  	sraiw	a2, a5, 2
   12b30: 85 47        	addi	a5, zero, 1
   12b32: b3 97 c7 00  	sll	a5, a5, a2
   12b36: b3 e7 07 01  	or	a5, a5, a6
   12b3a: 13 86 06 ff  	addi	a2, a3, -16
   12b3e: 10 ef        	sd	a2, 24(a4)
   12b40: 08 eb        	sd	a0, 16(a4)
   12b42: 9c e5        	sd	a5, 8(a1)
   12b44: 98 e2        	sd	a4, 0(a3)
   12b46: 18 ed        	sd	a4, 24(a0)
   12b48: 42 64        	ld	s0, 16(sp)
   12b4a: e2 60        	ld	ra, 24(sp)
   12b4c: 26 85        	mv	a0, s1
   12b4e: a2 64        	ld	s1, 8(sp)
   12b50: 05 61        	addi	sp, sp, 32
   12b52: 6f 30 c0 29  	j	0x15dee <__malloc_unlock>
   12b56: 03 35 88 00  	ld	a0, 8(a6)
   12b5a: 05 89        	andi	a0, a0, 1
   12b5c: 35 c1        	beqz	a0, 0x12bc0 <_free_r+0x12c>
   12b5e: 93 e6 17 00  	ori	a3, a5, 1
   12b62: 23 3c d4 fe  	sd	a3, -8(s0)
   12b66: 1c e2        	sd	a5, 0(a2)
   12b68: 93 06 f0 1f  	addi	a3, zero, 511
   12b6c: e3 f6 f6 fa  	bgeu	a3, a5, 0x12b18 <_free_r+0x84>
   12b70: 93 d6 97 00  	srli	a3, a5, 9
   12b74: 11 46        	addi	a2, zero, 4
   12b76: 63 62 d6 0c  	bltu	a2, a3, 0x12c3a <_free_r+0x1a6>
   12b7a: 93 d6 67 00  	srli	a3, a5, 6
   12b7e: 1b 85 96 03  	addiw	a0, a3, 57
   12b82: 1b 15 15 00  	slliw	a0, a0, 1
   12b86: 1b 86 86 03  	addiw	a2, a3, 56
   12b8a: 0e 05        	slli	a0, a0, 3
   12b8c: 2e 95        	add	a0, a0, a1
   12b8e: 14 61        	ld	a3, 0(a0)
   12b90: 41 15        	addi	a0, a0, -16
   12b92: 63 16 d5 00  	bne	a0, a3, 0x12b9e <_free_r+0x10a>
   12b96: ed a0        	j	0x12c80 <_free_r+0x1ec>
   12b98: 94 6a        	ld	a3, 16(a3)
   12b9a: 63 06 d5 00  	beq	a0, a3, 0x12ba6 <_free_r+0x112>
   12b9e: 90 66        	ld	a2, 8(a3)
   12ba0: 71 9a        	andi	a2, a2, -4
   12ba2: e3 eb c7 fe  	bltu	a5, a2, 0x12b98 <_free_r+0x104>
   12ba6: 88 6e        	ld	a0, 24(a3)
   12ba8: 08 ef        	sd	a0, 24(a4)
   12baa: 14 eb        	sd	a3, 16(a4)
   12bac: 42 64        	ld	s0, 16(sp)
   12bae: 18 e9        	sd	a4, 16(a0)
   12bb0: e2 60        	ld	ra, 24(sp)
   12bb2: 26 85        	mv	a0, s1
   12bb4: a2 64        	ld	s1, 8(sp)
   12bb6: 98 ee        	sd	a4, 24(a3)
   12bb8: 05 61        	addi	sp, sp, 32
   12bba: 6f 30 40 23  	j	0x15dee <__malloc_unlock>
   12bbe: 82 80        	ret
   12bc0: 37 05 02 00  	lui	a0, 32
   12bc4: b6 97        	add	a5, a5, a3
   12bc6: 13 05 05 f5  	addi	a0, a0, -176
   12bca: 14 6a        	ld	a3, 16(a2)
   12bcc: 63 80 a6 0a  	beq	a3, a0, 0x12c6c <_free_r+0x1d8>
   12bd0: 03 38 86 01  	ld	a6, 24(a2)
   12bd4: 13 e5 17 00  	ori	a0, a5, 1
   12bd8: 33 06 f7 00  	add	a2, a4, a5
   12bdc: 23 bc 06 01  	sd	a6, 24(a3)
   12be0: 23 38 d8 00  	sd	a3, 16(a6)
   12be4: 08 e7        	sd	a0, 8(a4)
   12be6: 1c e2        	sd	a5, 0(a2)
   12be8: 25 b7        	j	0x12b10 <_free_r+0x7c>
   12bea: 63 17 08 0e  	bnez	a6, 0x12cd8 <_free_r+0x244>
   12bee: 0c 6a        	ld	a1, 16(a2)
   12bf0: 10 6e        	ld	a2, 24(a2)
   12bf2: be 96        	add	a3, a3, a5
   12bf4: 93 e7 16 00  	ori	a5, a3, 1
   12bf8: 90 ed        	sd	a2, 24(a1)
   12bfa: 0c ea        	sd	a1, 16(a2)
   12bfc: 1c e7        	sd	a5, 8(a4)
   12bfe: 36 97        	add	a4, a4, a3
   12c00: 14 e3        	sd	a3, 0(a4)
   12c02: 99 b7        	j	0x12b48 <_free_r+0xb4>
   12c04: 05 89        	andi	a0, a0, 1
   12c06: be 96        	add	a3, a3, a5
   12c08: 09 e9        	bnez	a0, 0x12c1a <_free_r+0x186>
   12c0a: 03 35 04 ff  	ld	a0, -16(s0)
   12c0e: 09 8f        	sub	a4, a4, a0
   12c10: 1c 6f        	ld	a5, 24(a4)
   12c12: 10 6b        	ld	a2, 16(a4)
   12c14: aa 96        	add	a3, a3, a0
   12c16: 1c ee        	sd	a5, 24(a2)
   12c18: 90 eb        	sd	a2, 16(a5)
   12c1a: 13 e6 16 00  	ori	a2, a3, 1
   12c1e: 83 b7 81 86  	ld	a5, -1944(gp)
   12c22: 10 e7        	sd	a2, 8(a4)
   12c24: 98 e9        	sd	a4, 16(a1)
   12c26: e3 e1 f6 f2  	bltu	a3, a5, 0x12b48 <_free_r+0xb4>
   12c2a: 83 b5 01 89  	ld	a1, -1904(gp)
   12c2e: 26 85        	mv	a0, s1
   12c30: ef f0 1f d9  	jal	0x129c0 <_malloc_trim_r>
   12c34: 11 bf        	j	0x12b48 <_free_r+0xb4>
   12c36: b6 97        	add	a5, a5, a3
   12c38: 49 bf        	j	0x12bca <_free_r+0x136>
   12c3a: 51 46        	addi	a2, zero, 20
   12c3c: 63 70 d6 02  	bgeu	a2, a3, 0x12c5c <_free_r+0x1c8>
   12c40: 13 06 40 05  	addi	a2, zero, 84
   12c44: 63 69 d6 04  	bltu	a2, a3, 0x12c96 <_free_r+0x202>
   12c48: 93 d6 c7 00  	srli	a3, a5, 12
   12c4c: 1b 85 f6 06  	addiw	a0, a3, 111
   12c50: 1b 15 15 00  	slliw	a0, a0, 1
   12c54: 1b 86 e6 06  	addiw	a2, a3, 110
   12c58: 0e 05        	slli	a0, a0, 3
   12c5a: 0d bf        	j	0x12b8c <_free_r+0xf8>
   12c5c: 1b 85 c6 05  	addiw	a0, a3, 92
   12c60: 1b 15 15 00  	slliw	a0, a0, 1
   12c64: 1b 86 b6 05  	addiw	a2, a3, 91
   12c68: 0e 05        	slli	a0, a0, 3
   12c6a: 0d b7        	j	0x12b8c <_free_r+0xf8>
   12c6c: 98 f5        	sd	a4, 40(a1)
   12c6e: 98 f1        	sd	a4, 32(a1)
   12c70: 93 e6 17 00  	ori	a3, a5, 1
   12c74: 08 ef        	sd	a0, 24(a4)
   12c76: 08 eb        	sd	a0, 16(a4)
   12c78: 14 e7        	sd	a3, 8(a4)
   12c7a: 3e 97        	add	a4, a4, a5
   12c7c: 1c e3        	sd	a5, 0(a4)
   12c7e: e9 b5        	j	0x12b48 <_free_r+0xb4>
   12c80: 03 b8 85 00  	ld	a6, 8(a1)
   12c84: 1b 56 26 40  	sraiw	a2, a2, 2
   12c88: 85 47        	addi	a5, zero, 1
   12c8a: b3 97 c7 00  	sll	a5, a5, a2
   12c8e: b3 e7 07 01  	or	a5, a5, a6
   12c92: 9c e5        	sd	a5, 8(a1)
   12c94: 11 bf        	j	0x12ba8 <_free_r+0x114>
   12c96: 13 06 40 15  	addi	a2, zero, 340
   12c9a: 63 6c d6 00  	bltu	a2, a3, 0x12cb2 <_free_r+0x21e>
   12c9e: 93 d6 f7 00  	srli	a3, a5, 15
   12ca2: 1b 85 86 07  	addiw	a0, a3, 120
   12ca6: 1b 15 15 00  	slliw	a0, a0, 1
   12caa: 1b 86 76 07  	addiw	a2, a3, 119
   12cae: 0e 05        	slli	a0, a0, 3
   12cb0: f1 bd        	j	0x12b8c <_free_r+0xf8>
   12cb2: 13 06 40 55  	addi	a2, zero, 1364
   12cb6: 63 6c d6 00  	bltu	a2, a3, 0x12cce <_free_r+0x23a>
   12cba: 93 d6 27 01  	srli	a3, a5, 18
   12cbe: 1b 85 d6 07  	addiw	a0, a3, 125
   12cc2: 1b 15 15 00  	slliw	a0, a0, 1
   12cc6: 1b 86 c6 07  	addiw	a2, a3, 124
   12cca: 0e 05        	slli	a0, a0, 3
   12ccc: c1 b5        	j	0x12b8c <_free_r+0xf8>
   12cce: 13 05 00 7f  	addi	a0, zero, 2032
   12cd2: 13 06 e0 07  	addi	a2, zero, 126
   12cd6: 5d bd        	j	0x12b8c <_free_r+0xf8>
   12cd8: 93 e6 17 00  	ori	a3, a5, 1
   12cdc: 14 e7        	sd	a3, 8(a4)
   12cde: 1c e2        	sd	a5, 0(a2)
   12ce0: a5 b5        	j	0x12b48 <_free_r+0xb4>

0000000000012ce2 <_fwalk>:
   12ce2: 5d 71        	addi	sp, sp, -80
   12ce4: 4a f8        	sd	s2, 48(sp)
   12ce6: 4e f4        	sd	s3, 40(sp)
   12ce8: 52 f0        	sd	s4, 32(sp)
   12cea: 56 ec        	sd	s5, 24(sp)
   12cec: 5a e8        	sd	s6, 16(sp)
   12cee: 5e e4        	sd	s7, 8(sp)
   12cf0: 86 e4        	sd	ra, 72(sp)
   12cf2: a2 e0        	sd	s0, 64(sp)
   12cf4: 26 fc        	sd	s1, 56(sp)
   12cf6: ae 8a        	mv	s5, a1
   12cf8: 93 09 05 52  	addi	s3, a0, 1312
   12cfc: 01 49        	mv	s2, zero
   12cfe: 13 0b 00 0b  	addi	s6, zero, 176
   12d02: 85 4b        	addi	s7, zero, 1
   12d04: 7d 5a        	addi	s4, zero, -1
   12d06: 83 a4 89 00  	lw	s1, 8(s3)
   12d0a: 03 b4 09 01  	ld	s0, 16(s3)
   12d0e: 63 58 90 02  	blez	s1, 0x12d3e <_fwalk+0x5c>
   12d12: 82 14        	slli	s1, s1, 32
   12d14: 81 90        	srli	s1, s1, 32
   12d16: b3 84 64 03  	mul	s1, s1, s6
   12d1a: a2 94        	add	s1, s1, s0
   12d1c: 83 57 04 01  	lhu	a5, 16(s0)
   12d20: 63 fb fb 00  	bgeu	s7, a5, 0x12d36 <_fwalk+0x54>
   12d24: 83 17 24 01  	lh	a5, 18(s0)
   12d28: 22 85        	mv	a0, s0
   12d2a: 63 86 47 01  	beq	a5, s4, 0x12d36 <_fwalk+0x54>
   12d2e: 82 9a        	jalr	s5
   12d30: 33 69 25 01  	or	s2, a0, s2
   12d34: 01 29        	sext.w	s2, s2
   12d36: 13 04 04 0b  	addi	s0, s0, 176
   12d3a: e3 91 84 fe  	bne	s1, s0, 0x12d1c <_fwalk+0x3a>
   12d3e: 83 b9 09 00  	ld	s3, 0(s3)
   12d42: e3 92 09 fc  	bnez	s3, 0x12d06 <_fwalk+0x24>
   12d46: a6 60        	ld	ra, 72(sp)
   12d48: 06 64        	ld	s0, 64(sp)
   12d4a: e2 74        	ld	s1, 56(sp)
   12d4c: a2 79        	ld	s3, 40(sp)
   12d4e: 02 7a        	ld	s4, 32(sp)
   12d50: e2 6a        	ld	s5, 24(sp)
   12d52: 42 6b        	ld	s6, 16(sp)
   12d54: a2 6b        	ld	s7, 8(sp)
   12d56: 4a 85        	mv	a0, s2
   12d58: 42 79        	ld	s2, 48(sp)
   12d5a: 61 61        	addi	sp, sp, 80
   12d5c: 82 80        	ret

0000000000012d5e <_fwalk_reent>:
   12d5e: 5d 71        	addi	sp, sp, -80
   12d60: 4a f8        	sd	s2, 48(sp)
   12d62: 4e f4        	sd	s3, 40(sp)
   12d64: 52 f0        	sd	s4, 32(sp)
   12d66: 56 ec        	sd	s5, 24(sp)
   12d68: 5a e8        	sd	s6, 16(sp)
   12d6a: 5e e4        	sd	s7, 8(sp)
   12d6c: 62 e0        	sd	s8, 0(sp)
   12d6e: 86 e4        	sd	ra, 72(sp)
   12d70: a2 e0        	sd	s0, 64(sp)
   12d72: 26 fc        	sd	s1, 56(sp)
   12d74: aa 89        	mv	s3, a0
   12d76: 2e 8b        	mv	s6, a1
   12d78: 13 0a 05 52  	addi	s4, a0, 1312
   12d7c: 01 49        	mv	s2, zero
   12d7e: 93 0b 00 0b  	addi	s7, zero, 176
   12d82: 05 4c        	addi	s8, zero, 1
   12d84: fd 5a        	addi	s5, zero, -1
   12d86: 83 24 8a 00  	lw	s1, 8(s4)
   12d8a: 03 34 0a 01  	ld	s0, 16(s4)
   12d8e: 63 59 90 02  	blez	s1, 0x12dc0 <_fwalk_reent+0x62>
   12d92: 82 14        	slli	s1, s1, 32
   12d94: 81 90        	srli	s1, s1, 32
   12d96: b3 84 74 03  	mul	s1, s1, s7
   12d9a: a2 94        	add	s1, s1, s0
   12d9c: 83 57 04 01  	lhu	a5, 16(s0)
   12da0: 63 7c fc 00  	bgeu	s8, a5, 0x12db8 <_fwalk_reent+0x5a>
   12da4: 83 17 24 01  	lh	a5, 18(s0)
   12da8: a2 85        	mv	a1, s0
   12daa: 4e 85        	mv	a0, s3
   12dac: 63 86 57 01  	beq	a5, s5, 0x12db8 <_fwalk_reent+0x5a>
   12db0: 02 9b        	jalr	s6
   12db2: 33 69 25 01  	or	s2, a0, s2
   12db6: 01 29        	sext.w	s2, s2
   12db8: 13 04 04 0b  	addi	s0, s0, 176
   12dbc: e3 90 84 fe  	bne	s1, s0, 0x12d9c <_fwalk_reent+0x3e>
   12dc0: 03 3a 0a 00  	ld	s4, 0(s4)
   12dc4: e3 11 0a fc  	bnez	s4, 0x12d86 <_fwalk_reent+0x28>
   12dc8: a6 60        	ld	ra, 72(sp)
   12dca: 06 64        	ld	s0, 64(sp)
   12dcc: e2 74        	ld	s1, 56(sp)
   12dce: a2 79        	ld	s3, 40(sp)
   12dd0: 02 7a        	ld	s4, 32(sp)
   12dd2: e2 6a        	ld	s5, 24(sp)
   12dd4: 42 6b        	ld	s6, 16(sp)
   12dd6: a2 6b        	ld	s7, 8(sp)
   12dd8: 02 6c        	ld	s8, 0(sp)
   12dda: 4a 85        	mv	a0, s2
   12ddc: 42 79        	ld	s2, 48(sp)
   12dde: 61 61        	addi	sp, sp, 80
   12de0: 82 80        	ret

0000000000012de2 <eshdn1>:
   12de2: 93 06 45 00  	addi	a3, a0, 4
   12de6: 01 47        	mv	a4, zero
   12de8: 69 05        	addi	a0, a0, 26
   12dea: 61 78        	lui	a6, 1048568
   12dec: 83 d7 06 00  	lhu	a5, 0(a3)
   12df0: 13 f6 17 00  	andi	a2, a5, 1
   12df4: 19 c2        	beqz	a2, 0x12dfa <eshdn1+0x18>
   12df6: 13 67 17 00  	ori	a4, a4, 1
   12dfa: 13 76 27 00  	andi	a2, a4, 2
   12dfe: 85 83        	srli	a5, a5, 1
   12e00: 1b 17 17 00  	slliw	a4, a4, 1
   12e04: b3 e5 07 01  	or	a1, a5, a6
   12e08: 01 c6        	beqz	a2, 0x12e10 <eshdn1+0x2e>
   12e0a: 93 97 05 03  	slli	a5, a1, 48
   12e0e: c1 93        	srli	a5, a5, 48
   12e10: 23 90 f6 00  	sh	a5, 0(a3)
   12e14: 42 17        	slli	a4, a4, 48
   12e16: 89 06        	addi	a3, a3, 2
   12e18: 41 93        	srli	a4, a4, 48
   12e1a: e3 19 d5 fc  	bne	a0, a3, 0x12dec <eshdn1+0xa>
   12e1e: 82 80        	ret

0000000000012e20 <eshup1>:
   12e20: 93 06 85 01  	addi	a3, a0, 24
   12e24: 01 47        	mv	a4, zero
   12e26: 09 05        	addi	a0, a0, 2
   12e28: 83 d7 06 00  	lhu	a5, 0(a3)
   12e2c: 13 d6 f7 00  	srli	a2, a5, 15
   12e30: 9b 97 17 00  	slliw	a5, a5, 1
   12e34: 19 c2        	beqz	a2, 0x12e3a <eshup1+0x1a>
   12e36: 13 67 17 00  	ori	a4, a4, 1
   12e3a: 13 76 27 00  	andi	a2, a4, 2
   12e3e: c2 17        	slli	a5, a5, 48
   12e40: 1b 17 17 00  	slliw	a4, a4, 1
   12e44: c1 93        	srli	a5, a5, 48
   12e46: 19 c2        	beqz	a2, 0x12e4c <eshup1+0x2c>
   12e48: 93 e7 17 00  	ori	a5, a5, 1
   12e4c: 23 90 f6 00  	sh	a5, 0(a3)
   12e50: 42 17        	slli	a4, a4, 48
   12e52: f9 16        	addi	a3, a3, -2
   12e54: 41 93        	srli	a4, a4, 48
   12e56: e3 19 d5 fc  	bne	a0, a3, 0x12e28 <eshup1+0x8>
   12e5a: 82 80        	ret

0000000000012e5c <m16m>:
   12e5c: 01 11        	addi	sp, sp, -32
   12e5e: 41 6e        	lui	t3, 16
   12e60: 23 1b 01 00  	sh	zero, 22(sp)
   12e64: 23 1c 01 00  	sh	zero, 24(sp)
   12e68: e1 05        	addi	a1, a1, 24
   12e6a: 38 08        	addi	a4, sp, 24
   12e6c: 13 08 41 00  	addi	a6, sp, 4
   12e70: 7d 1e        	addi	t3, t3, -1
   12e72: 83 d7 05 00  	lhu	a5, 0(a1)
   12e76: 79 17        	addi	a4, a4, -2
   12e78: f9 15        	addi	a1, a1, -2
   12e7a: 85 c7        	beqz	a5, 0x12ea2 <m16m+0x46>
   12e7c: b3 87 a7 02  	mul	a5, a5, a0
   12e80: 83 56 27 00  	lhu	a3, 2(a4)
   12e84: 03 53 07 00  	lhu	t1, 0(a4)
   12e88: b3 f8 c7 01  	and	a7, a5, t3
   12e8c: c6 96        	add	a3, a3, a7
   12e8e: c1 83        	srli	a5, a5, 16
   12e90: 93 d8 06 01  	srli	a7, a3, 16
   12e94: 9a 97        	add	a5, a5, t1
   12e96: c6 97        	add	a5, a5, a7
   12e98: 23 10 f7 00  	sh	a5, 0(a4)
   12e9c: 23 11 d7 00  	sh	a3, 2(a4)
   12ea0: c1 83        	srli	a5, a5, 16
   12ea2: 23 1f f7 fe  	sh	a5, -2(a4)
   12ea6: e3 16 07 fd  	bne	a4, a6, 0x12e72 <m16m+0x16>
   12eaa: 93 07 46 00  	addi	a5, a2, 4
   12eae: d7 f6 85 c4  	<unknown>
   12eb2: ad 45        	addi	a1, zero, 11
   12eb4: 07 5c 07 02  	<unknown>
   12eb8: 27 dc 07 02  	<unknown>
   12ebc: b3 87 d5 40  	sub	a5, a1, a3
   12ec0: 63 81 d5 02  	beq	a1, a3, 0x12ee2 <m16m+0x86>
   12ec4: 73 27 20 c2  	csrr	a4, vlenb
   12ec8: 93 06 47 fe  	addi	a3, a4, -28
   12ecc: 93 86 06 02  	addi	a3, a3, 32
   12ed0: 11 07        	addi	a4, a4, 4
   12ed2: 8a 96        	add	a3, a3, sp
   12ed4: 3a 96        	add	a2, a2, a4
   12ed6: 57 f0 87 04  	<unknown>
   12eda: 07 dc 06 02  	<unknown>
   12ede: 27 5c 06 02  	<unknown>
   12ee2: 05 61        	addi	sp, sp, 32
   12ee4: 82 80        	ret

0000000000012ee6 <emovo.constprop.0>:
   12ee6: 03 57 05 00  	lhu	a4, 0(a0)
   12eea: 83 57 25 00  	lhu	a5, 2(a0)
   12eee: 19 c3        	beqz	a4, 0x12ef4 <emovo.constprop.0+0xe>
   12ef0: 21 67        	lui	a4, 8
   12ef2: d9 8f        	or	a5, a5, a4
   12ef4: 23 99 f5 00  	sh	a5, 18(a1)
   12ef8: 03 57 25 00  	lhu	a4, 2(a0)
   12efc: a1 67        	lui	a5, 8
   12efe: fd 17        	addi	a5, a5, -1
   12f00: 63 0f f7 00  	beq	a4, a5, 0x12f1e <emovo.constprop.0+0x38>
   12f04: 93 07 65 00  	addi	a5, a0, 6
   12f08: c1 05        	addi	a1, a1, 16
   12f0a: 61 05        	addi	a0, a0, 24
   12f0c: 03 d7 07 00  	lhu	a4, 0(a5)
   12f10: 89 07        	addi	a5, a5, 2
   12f12: f9 15        	addi	a1, a1, -2
   12f14: 23 91 e5 00  	sh	a4, 2(a1)
   12f18: e3 9a a7 fe  	bne	a5, a0, 0x12f0c <emovo.constprop.0+0x26>
   12f1c: 82 80        	ret
   12f1e: 93 07 65 00  	addi	a5, a0, 6
   12f22: 69 05        	addi	a0, a0, 26
   12f24: 03 d7 07 00  	lhu	a4, 0(a5)
   12f28: 89 07        	addi	a5, a5, 2
   12f2a: 29 e3        	bnez	a4, 0x12f6c <emovo.constprop.0+0x86>
   12f2c: e3 9c a7 fe  	bne	a5, a0, 0x12f24 <emovo.constprop.0+0x3e>
   12f30: d7 f7 84 c4  	<unknown>
   12f34: 57 77 80 04  	<unknown>
   12f38: 25 47        	addi	a4, zero, 9
   12f3a: 57 3c 00 5e  	<unknown>
   12f3e: 57 f0 84 c4  	<unknown>
   12f42: 27 dc 05 02  	<unknown>
   12f46: b3 06 f7 40  	sub	a3, a4, a5
   12f4a: 63 09 f7 00  	beq	a4, a5, 0x12f5c <emovo.constprop.0+0x76>
   12f4e: f3 27 20 c2  	csrr	a5, vlenb
   12f52: ae 97        	add	a5, a5, a1
   12f54: 57 f0 86 04  	<unknown>
   12f58: 27 dc 07 02  	<unknown>
   12f5c: 83 d7 25 01  	lhu	a5, 18(a1)
   12f60: 21 67        	lui	a4, 8
   12f62: 7d 17        	addi	a4, a4, -1
   12f64: d9 8f        	or	a5, a5, a4
   12f66: 23 99 f5 00  	sh	a5, 18(a1)
   12f6a: 82 80        	ret
   12f6c: d7 77 80 04  	<unknown>
   12f70: f1 77        	lui	a5, 1048572
   12f72: 57 3c 00 5e  	<unknown>
   12f76: 57 70 84 c4  	<unknown>
   12f7a: 27 dc 05 02  	<unknown>
   12f7e: 23 98 f5 00  	sh	a5, 16(a1)
   12f82: e1 77        	lui	a5, 1048568
   12f84: 93 c7 f7 ff  	not	a5, a5
   12f88: 23 99 f5 00  	sh	a5, 18(a1)
   12f8c: 82 80        	ret

0000000000012f8e <enormlz>:
   12f8e: 83 57 45 00  	lhu	a5, 4(a0)
   12f92: 01 11        	addi	sp, sp, -32
   12f94: 26 e4        	sd	s1, 8(sp)
   12f96: 06 ec        	sd	ra, 24(sp)
   12f98: 22 e8        	sd	s0, 16(sp)
   12f9a: 4a e0        	sd	s2, 0(sp)
   12f9c: aa 84        	mv	s1, a0
   12f9e: ad e7        	bnez	a5, 0x13008 <enormlz+0x7a>
   12fa0: 83 57 65 00  	lhu	a5, 6(a0)
   12fa4: 01 44        	mv	s0, zero
   12fa6: 13 d7 f7 00  	srli	a4, a5, 15
   12faa: 21 eb        	bnez	a4, 0x12ffa <enormlz+0x6c>
   12fac: 73 27 20 c2  	csrr	a4, vlenb
   12fb0: 13 08 67 00  	addi	a6, a4, 6
   12fb4: 13 06 47 00  	addi	a2, a4, 4
   12fb8: d7 75 85 c4  	<unknown>
   12fbc: 29 47        	addi	a4, zero, 10
   12fbe: 93 06 45 00  	addi	a3, a0, 4
   12fc2: 0d 8f        	sub	a4, a4, a1
   12fc4: 19 05        	addi	a0, a0, 6
   12fc6: 26 98        	add	a6, a6, s1
   12fc8: 26 96        	add	a2, a2, s1
   12fca: 93 05 00 0a  	addi	a1, zero, 160
   12fce: 19 a0        	j	0x12fd4 <enormlz+0x46>
   12fd0: 83 d7 64 00  	lhu	a5, 6(s1)
   12fd4: 57 70 85 c4  	<unknown>
   12fd8: c1 eb        	bnez	a5, 0x13068 <enormlz+0xda>
   12fda: 07 5c 05 02  	<unknown>
   12fde: 27 dc 06 02  	<unknown>
   12fe2: 41 24        	addiw	s0, s0, 16
   12fe4: 19 c7        	beqz	a4, 0x12ff2 <enormlz+0x64>
   12fe6: 57 70 87 04  	<unknown>
   12fea: 07 5c 08 02  	<unknown>
   12fee: 27 5c 06 02  	<unknown>
   12ff2: 23 9c 04 00  	sh	zero, 24(s1)
   12ff6: e3 1d b4 fc  	bne	s0, a1, 0x12fd0 <enormlz+0x42>
   12ffa: e2 60        	ld	ra, 24(sp)
   12ffc: 22 85        	mv	a0, s0
   12ffe: 42 64        	ld	s0, 16(sp)
   13000: a2 64        	ld	s1, 8(sp)
   13002: 02 69        	ld	s2, 0(sp)
   13004: 05 61        	addi	sp, sp, 32
   13006: 82 80        	ret
   13008: 13 07 f0 0f  	addi	a4, zero, 255
   1300c: 01 44        	mv	s0, zero
   1300e: 63 65 f7 02  	bltu	a4, a5, 0x13038 <enormlz+0xaa>
   13012: 13 09 f0 f6  	addi	s2, zero, -145
   13016: 01 a8        	j	0x13026 <enormlz+0x98>
   13018: 7d 34        	addiw	s0, s0, -1
   1301a: ef f0 9f dc  	jal	0x12de2 <eshdn1>
   1301e: e3 0e 24 fd  	beq	s0, s2, 0x12ffa <enormlz+0x6c>
   13022: 83 d7 44 00  	lhu	a5, 4(s1)
   13026: 26 85        	mv	a0, s1
   13028: e5 fb        	bnez	a5, 0x13018 <enormlz+0x8a>
   1302a: e2 60        	ld	ra, 24(sp)
   1302c: 22 85        	mv	a0, s0
   1302e: 42 64        	ld	s0, 16(sp)
   13030: a2 64        	ld	s1, 8(sp)
   13032: 02 69        	ld	s2, 0(sp)
   13034: 05 61        	addi	sp, sp, 32
   13036: 82 80        	ret
   13038: 13 07 45 00  	addi	a4, a0, 4
   1303c: 93 05 a5 01  	addi	a1, a0, 26
   13040: 81 47        	mv	a5, zero
   13042: 83 56 07 00  	lhu	a3, 0(a4)
   13046: 09 07        	addi	a4, a4, 2
   13048: 1b d6 86 00  	srliw	a2, a3, 8
   1304c: d1 8f        	or	a5, a5, a2
   1304e: 9b 96 86 00  	slliw	a3, a3, 8
   13052: 23 1f f7 fe  	sh	a5, -2(a4)
   13056: 93 97 06 03  	slli	a5, a3, 48
   1305a: c1 93        	srli	a5, a5, 48
   1305c: e3 13 b7 fe  	bne	a4, a1, 0x13042 <enormlz+0xb4>
   13060: 83 d7 44 00  	lhu	a5, 4(s1)
   13064: 61 54        	addi	s0, zero, -8
   13066: 75 b7        	j	0x13012 <enormlz+0x84>
   13068: 13 07 f0 0f  	addi	a4, zero, 255
   1306c: 63 6b f7 02  	bltu	a4, a5, 0x130a2 <enormlz+0x114>
   13070: 13 88 84 01  	addi	a6, s1, 24
   13074: 93 85 24 00  	addi	a1, s1, 2
   13078: 13 05 f0 0f  	addi	a0, zero, 255
   1307c: 81 47        	mv	a5, zero
   1307e: 42 87        	mv	a4, a6
   13080: 83 56 07 00  	lhu	a3, 0(a4)
   13084: 79 17        	addi	a4, a4, -2
   13086: 1b 96 86 00  	slliw	a2, a3, 8
   1308a: d1 8f        	or	a5, a5, a2
   1308c: 23 11 f7 00  	sh	a5, 2(a4)
   13090: 93 d7 86 00  	srli	a5, a3, 8
   13094: e3 16 b7 fe  	bne	a4, a1, 0x13080 <enormlz+0xf2>
   13098: 83 d7 64 00  	lhu	a5, 6(s1)
   1309c: 21 24        	addiw	s0, s0, 8
   1309e: e3 7f f5 fc  	bgeu	a0, a5, 0x1307c <enormlz+0xee>
   130a2: 13 09 00 0a  	addi	s2, zero, 160
   130a6: 01 a8        	j	0x130b6 <enormlz+0x128>
   130a8: 05 24        	addiw	s0, s0, 1
   130aa: ef f0 7f d7  	jal	0x12e20 <eshup1>
   130ae: e3 46 89 f4  	blt	s2, s0, 0x12ffa <enormlz+0x6c>
   130b2: 83 d7 64 00  	lhu	a5, 6(s1)
   130b6: bd 83        	srli	a5, a5, 15
   130b8: 26 85        	mv	a0, s1
   130ba: fd d7        	beqz	a5, 0x130a8 <enormlz+0x11a>
   130bc: 3d bf        	j	0x12ffa <enormlz+0x6c>

00000000000130be <eshift.part.0>:
   130be: 79 71        	addi	sp, sp, -48
   130c0: 22 f0        	sd	s0, 32(sp)
   130c2: 26 ec        	sd	s1, 24(sp)
   130c4: 06 f4        	sd	ra, 40(sp)
   130c6: 4a e8        	sd	s2, 16(sp)
   130c8: 4e e4        	sd	s3, 8(sp)
   130ca: ae 84        	mv	s1, a1
   130cc: 2a 84        	mv	s0, a0
   130ce: 63 cc 05 08  	bltz	a1, 0x13166 <eshift.part.0+0xa8>
   130d2: bd 47        	addi	a5, zero, 15
   130d4: 63 d6 b7 04  	bge	a5, a1, 0x13120 <eshift.part.0+0x62>
   130d8: f3 27 20 c2  	csrr	a5, vlenb
   130dc: 13 88 67 00  	addi	a6, a5, 6
   130e0: 13 86 47 00  	addi	a2, a5, 4
   130e4: 29 47        	addi	a4, zero, 10
   130e6: d7 77 85 c4  	<unknown>
   130ea: 93 06 45 00  	addi	a3, a0, 4
   130ee: 1d 8f        	sub	a4, a4, a5
   130f0: 19 05        	addi	a0, a0, 6
   130f2: ae 87        	mv	a5, a1
   130f4: 22 98        	add	a6, a6, s0
   130f6: 22 96        	add	a2, a2, s0
   130f8: bd 45        	addi	a1, zero, 15
   130fa: 57 70 85 c4  	<unknown>
   130fe: 07 5c 05 02  	<unknown>
   13102: 27 dc 06 02  	<unknown>
   13106: c1 37        	addiw	a5, a5, -16
   13108: 19 c7        	beqz	a4, 0x13116 <eshift.part.0+0x58>
   1310a: 57 70 87 04  	<unknown>
   1310e: 07 5c 08 02  	<unknown>
   13112: 27 5c 06 02  	<unknown>
   13116: 23 1c 04 00  	sh	zero, 24(s0)
   1311a: e3 c0 f5 fe  	blt	a1, a5, 0x130fa <eshift.part.0+0x3c>
   1311e: bd 88        	andi	s1, s1, 15
   13120: 9d 47        	addi	a5, zero, 7
   13122: 63 d4 97 02  	bge	a5, s1, 0x1314a <eshift.part.0+0x8c>
   13126: 13 07 84 01  	addi	a4, s0, 24
   1312a: 93 05 24 00  	addi	a1, s0, 2
   1312e: 81 47        	mv	a5, zero
   13130: 83 56 07 00  	lhu	a3, 0(a4)
   13134: 79 17        	addi	a4, a4, -2
   13136: 1b 96 86 00  	slliw	a2, a3, 8
   1313a: d1 8f        	or	a5, a5, a2
   1313c: 23 11 f7 00  	sh	a5, 2(a4)
   13140: 93 d7 86 00  	srli	a5, a3, 8
   13144: e3 16 b7 fe  	bne	a4, a1, 0x13130 <eshift.part.0+0x72>
   13148: e1 34        	addiw	s1, s1, -8
   1314a: 91 c4        	beqz	s1, 0x13156 <eshift.part.0+0x98>
   1314c: fd 34        	addiw	s1, s1, -1
   1314e: 22 85        	mv	a0, s0
   13150: ef f0 1f cd  	jal	0x12e20 <eshup1>
   13154: e5 fc        	bnez	s1, 0x1314c <eshift.part.0+0x8e>
   13156: 01 45        	mv	a0, zero
   13158: a2 70        	ld	ra, 40(sp)
   1315a: 02 74        	ld	s0, 32(sp)
   1315c: e2 64        	ld	s1, 24(sp)
   1315e: 42 69        	ld	s2, 16(sp)
   13160: a2 69        	ld	s3, 8(sp)
   13162: 45 61        	addi	sp, sp, 48
   13164: 82 80        	ret
   13166: c5 57        	addi	a5, zero, -15
   13168: 3b 09 b0 40  	negw	s2, a1
   1316c: 81 49        	mv	s3, zero
   1316e: 93 05 85 01  	addi	a1, a0, 24
   13172: 93 06 45 00  	addi	a3, a0, 4
   13176: 3d 46        	addi	a2, zero, 15
   13178: 63 d4 f4 0a  	bge	s1, a5, 0x13220 <eshift.part.0+0x162>
   1317c: 03 57 84 01  	lhu	a4, 24(s0)
   13180: ae 87        	mv	a5, a1
   13182: b3 e9 e9 00  	or	s3, s3, a4
   13186: 03 d7 e7 ff  	lhu	a4, -2(a5)
   1318a: f9 17        	addi	a5, a5, -2
   1318c: 23 91 e7 00  	sh	a4, 2(a5)
   13190: e3 9b d7 fe  	bne	a5, a3, 0x13186 <eshift.part.0+0xc8>
   13194: 23 12 04 00  	sh	zero, 4(s0)
   13198: 41 39        	addiw	s2, s2, -16
   1319a: e3 41 26 ff  	blt	a2, s2, 0x1317c <eshift.part.0+0xbe>
   1319e: 41 59        	addi	s2, zero, -16
   131a0: 45 57        	addi	a4, zero, -15
   131a2: 3b 09 99 40  	subw	s2, s2, s1
   131a6: 81 47        	mv	a5, zero
   131a8: 63 c7 e4 06  	blt	s1, a4, 0x13216 <eshift.part.0+0x158>
   131ac: 3b 09 f9 00  	addw	s2, s2, a5
   131b0: 9d 47        	addi	a5, zero, 7
   131b2: 63 d9 27 03  	bge	a5, s2, 0x131e4 <eshift.part.0+0x126>
   131b6: 83 47 84 01  	lbu	a5, 24(s0)
   131ba: 93 05 a4 01  	addi	a1, s0, 26
   131be: b3 e9 f9 00  	or	s3, s3, a5
   131c2: 81 47        	mv	a5, zero
   131c4: 03 d7 06 00  	lhu	a4, 0(a3)
   131c8: 89 06        	addi	a3, a3, 2
   131ca: 1b 56 87 00  	srliw	a2, a4, 8
   131ce: d1 8f        	or	a5, a5, a2
   131d0: 1b 17 87 00  	slliw	a4, a4, 8
   131d4: 23 9f f6 fe  	sh	a5, -2(a3)
   131d8: 93 17 07 03  	slli	a5, a4, 48
   131dc: c1 93        	srli	a5, a5, 48
   131de: e3 93 b6 fe  	bne	a3, a1, 0x131c4 <eshift.part.0+0x106>
   131e2: 61 39        	addiw	s2, s2, -8
   131e4: 33 35 30 01  	snez	a0, s3
   131e8: e3 08 09 f6  	beqz	s2, 0x13158 <eshift.part.0+0x9a>
   131ec: 83 57 84 01  	lhu	a5, 24(s0)
   131f0: 7d 39        	addiw	s2, s2, -1
   131f2: 22 85        	mv	a0, s0
   131f4: 85 8b        	andi	a5, a5, 1
   131f6: b3 e9 f9 00  	or	s3, s3, a5
   131fa: ef f0 9f be  	jal	0x12de2 <eshdn1>
   131fe: e3 17 09 fe  	bnez	s2, 0x131ec <eshift.part.0+0x12e>
   13202: a2 70        	ld	ra, 40(sp)
   13204: 02 74        	ld	s0, 32(sp)
   13206: 81 29        	sext.w	s3, s3
   13208: e2 64        	ld	s1, 24(sp)
   1320a: 42 69        	ld	s2, 16(sp)
   1320c: 33 35 30 01  	snez	a0, s3
   13210: a2 69        	ld	s3, 8(sp)
   13212: 45 61        	addi	sp, sp, 48
   13214: 82 80        	ret
   13216: 93 77 09 ff  	andi	a5, s2, -16
   1321a: bb 07 f0 40  	negw	a5, a5
   1321e: 79 b7        	j	0x131ac <eshift.part.0+0xee>
   13220: e5 57        	addi	a5, zero, -7
   13222: e3 ca f4 f8  	blt	s1, a5, 0x131b6 <eshift.part.0+0xf8>
   13226: d9 b7        	j	0x131ec <eshift.part.0+0x12e>

0000000000013228 <emovi>:
   13228: 83 57 25 01  	lhu	a5, 18(a0)
   1322c: a1 66        	lui	a3, 8
   1322e: fd 16        	addi	a3, a3, -1
   13230: 9b d7 f7 00  	srliw	a5, a5, 15
   13234: bb 07 f0 40  	negw	a5, a5
   13238: 23 90 f5 00  	sh	a5, 0(a1)
   1323c: 03 57 25 01  	lhu	a4, 18(a0)
   13240: 93 07 05 01  	addi	a5, a0, 16
   13244: 75 8f        	and	a4, a4, a3
   13246: 23 91 e5 00  	sh	a4, 2(a1)
   1324a: 1b 06 07 00  	sext.w	a2, a4
   1324e: 63 02 d7 02  	beq	a4, a3, 0x13272 <emovi+0x4a>
   13252: 13 87 65 00  	addi	a4, a1, 6
   13256: 23 92 05 00  	sh	zero, 4(a1)
   1325a: 79 15        	addi	a0, a0, -2
   1325c: 83 d6 07 00  	lhu	a3, 0(a5)
   13260: f9 17        	addi	a5, a5, -2
   13262: 09 07        	addi	a4, a4, 2
   13264: 23 1f d7 fe  	sh	a3, -2(a4)
   13268: e3 1a f5 fe  	bne	a0, a5, 0x1325c <emovi+0x34>
   1326c: 23 9c 05 00  	sh	zero, 24(a1)
   13270: 82 80        	ret
   13272: 03 57 25 01  	lhu	a4, 18(a0)
   13276: 71 8f        	and	a4, a4, a2
   13278: 63 1b c7 00  	bne	a4, a2, 0x1328e <emovi+0x66>
   1327c: 13 06 25 01  	addi	a2, a0, 18
   13280: 2a 87        	mv	a4, a0
   13282: 83 56 07 00  	lhu	a3, 0(a4)
   13286: 09 07        	addi	a4, a4, 2
   13288: 95 ee        	bnez	a3, 0x132c4 <emovi+0x9c>
   1328a: e3 1c e6 fe  	bne	a2, a4, 0x13282 <emovi+0x5a>
   1328e: 57 f7 85 c4  	<unknown>
   13292: d7 77 80 04  	<unknown>
   13296: 93 87 45 00  	addi	a5, a1, 4
   1329a: 57 3c 00 5e  	<unknown>
   1329e: ad 46        	addi	a3, zero, 11
   132a0: 57 f0 85 c4  	<unknown>
   132a4: 27 dc 07 02  	<unknown>
   132a8: 33 86 e6 40  	sub	a2, a3, a4
   132ac: 63 8b e6 00  	beq	a3, a4, 0x132c2 <emovi+0x9a>
   132b0: f3 27 20 c2  	csrr	a5, vlenb
   132b4: 91 07        	addi	a5, a5, 4
   132b6: ae 97        	add	a5, a5, a1
   132b8: 57 70 86 04  	<unknown>
   132bc: 27 dc 07 02  	<unknown>
   132c0: 82 80        	ret
   132c2: 82 80        	ret
   132c4: 13 87 65 00  	addi	a4, a1, 6
   132c8: 23 92 05 00  	sh	zero, 4(a1)
   132cc: 71 15        	addi	a0, a0, -4
   132ce: 83 d6 07 00  	lhu	a3, 0(a5)
   132d2: f9 17        	addi	a5, a5, -2
   132d4: 09 07        	addi	a4, a4, 2
   132d6: 23 1f d7 fe  	sh	a3, -2(a4)
   132da: e3 1a f5 fe  	bne	a0, a5, 0x132ce <emovi+0xa6>
   132de: 82 80        	ret

00000000000132e0 <ecmp>:
   132e0: 83 57 25 01  	lhu	a5, 18(a0)
   132e4: 5d 71        	addi	sp, sp, -80
   132e6: a2 e0        	sd	s0, 64(sp)
   132e8: 93 c7 f7 ff  	not	a5, a5
   132ec: 86 e4        	sd	ra, 72(sp)
   132ee: 13 97 17 03  	slli	a4, a5, 49
   132f2: 2e 84        	mv	s0, a1
   132f4: 11 eb        	bnez	a4, 0x13308 <ecmp+0x28>
   132f6: 93 06 25 01  	addi	a3, a0, 18
   132fa: aa 87        	mv	a5, a0
   132fc: 03 d7 07 00  	lhu	a4, 0(a5)
   13300: 89 07        	addi	a5, a5, 2
   13302: 25 eb        	bnez	a4, 0x13372 <ecmp+0x92>
   13304: e3 9c d7 fe  	bne	a5, a3, 0x132fc <ecmp+0x1c>
   13308: 83 57 24 01  	lhu	a5, 18(s0)
   1330c: 93 c7 f7 ff  	not	a5, a5
   13310: 13 97 17 03  	slli	a4, a5, 49
   13314: 31 c3        	beqz	a4, 0x13358 <ecmp+0x78>
   13316: 8a 85        	mv	a1, sp
   13318: ef f0 1f f1  	jal	0x13228 <emovi>
   1331c: 0c 10        	addi	a1, sp, 32
   1331e: 22 85        	mv	a0, s0
   13320: ef f0 9f f0  	jal	0x13228 <emovi>
   13324: 83 55 01 00  	lhu	a1, 0(sp)
   13328: 83 57 01 02  	lhu	a5, 32(sp)
   1332c: 63 88 b7 04  	beq	a5, a1, 0x1337c <ecmp+0x9c>
   13330: 93 07 21 00  	addi	a5, sp, 2
   13334: 13 07 21 02  	addi	a4, sp, 34
   13338: 30 08        	addi	a2, sp, 24
   1333a: 83 d6 07 00  	lhu	a3, 0(a5)
   1333e: 89 07        	addi	a5, a5, 2
   13340: b5 ea        	bnez	a3, 0x133b4 <ecmp+0xd4>
   13342: 83 56 07 00  	lhu	a3, 0(a4)
   13346: 09 07        	addi	a4, a4, 2
   13348: b5 e6        	bnez	a3, 0x133b4 <ecmp+0xd4>
   1334a: e3 18 f6 fe  	bne	a2, a5, 0x1333a <ecmp+0x5a>
   1334e: 01 45        	mv	a0, zero
   13350: a6 60        	ld	ra, 72(sp)
   13352: 06 64        	ld	s0, 64(sp)
   13354: 61 61        	addi	sp, sp, 80
   13356: 82 80        	ret
   13358: a2 87        	mv	a5, s0
   1335a: 03 d7 07 00  	lhu	a4, 0(a5)
   1335e: 93 06 24 01  	addi	a3, s0, 18
   13362: 89 07        	addi	a5, a5, 2
   13364: 19 e7        	bnez	a4, 0x13372 <ecmp+0x92>
   13366: e3 88 d7 fa  	beq	a5, a3, 0x13316 <ecmp+0x36>
   1336a: 03 d7 07 00  	lhu	a4, 0(a5)
   1336e: 89 07        	addi	a5, a5, 2
   13370: 7d db        	beqz	a4, 0x13366 <ecmp+0x86>
   13372: a6 60        	ld	ra, 72(sp)
   13374: 06 64        	ld	s0, 64(sp)
   13376: 79 55        	addi	a0, zero, -2
   13378: 61 61        	addi	sp, sp, 80
   1337a: 82 80        	ret
   1337c: 7d 55        	addi	a0, zero, -1
   1337e: a9 c1        	beqz	a1, 0x133c0 <ecmp+0xe0>
   13380: 8a 87        	mv	a5, sp
   13382: 03 d6 07 00  	lhu	a2, 0(a5)
   13386: 83 56 01 02  	lhu	a3, 32(sp)
   1338a: 18 10        	addi	a4, sp, 32
   1338c: 2c 08        	addi	a1, sp, 24
   1338e: 89 07        	addi	a5, a5, 2
   13390: 09 07        	addi	a4, a4, 2
   13392: 63 1c d6 00  	bne	a2, a3, 0x133aa <ecmp+0xca>
   13396: e3 8c b7 fa  	beq	a5, a1, 0x1334e <ecmp+0x6e>
   1339a: 03 d6 07 00  	lhu	a2, 0(a5)
   1339e: 83 56 07 00  	lhu	a3, 0(a4)
   133a2: 89 07        	addi	a5, a5, 2
   133a4: 09 07        	addi	a4, a4, 2
   133a6: e3 08 d6 fe  	beq	a2, a3, 0x13396 <ecmp+0xb6>
   133aa: e3 e3 c6 fa  	bltu	a3, a2, 0x13350 <ecmp+0x70>
   133ae: 3b 05 a0 40  	negw	a0, a0
   133b2: 79 bf        	j	0x13350 <ecmp+0x70>
   133b4: 81 e9        	bnez	a1, 0x133c4 <ecmp+0xe4>
   133b6: a6 60        	ld	ra, 72(sp)
   133b8: 06 64        	ld	s0, 64(sp)
   133ba: 05 45        	addi	a0, zero, 1
   133bc: 61 61        	addi	sp, sp, 80
   133be: 82 80        	ret
   133c0: 05 45        	addi	a0, zero, 1
   133c2: 7d bf        	j	0x13380 <ecmp+0xa0>
   133c4: 7d 55        	addi	a0, zero, -1
   133c6: 69 b7        	j	0x13350 <ecmp+0x70>

00000000000133c8 <emdnorm>:
   133c8: 39 71        	addi	sp, sp, -64
   133ca: 22 f8        	sd	s0, 48(sp)
   133cc: 26 f4        	sd	s1, 40(sp)
   133ce: 4a f0        	sd	s2, 32(sp)
   133d0: 4e ec        	sd	s3, 24(sp)
   133d2: 52 e8        	sd	s4, 16(sp)
   133d4: 56 e4        	sd	s5, 8(sp)
   133d6: b6 84        	mv	s1, a3
   133d8: 3e 84        	mv	s0, a5
   133da: 06 fc        	sd	ra, 56(sp)
   133dc: 2a 8a        	mv	s4, a0
   133de: 2e 89        	mv	s2, a1
   133e0: b2 89        	mv	s3, a2
   133e2: ba 8a        	mv	s5, a4
   133e4: ef f0 bf ba  	jal	0x12f8e <enormlz>
   133e8: 93 07 00 09  	addi	a5, zero, 144
   133ec: 89 8c        	sub	s1, s1, a0
   133ee: 63 d7 a7 16  	bge	a5, a0, 0x1355c <emdnorm+0x194>
   133f2: a1 67        	lui	a5, 8
   133f4: f9 17        	addi	a5, a5, -2
   133f6: 63 d2 97 1a  	bge	a5, s1, 0x1359a <emdnorm+0x1d2>
   133fa: 63 8c 0a 20  	beqz	s5, 0x13612 <emdnorm+0x24a>
   133fe: 83 2a 44 00  	lw	s5, 4(s0)
   13402: 1c 40        	lw	a5, 0(s0)
   13404: 63 84 fa 28  	beq	s5, a5, 0x1368c <emdnorm+0x2c4>
   13408: 57 f7 86 c4  	<unknown>
   1340c: d7 77 80 04  	<unknown>
   13410: 93 07 a4 01  	addi	a5, s0, 26
   13414: 57 3c 00 5e  	<unknown>
   13418: b5 46        	addi	a3, zero, 13
   1341a: 57 f0 86 c4  	<unknown>
   1341e: 27 dc 07 02  	<unknown>
   13422: 33 86 e6 40  	sub	a2, a3, a4
   13426: 63 96 e6 28  	bne	a3, a4, 0x136b2 <emdnorm+0x2ea>
   1342a: 93 07 80 03  	addi	a5, zero, 56
   1342e: 63 89 fa 34  	beq	s5, a5, 0x13780 <emdnorm+0x3b8>
   13432: 63 de 57 19  	bge	a5, s5, 0x135ce <emdnorm+0x206>
   13436: 93 07 00 04  	addi	a5, zero, 64
   1343a: 63 85 fa 28  	beq	s5, a5, 0x136c4 <emdnorm+0x2fc>
   1343e: 93 07 10 07  	addi	a5, zero, 113
   13442: 63 9e fa 30  	bne	s5, a5, 0x1375e <emdnorm+0x396>
   13446: b7 87 00 40  	lui	a5, 262152
   1344a: a9 46        	addi	a3, zero, 10
   1344c: 61 77        	lui	a4, 1048568
   1344e: fd 17        	addi	a5, a5, -1
   13450: 14 c4        	sw	a3, 8(s0)
   13452: 5c c8        	sw	a5, 20(s0)
   13454: 23 1c e4 00  	sh	a4, 24(s0)
   13458: 54 c4        	sw	a3, 12(s0)
   1345a: 23 17 e4 02  	sh	a4, 46(s0)
   1345e: 23 20 54 01  	sw	s5, 0(s0)
   13462: 63 5b 90 20  	blez	s1, 0x13678 <emdnorm+0x2b0>
   13466: 03 55 4a 01  	lhu	a0, 20(s4)
   1346a: 13 06 4a 01  	addi	a2, s4, 20
   1346e: ad 47        	addi	a5, zero, 11
   13470: 46 15        	slli	a0, a0, 49
   13472: 45 91        	srli	a0, a0, 49
   13474: 35 47        	addi	a4, zero, 13
   13476: 1d 9f        	subw	a4, a4, a5
   13478: 02 17        	slli	a4, a4, 32
   1347a: 86 07        	slli	a5, a5, 1
   1347c: 01 93        	srli	a4, a4, 32
   1347e: d2 97        	add	a5, a5, s4
   13480: f3 25 20 c2  	csrr	a1, vlenb
   13484: d7 76 80 04  	<unknown>
   13488: 57 3c 00 5e  	<unknown>
   1348c: d7 3c 80 9f  	<unknown>
   13490: d7 76 87 04  	<unknown>
   13494: 07 d0 07 02  	<unknown>
   13498: 15 8f        	sub	a4, a4, a3
   1349a: a7 dc 07 02  	<unknown>
   1349e: 57 30 00 66  	<unknown>
   134a2: ae 97        	add	a5, a5, a1
   134a4: 57 bd 90 5d  	<unknown>
   134a8: 57 70 80 00  	<unknown>
   134ac: 57 0c 8d 2b  	<unknown>
   134b0: 65 f3        	bnez	a4, 0x13490 <emdnorm+0xc8>
   134b2: d7 77 80 00  	<unknown>
   134b6: 03 57 06 00  	lhu	a4, 0(a2)
   134ba: d7 6c 00 42  	<unknown>
   134be: 83 56 44 01  	lhu	a3, 20(s0)
   134c2: d7 ac 8c 0b  	<unknown>
   134c6: 57 70 80 04  	<unknown>
   134ca: d7 27 90 43  	<unknown>
   134ce: c2 17        	slli	a5, a5, 48
   134d0: c1 93        	srli	a5, a5, 48
   134d2: 5d 8d        	or	a0, a0, a5
   134d4: 93 c6 f6 ff  	not	a3, a3
   134d8: 75 8f        	and	a4, a4, a3
   134da: 23 10 e6 00  	sh	a4, 0(a2)
   134de: 83 57 64 01  	lhu	a5, 22(s0)
   134e2: 33 f7 a7 00  	and	a4, a5, a0
   134e6: 21 c7        	beqz	a4, 0x1352e <emdnorm+0x166>
   134e8: 63 9d a7 00  	bne	a5, a0, 0x13502 <emdnorm+0x13a>
   134ec: 63 16 09 26  	bnez	s2, 0x13758 <emdnorm+0x390>
   134f0: 5c 44        	lw	a5, 12(s0)
   134f2: 03 57 84 01  	lhu	a4, 24(s0)
   134f6: 86 07        	slli	a5, a5, 1
   134f8: d2 97        	add	a5, a5, s4
   134fa: 83 d7 07 00  	lhu	a5, 0(a5)
   134fe: f9 8f        	and	a5, a5, a4
   13500: 9d c7        	beqz	a5, 0x1352e <emdnorm+0x166>
   13502: 93 07 24 03  	addi	a5, s0, 50
   13506: 13 06 8a 01  	addi	a2, s4, 24
   1350a: 13 05 2a 00  	addi	a0, s4, 2
   1350e: 81 45        	mv	a1, zero
   13510: 03 d7 07 00  	lhu	a4, 0(a5)
   13514: 83 56 06 00  	lhu	a3, 0(a2)
   13518: 79 16        	addi	a2, a2, -2
   1351a: f9 17        	addi	a5, a5, -2
   1351c: 36 97        	add	a4, a4, a3
   1351e: 2e 97        	add	a4, a4, a1
   13520: 93 55 07 01  	srli	a1, a4, 16
   13524: 23 11 e6 00  	sh	a4, 2(a2)
   13528: 85 89        	andi	a1, a1, 1
   1352a: e3 13 a6 fe  	bne	a2, a0, 0x13510 <emdnorm+0x148>
   1352e: 63 5c 90 1e  	blez	s1, 0x13726 <emdnorm+0x35e>
   13532: 83 57 4a 00  	lhu	a5, 4(s4)
   13536: 63 9d 07 1c  	bnez	a5, 0x13710 <emdnorm+0x348>
   1353a: a1 67        	lui	a5, 8
   1353c: 23 1c 0a 00  	sh	zero, 24(s4)
   13540: f9 17        	addi	a5, a5, -2
   13542: 63 ca 97 0c  	blt	a5, s1, 0x13616 <emdnorm+0x24e>
   13546: 23 11 9a 00  	sh	s1, 2(s4)
   1354a: e2 70        	ld	ra, 56(sp)
   1354c: 42 74        	ld	s0, 48(sp)
   1354e: a2 74        	ld	s1, 40(sp)
   13550: 02 79        	ld	s2, 32(sp)
   13552: e2 69        	ld	s3, 24(sp)
   13554: 42 6a        	ld	s4, 16(sp)
   13556: a2 6a        	ld	s5, 8(sp)
   13558: 21 61        	addi	sp, sp, 64
   1355a: 82 80        	ret
   1355c: 63 cb 04 02  	bltz	s1, 0x13592 <emdnorm+0x1ca>
   13560: e3 8d 0a fc  	beqz	s5, 0x1353a <emdnorm+0x172>
   13564: 83 2a 44 00  	lw	s5, 4(s0)
   13568: 1c 40        	lw	a5, 0(s0)
   1356a: e3 9f fa e8  	bne	s5, a5, 0x13408 <emdnorm+0x40>
   1356e: 63 44 90 26  	bgtz	s1, 0x137d6 <emdnorm+0x40e>
   13572: 13 07 00 09  	addi	a4, zero, 144
   13576: 63 92 e7 26  	bne	a5, a4, 0x137da <emdnorm+0x412>
   1357a: 10 44        	lw	a2, 8(s0)
   1357c: 83 56 44 01  	lhu	a3, 20(s0)
   13580: 93 0a 00 09  	addi	s5, zero, 144
   13584: 06 06        	slli	a2, a2, 1
   13586: 52 96        	add	a2, a2, s4
   13588: 03 57 06 00  	lhu	a4, 0(a2)
   1358c: 33 75 d7 00  	and	a0, a4, a3
   13590: 91 b7        	j	0x134d4 <emdnorm+0x10c>
   13592: 93 07 00 f7  	addi	a5, zero, -144
   13596: 63 de f4 14  	bge	s1, a5, 0x136f2 <emdnorm+0x32a>
   1359a: 57 77 86 c4  	<unknown>
   1359e: d7 77 80 04  	<unknown>
   135a2: 93 07 2a 00  	addi	a5, s4, 2
   135a6: 57 3c 00 5e  	<unknown>
   135aa: b1 46        	addi	a3, zero, 12
   135ac: 57 70 86 c4  	<unknown>
   135b0: 27 dc 07 02  	<unknown>
   135b4: 33 86 e6 40  	sub	a2, a3, a4
   135b8: e3 89 e6 f8  	beq	a3, a4, 0x1354a <emdnorm+0x182>
   135bc: f3 27 20 c2  	csrr	a5, vlenb
   135c0: 89 07        	addi	a5, a5, 2
   135c2: d2 97        	add	a5, a5, s4
   135c4: 57 70 86 04  	<unknown>
   135c8: 27 dc 07 02  	<unknown>
   135cc: bd bf        	j	0x1354a <emdnorm+0x182>
   135ce: e1 47        	addi	a5, zero, 24
   135d0: 63 82 fa 08  	beq	s5, a5, 0x13654 <emdnorm+0x28c>
   135d4: 93 07 50 03  	addi	a5, zero, 53
   135d8: 63 93 fa 18  	bne	s5, a5, 0x1375e <emdnorm+0x396>
   135dc: 85 67        	lui	a5, 1
   135de: 37 07 00 04  	lui	a4, 16384
   135e2: 99 46        	addi	a3, zero, 6
   135e4: 93 87 07 80  	addi	a5, a5, -2048
   135e8: 13 07 f7 7f  	addi	a4, a4, 2047
   135ec: 14 c4        	sw	a3, 8(s0)
   135ee: 58 c8        	sw	a4, 20(s0)
   135f0: 23 1c f4 00  	sh	a5, 24(s0)
   135f4: 54 c4        	sw	a3, 12(s0)
   135f6: 23 13 f4 02  	sh	a5, 38(s0)
   135fa: 23 20 54 01  	sw	s5, 0(s0)
   135fe: 63 5d 90 06  	blez	s1, 0x13678 <emdnorm+0x2b0>
   13602: 03 55 ca 00  	lhu	a0, 12(s4)
   13606: 13 06 ca 00  	addi	a2, s4, 12
   1360a: 9d 47        	addi	a5, zero, 7
   1360c: 13 75 f5 7f  	andi	a0, a0, 2047
   13610: 95 b5        	j	0x13474 <emdnorm+0xac>
   13612: 23 1c 0a 00  	sh	zero, 24(s4)
   13616: 57 77 85 c4  	<unknown>
   1361a: d7 77 80 04  	<unknown>
   1361e: e1 77        	lui	a5, 1048568
   13620: 93 c7 f7 ff  	not	a5, a5
   13624: 23 11 fa 00  	sh	a5, 2(s4)
   13628: 93 07 4a 00  	addi	a5, s4, 4
   1362c: 57 3c 00 5e  	<unknown>
   13630: a9 46        	addi	a3, zero, 10
   13632: 57 70 85 c4  	<unknown>
   13636: 27 dc 07 02  	<unknown>
   1363a: 33 86 e6 40  	sub	a2, a3, a4
   1363e: e3 86 e6 f0  	beq	a3, a4, 0x1354a <emdnorm+0x182>
   13642: f3 27 20 c2  	csrr	a5, vlenb
   13646: 91 07        	addi	a5, a5, 4
   13648: d2 97        	add	a5, a5, s4
   1364a: 57 70 86 04  	<unknown>
   1364e: 27 dc 07 02  	<unknown>
   13652: e5 bd        	j	0x1354a <emdnorm+0x182>
   13654: b7 07 80 00  	lui	a5, 2048
   13658: 91 46        	addi	a3, zero, 4
   1365a: 13 07 00 10  	addi	a4, zero, 256
   1365e: 93 87 f7 0f  	addi	a5, a5, 255
   13662: 14 c4        	sw	a3, 8(s0)
   13664: 5c c8        	sw	a5, 20(s0)
   13666: 23 1c e4 00  	sh	a4, 24(s0)
   1366a: 54 c4        	sw	a3, 12(s0)
   1366c: 23 11 e4 02  	sh	a4, 34(s0)
   13670: 23 20 54 01  	sw	s5, 0(s0)
   13674: 63 46 90 14  	bgtz	s1, 0x137c0 <emdnorm+0x3f8>
   13678: 83 57 8a 01  	lhu	a5, 24(s4)
   1367c: 52 85        	mv	a0, s4
   1367e: 85 8b        	andi	a5, a5, 1
   13680: b3 67 f9 00  	or	a5, s2, a5
   13684: 1b 89 07 00  	sext.w	s2, a5
   13688: ef f0 af f5  	jal	0x12de2 <eshdn1>
   1368c: 1c 44        	lw	a5, 8(s0)
   1368e: 83 56 44 01  	lhu	a3, 20(s0)
   13692: 93 05 f0 08  	addi	a1, zero, 143
   13696: 13 96 17 00  	slli	a2, a5, 1
   1369a: 52 96        	add	a2, a2, s4
   1369c: 03 57 06 00  	lhu	a4, 0(a2)
   136a0: 33 75 d7 00  	and	a0, a4, a3
   136a4: e3 c8 55 e3  	blt	a1, s5, 0x134d4 <emdnorm+0x10c>
   136a8: 85 27        	addiw	a5, a5, 1
   136aa: b1 45        	addi	a1, zero, 12
   136ac: e3 d4 f5 dc  	bge	a1, a5, 0x13474 <emdnorm+0xac>
   136b0: 15 b5        	j	0x134d4 <emdnorm+0x10c>
   136b2: f3 27 20 c2  	csrr	a5, vlenb
   136b6: e9 07        	addi	a5, a5, 26
   136b8: a2 97        	add	a5, a5, s0
   136ba: 57 70 86 04  	<unknown>
   136be: 27 dc 07 02  	<unknown>
   136c2: a5 b3        	j	0x1342a <emdnorm+0x62>
   136c4: 9d 47        	addi	a5, zero, 7
   136c6: 1c c4        	sw	a5, 8(s0)
   136c8: b7 07 01 80  	lui	a5, 524304
   136cc: fd 17        	addi	a5, a5, -1
   136ce: 05 47        	addi	a4, zero, 1
   136d0: 5c c8        	sw	a5, 20(s0)
   136d2: 99 47        	addi	a5, zero, 6
   136d4: 23 1c e4 00  	sh	a4, 24(s0)
   136d8: 5c c4        	sw	a5, 12(s0)
   136da: 23 13 e4 02  	sh	a4, 38(s0)
   136de: 23 20 54 01  	sw	s5, 0(s0)
   136e2: e3 5b 90 f8  	blez	s1, 0x13678 <emdnorm+0x2b0>
   136e6: 03 55 ea 00  	lhu	a0, 14(s4)
   136ea: 13 06 ea 00  	addi	a2, s4, 14
   136ee: a1 47        	addi	a5, zero, 8
   136f0: 51 b3        	j	0x13474 <emdnorm+0xac>
   136f2: 9b 85 04 00  	sext.w	a1, s1
   136f6: 52 85        	mv	a0, s4
   136f8: ef f0 7f 9c  	jal	0x130be <eshift.part.0>
   136fc: 55 c9        	beqz	a0, 0x137b0 <emdnorm+0x3e8>
   136fe: 63 87 0a 0c  	beqz	s5, 0x137cc <emdnorm+0x404>
   13702: 83 2a 44 00  	lw	s5, 4(s0)
   13706: 1c 40        	lw	a5, 0(s0)
   13708: 05 49        	addi	s2, zero, 1
   1370a: e3 9f fa ce  	bne	s5, a5, 0x13408 <emdnorm+0x40>
   1370e: 95 b5        	j	0x13572 <emdnorm+0x1aa>
   13710: 52 85        	mv	a0, s4
   13712: ef f0 0f ed  	jal	0x12de2 <eshdn1>
   13716: a1 67        	lui	a5, 8
   13718: 85 04        	addi	s1, s1, 1
   1371a: 23 1c 0a 00  	sh	zero, 24(s4)
   1371e: f9 17        	addi	a5, a5, -2
   13720: e3 cb 97 ee  	blt	a5, s1, 0x13616 <emdnorm+0x24e>
   13724: 0d b5        	j	0x13546 <emdnorm+0x17e>
   13726: 93 07 00 09  	addi	a5, zero, 144
   1372a: 63 8f fa 00  	beq	s5, a5, 0x13748 <emdnorm+0x380>
   1372e: 52 85        	mv	a0, s4
   13730: ef f0 0f ef  	jal	0x12e20 <eshup1>
   13734: 83 57 4a 00  	lhu	a5, 4(s4)
   13738: 99 eb        	bnez	a5, 0x1374e <emdnorm+0x386>
   1373a: 23 1c 0a 00  	sh	zero, 24(s4)
   1373e: e3 d4 04 e0  	bgez	s1, 0x13546 <emdnorm+0x17e>
   13742: 23 11 0a 00  	sh	zero, 2(s4)
   13746: 11 b5        	j	0x1354a <emdnorm+0x182>
   13748: 83 57 4a 00  	lhu	a5, 4(s4)
   1374c: fd d7        	beqz	a5, 0x1373a <emdnorm+0x372>
   1374e: 52 85        	mv	a0, s4
   13750: ef f0 2f e9  	jal	0x12de2 <eshdn1>
   13754: 85 04        	addi	s1, s1, 1
   13756: d5 b7        	j	0x1373a <emdnorm+0x372>
   13758: e3 85 09 da  	beqz	s3, 0x13502 <emdnorm+0x13a>
   1375c: c9 bb        	j	0x1352e <emdnorm+0x166>
   1375e: b1 47        	addi	a5, zero, 12
   13760: 1c c4        	sw	a5, 8(s0)
   13762: b7 07 01 80  	lui	a5, 524304
   13766: fd 17        	addi	a5, a5, -1
   13768: 05 47        	addi	a4, zero, 1
   1376a: 5c c8        	sw	a5, 20(s0)
   1376c: ad 47        	addi	a5, zero, 11
   1376e: 5c c4        	sw	a5, 12(s0)
   13770: 23 1c e4 00  	sh	a4, 24(s0)
   13774: 23 18 e4 02  	sh	a4, 48(s0)
   13778: 23 20 54 01  	sw	s5, 0(s0)
   1377c: d6 87        	mv	a5, s5
   1377e: c5 bb        	j	0x1356e <emdnorm+0x1a6>
   13780: b7 07 80 00  	lui	a5, 2048
   13784: 99 46        	addi	a3, zero, 6
   13786: 13 07 00 10  	addi	a4, zero, 256
   1378a: 93 87 f7 0f  	addi	a5, a5, 255
   1378e: 14 c4        	sw	a3, 8(s0)
   13790: 5c c8        	sw	a5, 20(s0)
   13792: 23 1c e4 00  	sh	a4, 24(s0)
   13796: 54 c4        	sw	a3, 12(s0)
   13798: 23 13 e4 02  	sh	a4, 38(s0)
   1379c: 23 20 54 01  	sw	s5, 0(s0)
   137a0: e3 5c 90 ec  	blez	s1, 0x13678 <emdnorm+0x2b0>
   137a4: 03 45 ca 00  	lbu	a0, 12(s4)
   137a8: 13 06 ca 00  	addi	a2, s4, 12
   137ac: 9d 47        	addi	a5, zero, 7
   137ae: d9 b1        	j	0x13474 <emdnorm+0xac>
   137b0: 63 8e 0a 00  	beqz	s5, 0x137cc <emdnorm+0x404>
   137b4: 83 2a 44 00  	lw	s5, 4(s0)
   137b8: 1c 40        	lw	a5, 0(s0)
   137ba: e3 97 fa c4  	bne	s5, a5, 0x13408 <emdnorm+0x40>
   137be: 55 bb        	j	0x13572 <emdnorm+0x1aa>
   137c0: 03 45 8a 00  	lbu	a0, 8(s4)
   137c4: 13 06 8a 00  	addi	a2, s4, 8
   137c8: 95 47        	addi	a5, zero, 5
   137ca: 6d b1        	j	0x13474 <emdnorm+0xac>
   137cc: 23 1c 0a 00  	sh	zero, 24(s4)
   137d0: 23 11 0a 00  	sh	zero, 2(s4)
   137d4: 9d bb        	j	0x1354a <emdnorm+0x182>
   137d6: be 8a        	mv	s5, a5
   137d8: 55 bd        	j	0x1368c <emdnorm+0x2c4>
   137da: be 8a        	mv	s5, a5
   137dc: 71 bd        	j	0x13678 <emdnorm+0x2b0>

00000000000137de <eiremain>:
   137de: 1d 71        	addi	sp, sp, -96
   137e0: 4e fc        	sd	s3, 56(sp)
   137e2: 83 59 25 00  	lhu	s3, 2(a0)
   137e6: 86 ec        	sd	ra, 88(sp)
   137e8: a2 e8        	sd	s0, 80(sp)
   137ea: a6 e4        	sd	s1, 72(sp)
   137ec: ca e0        	sd	s2, 64(sp)
   137ee: b2 84        	mv	s1, a2
   137f0: 2e 89        	mv	s2, a1
   137f2: 52 f8        	sd	s4, 48(sp)
   137f4: 5e ec        	sd	s7, 24(sp)
   137f6: 56 f4        	sd	s5, 40(sp)
   137f8: 5a f0        	sd	s6, 32(sp)
   137fa: 62 e8        	sd	s8, 16(sp)
   137fc: 66 e4        	sd	s9, 8(sp)
   137fe: 6a e0        	sd	s10, 0(sp)
   13800: aa 8b        	mv	s7, a0
   13802: ef f0 cf f8  	jal	0x12f8e <enormlz>
   13806: aa 87        	mv	a5, a0
   13808: 4a 85        	mv	a0, s2
   1380a: 03 54 29 00  	lhu	s0, 2(s2)
   1380e: b3 89 f9 40  	sub	s3, s3, a5
   13812: ef f0 cf f7  	jal	0x12f8e <enormlz>
   13816: d7 f7 86 c4  	<unknown>
   1381a: 57 77 80 04  	<unknown>
   1381e: 13 8a 44 03  	addi	s4, s1, 52
   13822: 57 3c 00 5e  	<unknown>
   13826: 35 47        	addi	a4, zero, 13
   13828: 57 f0 86 c4  	<unknown>
   1382c: 27 5c 0a 02  	<unknown>
   13830: 09 8c        	sub	s0, s0, a0
   13832: b3 06 f7 40  	sub	a3, a4, a5
   13836: 63 0b f7 00  	beq	a4, a5, 0x1384c <eiremain+0x6e>
   1383a: f3 27 20 c2  	csrr	a5, vlenb
   1383e: 93 87 47 03  	addi	a5, a5, 52
   13842: a6 97        	add	a5, a5, s1
   13844: 57 f0 86 04  	<unknown>
   13848: 27 dc 07 02  	<unknown>
   1384c: 63 4a 34 07  	blt	s0, s3, 0x138c0 <eiremain+0xe2>
   13850: 13 8b 4b 00  	addi	s6, s7, 4
   13854: 93 0a 49 00  	addi	s5, s2, 4
   13858: 13 8c ab 01  	addi	s8, s7, 26
   1385c: 13 0d 29 00  	addi	s10, s2, 2
   13860: 56 87        	mv	a4, s5
   13862: da 87        	mv	a5, s6
   13864: 03 d6 07 00  	lhu	a2, 0(a5)
   13868: 83 56 07 00  	lhu	a3, 0(a4)
   1386c: 89 07        	addi	a5, a5, 2
   1386e: 09 07        	addi	a4, a4, 2
   13870: 63 1c d6 06  	bne	a2, a3, 0x138e8 <eiremain+0x10a>
   13874: e3 18 fc fe  	bne	s8, a5, 0x13864 <eiremain+0x86>
   13878: 13 86 8b 01  	addi	a2, s7, 24
   1387c: 13 07 89 01  	addi	a4, s2, 24
   13880: 81 46        	mv	a3, zero
   13882: 83 57 07 00  	lhu	a5, 0(a4)
   13886: 83 55 06 00  	lhu	a1, 0(a2)
   1388a: 79 17        	addi	a4, a4, -2
   1388c: 79 16        	addi	a2, a2, -2
   1388e: 8d 8f        	sub	a5, a5, a1
   13890: 95 8f        	sub	a5, a5, a3
   13892: 93 d6 07 01  	srli	a3, a5, 16
   13896: 23 11 f7 00  	sh	a5, 2(a4)
   1389a: 85 8a        	andi	a3, a3, 1
   1389c: e3 13 ed fe  	bne	s10, a4, 0x13882 <eiremain+0xa4>
   138a0: 85 4c        	addi	s9, zero, 1
   138a2: 52 85        	mv	a0, s4
   138a4: ef f0 cf d7  	jal	0x12e20 <eshup1>
   138a8: 83 d7 c4 04  	lhu	a5, 76(s1)
   138ac: 7d 14        	addi	s0, s0, -1
   138ae: 4a 85        	mv	a0, s2
   138b0: b3 e7 fc 00  	or	a5, s9, a5
   138b4: 23 96 f4 04  	sh	a5, 76(s1)
   138b8: ef f0 8f d6  	jal	0x12e20 <eshup1>
   138bc: e3 52 34 fb  	bge	s0, s3, 0x13860 <eiremain+0x82>
   138c0: a2 86        	mv	a3, s0
   138c2: 46 64        	ld	s0, 80(sp)
   138c4: e6 60        	ld	ra, 88(sp)
   138c6: e2 79        	ld	s3, 56(sp)
   138c8: 42 7a        	ld	s4, 48(sp)
   138ca: a2 7a        	ld	s5, 40(sp)
   138cc: 02 7b        	ld	s6, 32(sp)
   138ce: e2 6b        	ld	s7, 24(sp)
   138d0: 42 6c        	ld	s8, 16(sp)
   138d2: a2 6c        	ld	s9, 8(sp)
   138d4: 02 6d        	ld	s10, 0(sp)
   138d6: a6 87        	mv	a5, s1
   138d8: 4a 85        	mv	a0, s2
   138da: a6 64        	ld	s1, 72(sp)
   138dc: 06 69        	ld	s2, 64(sp)
   138de: 01 47        	mv	a4, zero
   138e0: 01 46        	mv	a2, zero
   138e2: 81 45        	mv	a1, zero
   138e4: 25 61        	addi	sp, sp, 96
   138e6: cd b4        	j	0x133c8 <emdnorm>
   138e8: 81 4c        	mv	s9, zero
   138ea: e3 ec c6 fa  	bltu	a3, a2, 0x138a2 <eiremain+0xc4>
   138ee: 69 b7        	j	0x13878 <eiremain+0x9a>

00000000000138f0 <e113toe.isra.0>:
   138f0: 39 71        	addi	sp, sp, -64
   138f2: d7 f7 86 c4  	<unknown>
   138f6: 57 77 80 04  	<unknown>
   138fa: 22 f8        	sd	s0, 48(sp)
   138fc: 26 f4        	sd	s1, 40(sp)
   138fe: 06 fc        	sd	ra, 56(sp)
   13900: 0a 84        	mv	s0, sp
   13902: 57 3c 00 5e  	<unknown>
   13906: 35 47        	addi	a4, zero, 13
   13908: 57 f0 86 c4  	<unknown>
   1390c: 27 5c 04 02  	<unknown>
   13910: ae 84        	mv	s1, a1
   13912: b3 06 f7 40  	sub	a3, a4, a5
   13916: 63 09 f7 00  	beq	a4, a5, 0x13928 <e113toe.isra.0+0x38>
   1391a: f3 27 20 c2  	csrr	a5, vlenb
   1391e: 8a 97        	add	a5, a5, sp
   13920: 57 f0 86 04  	<unknown>
   13924: 27 dc 07 02  	<unknown>
   13928: 83 57 e5 00  	lhu	a5, 14(a0)
   1392c: 21 67        	lui	a4, 8
   1392e: 7d 17        	addi	a4, a4, -1
   13930: 93 96 07 03  	slli	a3, a5, 48
   13934: fd 96        	srai	a3, a3, 63
   13936: 23 10 d1 00  	sh	a3, 0(sp)
   1393a: b3 f6 e7 00  	and	a3, a5, a4
   1393e: 36 86        	mv	a2, a3
   13940: 63 84 e6 04  	beq	a3, a4, 0x13988 <e113toe.isra.0+0x98>
   13944: 93 07 e5 00  	addi	a5, a0, 14
   13948: 23 11 d1 00  	sh	a3, 2(sp)
   1394c: 13 07 61 00  	addi	a4, sp, 6
   13950: 83 d6 e7 ff  	lhu	a3, -2(a5)
   13954: f9 17        	addi	a5, a5, -2
   13956: 09 07        	addi	a4, a4, 2
   13958: 23 1f d7 fe  	sh	a3, -2(a4)
   1395c: e3 1a f5 fe  	bne	a0, a5, 0x13950 <e113toe.isra.0+0x60>
   13960: 01 ee        	bnez	a2, 0x13978 <e113toe.isra.0+0x88>
   13962: 23 12 01 00  	sh	zero, 4(sp)
   13966: a6 85        	mv	a1, s1
   13968: 22 85        	mv	a0, s0
   1396a: ef f0 cf d7  	jal	0x12ee6 <emovo.constprop.0>
   1396e: e2 70        	ld	ra, 56(sp)
   13970: 42 74        	ld	s0, 48(sp)
   13972: a2 74        	ld	s1, 40(sp)
   13974: 21 61        	addi	sp, sp, 64
   13976: 82 80        	ret
   13978: 85 47        	addi	a5, zero, 1
   1397a: fd 55        	addi	a1, zero, -1
   1397c: 22 85        	mv	a0, s0
   1397e: 23 12 f1 00  	sh	a5, 4(sp)
   13982: ef f0 cf f3  	jal	0x130be <eshift.part.0>
   13986: c5 b7        	j	0x13966 <e113toe.isra.0+0x76>
   13988: aa 87        	mv	a5, a0
   1398a: 93 06 e5 00  	addi	a3, a0, 14
   1398e: 03 d7 07 00  	lhu	a4, 0(a5)
   13992: 89 07        	addi	a5, a5, 2
   13994: 59 e7        	bnez	a4, 0x13a22 <e113toe.isra.0+0x132>
   13996: e3 9c f6 fe  	bne	a3, a5, 0x1398e <e113toe.isra.0+0x9e>
   1399a: 57 77 85 c4  	<unknown>
   1399e: d7 77 80 04  	<unknown>
   139a2: a9 46        	addi	a3, zero, 10
   139a4: 57 3c 00 5e  	<unknown>
   139a8: f3 27 20 c2  	csrr	a5, vlenb
   139ac: 57 70 85 c4  	<unknown>
   139b0: 27 dc 04 02  	<unknown>
   139b4: a6 97        	add	a5, a5, s1
   139b6: 33 86 e6 40  	sub	a2, a3, a4
   139ba: 63 86 e6 00  	beq	a3, a4, 0x139c6 <e113toe.isra.0+0xd6>
   139be: 57 70 86 04  	<unknown>
   139c2: 27 dc 07 02  	<unknown>
   139c6: 57 f7 84 c4  	<unknown>
   139ca: d7 76 80 04  	<unknown>
   139ce: a5 46        	addi	a3, zero, 9
   139d0: 57 3c 00 5e  	<unknown>
   139d4: 57 f0 84 c4  	<unknown>
   139d8: 27 dc 04 02  	<unknown>
   139dc: 33 86 e6 40  	sub	a2, a3, a4
   139e0: 63 86 e6 00  	beq	a3, a4, 0x139ec <e113toe.isra.0+0xfc>
   139e4: 57 70 86 04  	<unknown>
   139e8: 27 dc 07 02  	<unknown>
   139ec: 83 d7 24 01  	lhu	a5, 18(s1)
   139f0: 21 67        	lui	a4, 8
   139f2: 7d 17        	addi	a4, a4, -1
   139f4: d9 8f        	or	a5, a5, a4
   139f6: c2 17        	slli	a5, a5, 48
   139f8: c1 93        	srli	a5, a5, 48
   139fa: 23 99 f4 00  	sh	a5, 18(s1)
   139fe: 03 17 e5 00  	lh	a4, 14(a0)
   13a02: e3 56 07 f6  	bgez	a4, 0x1396e <e113toe.isra.0+0x7e>
   13a06: 13 86 24 01  	addi	a2, s1, 18
   13a0a: 26 87        	mv	a4, s1
   13a0c: 83 56 07 00  	lhu	a3, 0(a4)
   13a10: 09 07        	addi	a4, a4, 2
   13a12: b1 fe        	bnez	a3, 0x1396e <e113toe.isra.0+0x7e>
   13a14: e3 1c e6 fe  	bne	a2, a4, 0x13a0c <e113toe.isra.0+0x11c>
   13a18: 61 77        	lui	a4, 1048568
   13a1a: b9 8f        	xor	a5, a5, a4
   13a1c: 23 99 f4 00  	sh	a5, 18(s1)
   13a20: b9 b7        	j	0x1396e <e113toe.isra.0+0x7e>
   13a22: d7 77 80 04  	<unknown>
   13a26: f1 77        	lui	a5, 1048572
   13a28: 57 3c 00 5e  	<unknown>
   13a2c: 57 70 84 c4  	<unknown>
   13a30: 27 dc 04 02  	<unknown>
   13a34: 23 98 f4 00  	sh	a5, 16(s1)
   13a38: e1 77        	lui	a5, 1048568
   13a3a: 93 c7 f7 ff  	not	a5, a5
   13a3e: 23 99 f4 00  	sh	a5, 18(s1)
   13a42: 35 b7        	j	0x1396e <e113toe.isra.0+0x7e>

0000000000013a44 <ediv>:
   13a44: 83 57 25 01  	lhu	a5, 18(a0)
   13a48: 6d 71        	addi	sp, sp, -272
   13a4a: 22 e2        	sd	s0, 256(sp)
   13a4c: 93 c7 f7 ff  	not	a5, a5
   13a50: a6 fd        	sd	s1, 248(sp)
   13a52: ca f9        	sd	s2, 240(sp)
   13a54: 6e f5        	sd	s11, 168(sp)
   13a56: 06 e6        	sd	ra, 264(sp)
   13a58: ce f5        	sd	s3, 232(sp)
   13a5a: d2 f1        	sd	s4, 224(sp)
   13a5c: d6 ed        	sd	s5, 216(sp)
   13a5e: da e9        	sd	s6, 208(sp)
   13a60: de e5        	sd	s7, 200(sp)
   13a62: e2 e1        	sd	s8, 192(sp)
   13a64: 66 fd        	sd	s9, 184(sp)
   13a66: 6a f9        	sd	s10, 176(sp)
   13a68: 13 97 17 03  	slli	a4, a5, 49
   13a6c: aa 84        	mv	s1, a0
   13a6e: 2e 89        	mv	s2, a1
   13a70: b2 8d        	mv	s11, a2
   13a72: 36 84        	mv	s0, a3
   13a74: 11 eb        	bnez	a4, 0x13a88 <ediv+0x44>
   13a76: 93 06 25 01  	addi	a3, a0, 18
   13a7a: aa 87        	mv	a5, a0
   13a7c: 03 d7 07 00  	lhu	a4, 0(a5)
   13a80: 89 07        	addi	a5, a5, 2
   13a82: 4d ef        	bnez	a4, 0x13b3c <ediv+0xf8>
   13a84: e3 9c d7 fe  	bne	a5, a3, 0x13a7c <ediv+0x38>
   13a88: 83 57 29 01  	lhu	a5, 18(s2)
   13a8c: 93 c7 f7 ff  	not	a5, a5
   13a90: 13 97 17 03  	slli	a4, a5, 49
   13a94: 11 eb        	bnez	a4, 0x13aa8 <ediv+0x64>
   13a96: 93 06 29 01  	addi	a3, s2, 18
   13a9a: ca 87        	mv	a5, s2
   13a9c: 03 d7 07 00  	lhu	a4, 0(a5)
   13aa0: 89 07        	addi	a5, a5, 2
   13aa2: 45 eb        	bnez	a4, 0x13b52 <ediv+0x10e>
   13aa4: e3 9c d7 fe  	bne	a5, a3, 0x13a9c <ediv+0x58>
   13aa8: f9 69        	lui	s3, 30
   13aaa: 93 85 09 cf  	addi	a1, s3, -784
   13aae: 26 85        	mv	a0, s1
   13ab0: ef f0 1f 83  	jal	0x132e0 <ecmp>
   13ab4: 63 05 05 16  	beqz	a0, 0x13c1e <ediv+0x1da>
   13ab8: 03 d6 24 01  	lhu	a2, 18(s1)
   13abc: a1 67        	lui	a5, 8
   13abe: fd 17        	addi	a5, a5, -1
   13ac0: 33 75 f6 00  	and	a0, a2, a5
   13ac4: 63 12 f5 0a  	bne	a0, a5, 0x13b68 <ediv+0x124>
   13ac8: 93 86 24 01  	addi	a3, s1, 18
   13acc: a6 87        	mv	a5, s1
   13ace: 03 d7 07 00  	lhu	a4, 0(a5)
   13ad2: 89 07        	addi	a5, a5, 2
   13ad4: 51 eb        	bnez	a4, 0x13b68 <ediv+0x124>
   13ad6: e3 9c f6 fe  	bne	a3, a5, 0x13ace <ediv+0x8a>
   13ada: 03 57 29 01  	lhu	a4, 18(s2)
   13ade: 93 47 f7 ff  	not	a5, a4
   13ae2: 93 96 17 03  	slli	a3, a5, 49
   13ae6: 63 95 06 10  	bnez	a3, 0x13bf0 <ediv+0x1ac>
   13aea: 93 06 29 01  	addi	a3, s2, 18
   13aee: ca 87        	mv	a5, s2
   13af0: 83 d5 07 00  	lhu	a1, 0(a5)
   13af4: 89 07        	addi	a5, a5, 2
   13af6: 63 94 05 44  	bnez	a1, 0x13f3e <ediv+0x4fa>
   13afa: e3 9b d7 fe  	bne	a5, a3, 0x13af0 <ediv+0xac>
   13afe: d7 77 80 04  	<unknown>
   13b02: f1 77        	lui	a5, 1048572
   13b04: 57 3c 00 5e  	<unknown>
   13b08: 57 70 84 c4  	<unknown>
   13b0c: 27 dc 0d 02  	<unknown>
   13b10: 23 98 fd 00  	sh	a5, 16(s11)
   13b14: e1 77        	lui	a5, 1048568
   13b16: 93 c7 f7 ff  	not	a5, a5
   13b1a: 23 99 fd 00  	sh	a5, 18(s11)
   13b1e: b2 60        	ld	ra, 264(sp)
   13b20: 12 64        	ld	s0, 256(sp)
   13b22: ee 74        	ld	s1, 248(sp)
   13b24: 4e 79        	ld	s2, 240(sp)
   13b26: ae 79        	ld	s3, 232(sp)
   13b28: 0e 7a        	ld	s4, 224(sp)
   13b2a: ee 6a        	ld	s5, 216(sp)
   13b2c: 4e 6b        	ld	s6, 208(sp)
   13b2e: ae 6b        	ld	s7, 200(sp)
   13b30: 0e 6c        	ld	s8, 192(sp)
   13b32: ea 7c        	ld	s9, 184(sp)
   13b34: 4a 7d        	ld	s10, 176(sp)
   13b36: aa 7d        	ld	s11, 168(sp)
   13b38: 51 61        	addi	sp, sp, 272
   13b3a: 82 80        	ret
   13b3c: 13 87 44 01  	addi	a4, s1, 20
   13b40: 83 d7 04 00  	lhu	a5, 0(s1)
   13b44: 89 04        	addi	s1, s1, 2
   13b46: 89 0d        	addi	s11, s11, 2
   13b48: 23 9f fd fe  	sh	a5, -2(s11)
   13b4c: e3 9a e4 fe  	bne	s1, a4, 0x13b40 <ediv+0xfc>
   13b50: f9 b7        	j	0x13b1e <ediv+0xda>
   13b52: 13 87 4d 01  	addi	a4, s11, 20
   13b56: 83 57 09 00  	lhu	a5, 0(s2)
   13b5a: 89 0d        	addi	s11, s11, 2
   13b5c: 09 09        	addi	s2, s2, 2
   13b5e: 23 9f fd fe  	sh	a5, -2(s11)
   13b62: e3 9a ed fe  	bne	s11, a4, 0x13b56 <ediv+0x112>
   13b66: 65 bf        	j	0x13b1e <ediv+0xda>
   13b68: 03 57 29 01  	lhu	a4, 18(s2)
   13b6c: 93 47 f7 ff  	not	a5, a4
   13b70: 93 96 17 03  	slli	a3, a5, 49
   13b74: cd ee        	bnez	a3, 0x13c2e <ediv+0x1ea>
   13b76: 81 48        	mv	a7, zero
   13b78: 93 06 29 01  	addi	a3, s2, 18
   13b7c: ca 87        	mv	a5, s2
   13b7e: 83 d5 07 00  	lhu	a1, 0(a5)
   13b82: 89 07        	addi	a5, a5, 2
   13b84: a5 e5        	bnez	a1, 0x13bec <ediv+0x1a8>
   13b86: e3 9c d7 fe  	bne	a5, a3, 0x13b7e <ediv+0x13a>
   13b8a: a1 67        	lui	a5, 8
   13b8c: fd 17        	addi	a5, a5, -1
   13b8e: 63 09 f5 0c  	beq	a0, a5, 0x13c60 <ediv+0x21c>
   13b92: 3d 82        	srli	a2, a2, 15
   13b94: 83 57 09 00  	lhu	a5, 0(s2)
   13b98: 09 09        	addi	s2, s2, 2
   13b9a: 63 9e 07 38  	bnez	a5, 0x13f36 <ediv+0x4f2>
   13b9e: e3 1b d9 fe  	bne	s2, a3, 0x13b94 <ediv+0x150>
   13ba2: 93 57 f7 00  	srli	a5, a4, 15
   13ba6: b1 8f        	xor	a5, a5, a2
   13ba8: 57 f7 84 c4  	<unknown>
   13bac: 9b 97 f7 00  	slliw	a5, a5, 15
   13bb0: d7 76 80 04  	<unknown>
   13bb4: 23 99 fd 00  	sh	a5, 18(s11)
   13bb8: 57 3c 00 5e  	<unknown>
   13bbc: a5 47        	addi	a5, zero, 9
   13bbe: 57 f0 84 c4  	<unknown>
   13bc2: 27 dc 0d 02  	<unknown>
   13bc6: b3 86 e7 40  	sub	a3, a5, a4
   13bca: 63 89 e7 00  	beq	a5, a4, 0x13bdc <ediv+0x198>
   13bce: f3 27 20 c2  	csrr	a5, vlenb
   13bd2: ee 97        	add	a5, a5, s11
   13bd4: 57 f0 86 04  	<unknown>
   13bd8: 27 dc 07 02  	<unknown>
   13bdc: 83 d7 2d 01  	lhu	a5, 18(s11)
   13be0: 21 67        	lui	a4, 8
   13be2: 7d 17        	addi	a4, a4, -1
   13be4: d9 8f        	or	a5, a5, a4
   13be6: 23 99 fd 00  	sh	a5, 18(s11)
   13bea: 15 bf        	j	0x13b1e <ediv+0xda>
   13bec: 63 81 08 04  	beqz	a7, 0x13c2e <ediv+0x1ea>
   13bf0: d7 77 85 c4  	<unknown>
   13bf4: 57 77 80 04  	<unknown>
   13bf8: 29 47        	addi	a4, zero, 10
   13bfa: 57 3c 00 5e  	<unknown>
   13bfe: 57 70 85 c4  	<unknown>
   13c02: 27 dc 0d 02  	<unknown>
   13c06: b3 06 f7 40  	sub	a3, a4, a5
   13c0a: e3 0a f7 f0  	beq	a4, a5, 0x13b1e <ediv+0xda>
   13c0e: f3 27 20 c2  	csrr	a5, vlenb
   13c12: ee 97        	add	a5, a5, s11
   13c14: 57 f0 86 04  	<unknown>
   13c18: 27 dc 07 02  	<unknown>
   13c1c: 09 b7        	j	0x13b1e <ediv+0xda>
   13c1e: 93 85 09 cf  	addi	a1, s3, -784
   13c22: 4a 85        	mv	a0, s2
   13c24: ef f0 cf eb  	jal	0x132e0 <ecmp>
   13c28: e3 18 05 e8  	bnez	a0, 0x13ab8 <ediv+0x74>
   13c2c: c9 bd        	j	0x13afe <ediv+0xba>
   13c2e: 8c 00        	addi	a1, sp, 64
   13c30: 26 85        	mv	a0, s1
   13c32: ef f0 6f df  	jal	0x13228 <emovi>
   13c36: 4a 85        	mv	a0, s2
   13c38: 8c 10        	addi	a1, sp, 96
   13c3a: ef f0 ef de  	jal	0x13228 <emovi>
   13c3e: 03 5c 21 06  	lhu	s8, 98(sp)
   13c42: 03 59 21 04  	lhu	s2, 66(sp)
   13c46: 63 17 0c 02  	bnez	s8, 0x13c74 <ediv+0x230>
   13c4a: 93 07 21 06  	addi	a5, sp, 98
   13c4e: a4 18        	addi	s1, sp, 120
   13c50: 03 d7 07 00  	lhu	a4, 0(a5)
   13c54: 89 07        	addi	a5, a5, 2
   13c56: 63 12 07 2c  	bnez	a4, 0x13f1a <ediv+0x4d6>
   13c5a: e3 9b f4 fe  	bne	s1, a5, 0x13c50 <ediv+0x20c>
   13c5e: 49 bf        	j	0x13bf0 <ediv+0x1ac>
   13c60: 93 85 24 01  	addi	a1, s1, 18
   13c64: 83 d7 04 00  	lhu	a5, 0(s1)
   13c68: 89 04        	addi	s1, s1, 2
   13c6a: 63 98 07 2c  	bnez	a5, 0x13f3a <ediv+0x4f6>
   13c6e: e3 9b 95 fe  	bne	a1, s1, 0x13c64 <ediv+0x220>
   13c72: 05 b7        	j	0x13b92 <ediv+0x14e>
   13c74: 62 f0        	sd	s8, 32(sp)
   13c76: 63 1a 09 02  	bnez	s2, 0x13caa <ediv+0x266>
   13c7a: 93 07 21 04  	addi	a5, sp, 66
   13c7e: b4 08        	addi	a3, sp, 88
   13c80: 03 d7 07 00  	lhu	a4, 0(a5)
   13c84: 89 07        	addi	a5, a5, 2
   13c86: 63 11 07 2a  	bnez	a4, 0x13f28 <ediv+0x4e4>
   13c8a: e3 9b f6 fe  	bne	a3, a5, 0x13c80 <ediv+0x23c>
   13c8e: 83 57 01 04  	lhu	a5, 64(sp)
   13c92: 83 56 01 06  	lhu	a3, 96(sp)
   13c96: 57 f7 84 c4  	<unknown>
   13c9a: 57 76 80 04  	<unknown>
   13c9e: 95 8f        	sub	a5, a5, a3
   13ca0: b3 37 f0 00  	snez	a5, a5
   13ca4: 9b 97 f7 00  	slliw	a5, a5, 15
   13ca8: 31 b7        	j	0x13bb4 <ediv+0x170>
   13caa: 4a f4        	sd	s2, 40(sp)
   13cac: 06 57        	lw	a4, 96(sp)
   13cae: d7 f7 85 c4  	<unknown>
   13cb2: d7 76 80 04  	<unknown>
   13cb6: 58 d8        	sw	a4, 52(s0)
   13cb8: 93 0b 84 03  	addi	s7, s0, 56
   13cbc: 93 06 44 03  	addi	a3, s0, 52
   13cc0: 57 3c 00 5e  	<unknown>
   13cc4: 2d 47        	addi	a4, zero, 11
   13cc6: 57 f0 85 c4  	<unknown>
   13cca: 27 dc 0b 02  	<unknown>
   13cce: 36 f8        	sd	a3, 48(sp)
   13cd0: b3 06 f7 40  	sub	a3, a4, a5
   13cd4: 36 fc        	sd	a3, 56(sp)
   13cd6: 63 0b f7 00  	beq	a4, a5, 0x13cec <ediv+0x2a8>
   13cda: f3 27 20 c2  	csrr	a5, vlenb
   13cde: 93 87 87 03  	addi	a5, a5, 56
   13ce2: a2 97        	add	a5, a5, s0
   13ce4: 57 f0 86 04  	<unknown>
   13ce8: 27 dc 07 02  	<unknown>
   13cec: 88 10        	addi	a0, sp, 96
   13cee: ef f0 4f 8f  	jal	0x12de2 <eshdn1>
   13cf2: 83 56 61 04  	lhu	a3, 70(sp)
   13cf6: f3 27 20 c2  	csrr	a5, vlenb
   13cfa: 57 77 85 c4  	<unknown>
   13cfe: 93 99 06 01  	slli	s3, a3, 16
   13d02: 29 4f        	addi	t5, zero, 10
   13d04: 13 8a 47 00  	addi	s4, a5, 4
   13d08: 36 e8        	sd	a3, 16(sp)
   13d0a: 99 07        	addi	a5, a5, 6
   13d0c: b3 89 d9 40  	sub	s3, s3, a3
   13d10: b3 0c ef 40  	sub	s9, t5, a4
   13d14: 94 10        	addi	a3, sp, 96
   13d16: 98 10        	addi	a4, sp, 96
   13d18: d2 96        	add	a3, a3, s4
   13d1a: 41 69        	lui	s2, 16
   13d1c: ba 97        	add	a5, a5, a4
   13d1e: 36 e4        	sd	a3, 8(sp)
   13d20: 13 0b e4 04  	addi	s6, s0, 78
   13d24: 13 0d 21 06  	addi	s10, sp, 98
   13d28: a4 18        	addi	s1, sp, 120
   13d2a: 13 0c 41 06  	addi	s8, sp, 100
   13d2e: 7d 19        	addi	s2, s2, -1
   13d30: 13 0a a1 09  	addi	s4, sp, 154
   13d34: 3e ec        	sd	a5, 24(sp)
   13d36: 83 57 41 06  	lhu	a5, 100(sp)
   13d3a: 03 57 61 06  	lhu	a4, 102(sp)
   13d3e: ca 8a        	mv	s5, s2
   13d40: c2 07        	slli	a5, a5, 16
   13d42: ba 97        	add	a5, a5, a4
   13d44: 63 e9 f9 00  	bltu	s3, a5, 0x13d56 <ediv+0x312>
   13d48: 42 67        	ld	a4, 16(sp)
   13d4a: b3 d7 e7 02  	divu	a5, a5, a4
   13d4e: 93 9a 07 03  	slli	s5, a5, 48
   13d52: 93 da 0a 03  	srli	s5, s5, 48
   13d56: 10 01        	addi	a2, sp, 128
   13d58: 8c 00        	addi	a1, sp, 64
   13d5a: 56 85        	mv	a0, s5
   13d5c: ef f0 0f 90  	jal	0x12e5c <m16m>
   13d60: 62 87        	mv	a4, s8
   13d62: 5c 01        	addi	a5, sp, 132
   13d64: 03 d6 07 00  	lhu	a2, 0(a5)
   13d68: 83 56 07 00  	lhu	a3, 0(a4)
   13d6c: 89 07        	addi	a5, a5, 2
   13d6e: 09 07        	addi	a4, a4, 2
   13d70: 63 1f d6 10  	bne	a2, a3, 0x13e8e <ediv+0x44a>
   13d74: e3 98 47 ff  	bne	a5, s4, 0x13d64 <ediv+0x320>
   13d78: 81 46        	mv	a3, zero
   13d7a: 30 09        	addi	a2, sp, 152
   13d7c: 26 87        	mv	a4, s1
   13d7e: 83 57 07 00  	lhu	a5, 0(a4)
   13d82: 83 55 06 00  	lhu	a1, 0(a2)
   13d86: 79 17        	addi	a4, a4, -2
   13d88: 79 16        	addi	a2, a2, -2
   13d8a: 8d 8f        	sub	a5, a5, a1
   13d8c: 95 8f        	sub	a5, a5, a3
   13d8e: 93 d6 07 01  	srli	a3, a5, 16
   13d92: 23 11 f7 00  	sh	a5, 2(a4)
   13d96: 85 8a        	andi	a3, a3, 1
   13d98: e3 13 a7 ff  	bne	a4, s10, 0x13d7e <ediv+0x33a>
   13d9c: 23 90 5b 01  	sh	s5, 0(s7)
   13da0: 93 07 61 06  	addi	a5, sp, 102
   13da4: 57 70 85 c4  	<unknown>
   13da8: 07 dc 07 02  	<unknown>
   13dac: 27 5c 0c 02  	<unknown>
   13db0: 63 8a 0c 00  	beqz	s9, 0x13dc4 <ediv+0x380>
   13db4: e2 67        	ld	a5, 24(sp)
   13db6: 57 f0 8c 04  	<unknown>
   13dba: 07 dc 07 02  	<unknown>
   13dbe: a2 67        	ld	a5, 8(sp)
   13dc0: 27 dc 07 02  	<unknown>
   13dc4: 23 1c 01 06  	sh	zero, 120(sp)
   13dc8: 89 0b        	addi	s7, s7, 2
   13dca: e3 16 7b f7  	bne	s6, s7, 0x13d36 <ediv+0x2f2>
   13dce: 62 77        	ld	a4, 56(sp)
   13dd0: 57 f0 85 c4  	<unknown>
   13dd4: 87 5c 0c 02  	<unknown>
   13dd8: d7 77 80 04  	<unknown>
   13ddc: 57 3c 00 5e  	<unknown>
   13de0: 57 f0 85 c0  	<unknown>
   13de4: 57 0c 9c 2b  	<unknown>
   13de8: 09 cf        	beqz	a4, 0x13e02 <ediv+0x3be>
   13dea: f3 27 20 c2  	csrr	a5, vlenb
   13dee: 93 87 47 02  	addi	a5, a5, 36
   13df2: 94 00        	addi	a3, sp, 64
   13df4: b6 97        	add	a5, a5, a3
   13df6: 57 70 87 00  	<unknown>
   13dfa: 87 dc 07 02  	<unknown>
   13dfe: 57 0c 9c 2b  	<unknown>
   13e02: c2 76        	ld	a3, 48(sp)
   13e04: d7 77 80 04  	<unknown>
   13e08: d7 6c 00 42  	<unknown>
   13e0c: d7 ac 8c 0b  	<unknown>
   13e10: d7 25 90 43  	<unknown>
   13e14: d7 f7 86 c4  	<unknown>
   13e18: c2 15        	slli	a1, a1, 48
   13e1a: 07 dc 06 02  	<unknown>
   13e1e: 94 10        	addi	a3, sp, 96
   13e20: 35 47        	addi	a4, zero, 13
   13e22: c1 91        	srli	a1, a1, 48
   13e24: 27 dc 06 02  	<unknown>
   13e28: b3 35 b0 00  	snez	a1, a1
   13e2c: b3 06 f7 40  	sub	a3, a4, a5
   13e30: 63 01 f7 02  	beq	a4, a5, 0x13e52 <ediv+0x40e>
   13e34: f3 27 20 c2  	csrr	a5, vlenb
   13e38: 13 87 47 03  	addi	a4, a5, 52
   13e3c: 90 00        	addi	a2, sp, 64
   13e3e: 93 87 07 02  	addi	a5, a5, 32
   13e42: 22 97        	add	a4, a4, s0
   13e44: b2 97        	add	a5, a5, a2
   13e46: 57 f0 86 04  	<unknown>
   13e4a: 07 5c 07 02  	<unknown>
   13e4e: 27 dc 07 02  	<unknown>
   13e52: 82 77        	ld	a5, 32(sp)
   13e54: 22 77        	ld	a4, 40(sp)
   13e56: 91 66        	lui	a3, 4
   13e58: fd 16        	addi	a3, a3, -1
   13e5a: 33 8c e7 40  	sub	s8, a5, a4
   13e5e: 88 10        	addi	a0, sp, 96
   13e60: a2 87        	mv	a5, s0
   13e62: 13 07 00 04  	addi	a4, zero, 64
   13e66: e2 96        	add	a3, a3, s8
   13e68: 01 46        	mv	a2, zero
   13e6a: ef f0 ef d5  	jal	0x133c8 <emdnorm>
   13e6e: 03 57 01 06  	lhu	a4, 96(sp)
   13e72: 83 57 01 04  	lhu	a5, 64(sp)
   13e76: ee 85        	mv	a1, s11
   13e78: 88 10        	addi	a0, sp, 96
   13e7a: 99 8f        	sub	a5, a5, a4
   13e7c: b3 37 f0 00  	snez	a5, a5
   13e80: bb 07 f0 40  	negw	a5, a5
   13e84: 23 10 f1 06  	sh	a5, 96(sp)
   13e88: ef f0 ef 85  	jal	0x12ee6 <emovo.constprop.0>
   13e8c: 49 b9        	j	0x13b1e <ediv+0xda>
   13e8e: e3 f5 c6 ee  	bgeu	a3, a2, 0x13d78 <ediv+0x334>
   13e92: 9b 8f fa ff  	addiw	t6, s5, -1
   13e96: c2 1f        	slli	t6, t6, 48
   13e98: 93 df 0f 03  	srli	t6, t6, 48
   13e9c: 81 46        	mv	a3, zero
   13e9e: b0 08        	addi	a2, sp, 88
   13ea0: 38 09        	addi	a4, sp, 152
   13ea2: 93 05 21 08  	addi	a1, sp, 130
   13ea6: 83 57 07 00  	lhu	a5, 0(a4)
   13eaa: 83 52 06 00  	lhu	t0, 0(a2)
   13eae: 79 17        	addi	a4, a4, -2
   13eb0: 79 16        	addi	a2, a2, -2
   13eb2: b3 87 57 40  	sub	a5, a5, t0
   13eb6: 95 8f        	sub	a5, a5, a3
   13eb8: 93 d6 07 01  	srli	a3, a5, 16
   13ebc: 23 11 f7 00  	sh	a5, 2(a4)
   13ec0: 85 8a        	andi	a3, a3, 1
   13ec2: e3 12 b7 fe  	bne	a4, a1, 0x13ea6 <ediv+0x462>
   13ec6: 62 87        	mv	a4, s8
   13ec8: 5c 01        	addi	a5, sp, 132
   13eca: 03 d6 07 00  	lhu	a2, 0(a5)
   13ece: 83 56 07 00  	lhu	a3, 0(a4)
   13ed2: 89 07        	addi	a5, a5, 2
   13ed4: 09 07        	addi	a4, a4, 2
   13ed6: 63 16 d6 00  	bne	a2, a3, 0x13ee2 <ediv+0x49e>
   13eda: e3 98 47 ff  	bne	a5, s4, 0x13eca <ediv+0x486>
   13ede: fe 8a        	mv	s5, t6
   13ee0: 61 bd        	j	0x13d78 <ediv+0x334>
   13ee2: e3 fe c6 fe  	bgeu	a3, a2, 0x13ede <ediv+0x49a>
   13ee6: 1b 85 ea ff  	addiw	a0, s5, -2
   13eea: 93 1a 05 03  	slli	s5, a0, 48
   13eee: 93 da 0a 03  	srli	s5, s5, 48
   13ef2: 81 46        	mv	a3, zero
   13ef4: b0 08        	addi	a2, sp, 88
   13ef6: 38 09        	addi	a4, sp, 152
   13ef8: 83 57 07 00  	lhu	a5, 0(a4)
   13efc: 83 5f 06 00  	lhu	t6, 0(a2)
   13f00: 79 17        	addi	a4, a4, -2
   13f02: 79 16        	addi	a2, a2, -2
   13f04: b3 87 f7 41  	sub	a5, a5, t6
   13f08: 95 8f        	sub	a5, a5, a3
   13f0a: 93 d6 07 01  	srli	a3, a5, 16
   13f0e: 23 11 f7 00  	sh	a5, 2(a4)
   13f12: 85 8a        	andi	a3, a3, 1
   13f14: e3 12 b7 fe  	bne	a4, a1, 0x13ef8 <ediv+0x4b4>
   13f18: 85 b5        	j	0x13d78 <ediv+0x334>
   13f1a: 88 10        	addi	a0, sp, 96
   13f1c: ef f0 2f 87  	jal	0x12f8e <enormlz>
   13f20: b3 07 a0 40  	neg	a5, a0
   13f24: 3e f0        	sd	a5, 32(sp)
   13f26: 81 bb        	j	0x13c76 <ediv+0x232>
   13f28: 88 00        	addi	a0, sp, 64
   13f2a: ef f0 4f 86  	jal	0x12f8e <enormlz>
   13f2e: b3 07 a0 40  	neg	a5, a0
   13f32: 3e f4        	sd	a5, 40(sp)
   13f34: a5 bb        	j	0x13cac <ediv+0x268>
   13f36: 81 47        	mv	a5, zero
   13f38: bd b1        	j	0x13ba6 <ediv+0x162>
   13f3a: 01 46        	mv	a2, zero
   13f3c: a1 b9        	j	0x13b94 <ediv+0x150>
   13f3e: 85 48        	addi	a7, zero, 1
   13f40: 35 b9        	j	0x13b7c <ediv+0x138>

0000000000013f42 <emul>:
   13f42: 31 71        	addi	sp, sp, -192
   13f44: 22 f9        	sd	s0, 176(sp)
   13f46: 2a 84        	mv	s0, a0
   13f48: 03 55 25 01  	lhu	a0, 18(a0)
   13f4c: 21 67        	lui	a4, 8
   13f4e: 7d 17        	addi	a4, a4, -1
   13f50: 26 f5        	sd	s1, 168(sp)
   13f52: 4a f1        	sd	s2, 160(sp)
   13f54: 56 e5        	sd	s5, 136(sp)
   13f56: 06 fd        	sd	ra, 184(sp)
   13f58: 4e ed        	sd	s3, 152(sp)
   13f5a: 52 e9        	sd	s4, 144(sp)
   13f5c: 5a e1        	sd	s6, 128(sp)
   13f5e: de fc        	sd	s7, 120(sp)
   13f60: e2 f8        	sd	s8, 112(sp)
   13f62: e6 f4        	sd	s9, 104(sp)
   13f64: ea f0        	sd	s10, 96(sp)
   13f66: 33 78 e5 00  	and	a6, a0, a4
   13f6a: 2e 89        	mv	s2, a1
   13f6c: b2 84        	mv	s1, a2
   13f6e: b6 8a        	mv	s5, a3
   13f70: 63 12 e8 14  	bne	a6, a4, 0x140b4 <emul+0x172>
   13f74: 93 09 24 01  	addi	s3, s0, 18
   13f78: 22 87        	mv	a4, s0
   13f7a: 03 56 07 00  	lhu	a2, 0(a4)
   13f7e: 09 07        	addi	a4, a4, 2
   13f80: 63 15 06 1c  	bnez	a2, 0x1414a <emul+0x208>
   13f84: e3 1b 37 ff  	bne	a4, s3, 0x13f7a <emul+0x38>
   13f88: 03 56 29 01  	lhu	a2, 18(s2)
   13f8c: 93 47 f6 ff  	not	a5, a2
   13f90: 13 97 17 03  	slli	a4, a5, 49
   13f94: 63 07 07 18  	beqz	a4, 0x14122 <emul+0x1e0>
   13f98: a2 87        	mv	a5, s0
   13f9a: 03 d7 07 00  	lhu	a4, 0(a5)
   13f9e: 89 07        	addi	a5, a5, 2
   13fa0: 63 11 07 20  	bnez	a4, 0x141a2 <emul+0x260>
   13fa4: e3 9b 37 ff  	bne	a5, s3, 0x13f9a <emul+0x58>
   13fa8: f9 65        	lui	a1, 30
   13faa: 93 85 05 cf  	addi	a1, a1, -784
   13fae: 4a 85        	mv	a0, s2
   13fb0: ef f0 0f b3  	jal	0x132e0 <ecmp>
   13fb4: 63 0b 05 20  	beqz	a0, 0x141ca <emul+0x288>
   13fb8: 03 56 29 01  	lhu	a2, 18(s2)
   13fbc: a1 65        	lui	a1, 8
   13fbe: fd 15        	addi	a1, a1, -1
   13fc0: 33 77 b6 00  	and	a4, a2, a1
   13fc4: ba 86        	mv	a3, a4
   13fc6: 93 07 29 01  	addi	a5, s2, 18
   13fca: 63 12 b7 38  	bne	a4, a1, 0x1434e <emul+0x40c>
   13fce: 4a 87        	mv	a4, s2
   13fd0: 83 56 07 00  	lhu	a3, 0(a4)
   13fd4: 09 07        	addi	a4, a4, 2
   13fd6: 63 90 06 1a  	bnez	a3, 0x14176 <emul+0x234>
   13fda: e3 9b e7 fe  	bne	a5, a4, 0x13fd0 <emul+0x8e>
   13fde: f9 65        	lui	a1, 30
   13fe0: 93 85 05 cf  	addi	a1, a1, -784
   13fe4: 22 85        	mv	a0, s0
   13fe6: ef f0 af af  	jal	0x132e0 <ecmp>
   13fea: 63 00 05 1e  	beqz	a0, 0x141ca <emul+0x288>
   13fee: 03 55 24 01  	lhu	a0, 18(s0)
   13ff2: 03 56 29 01  	lhu	a2, 18(s2)
   13ff6: a1 67        	lui	a5, 8
   13ff8: fd 17        	addi	a5, a5, -1
   13ffa: 33 77 f5 00  	and	a4, a0, a5
   13ffe: b3 76 f6 00  	and	a3, a2, a5
   14002: ba 85        	mv	a1, a4
   14004: 63 02 f7 18  	beq	a4, a5, 0x14188 <emul+0x246>
   14008: 36 87        	mv	a4, a3
   1400a: ae 86        	mv	a3, a1
   1400c: a1 67        	lui	a5, 8
   1400e: 01 27        	sext.w	a4, a4
   14010: fd 17        	addi	a5, a5, -1
   14012: 63 17 f7 0a  	bne	a4, a5, 0x140c0 <emul+0x17e>
   14016: 93 07 29 01  	addi	a5, s2, 18
   1401a: 4a 87        	mv	a4, s2
   1401c: 83 55 07 00  	lhu	a1, 0(a4)
   14020: 09 07        	addi	a4, a4, 2
   14022: d9 ed        	bnez	a1, 0x140c0 <emul+0x17e>
   14024: e3 9c e7 fe  	bne	a5, a4, 0x1401c <emul+0xda>
   14028: a1 65        	lui	a1, 8
   1402a: fd 15        	addi	a1, a1, -1
   1402c: 1b 87 06 00  	sext.w	a4, a3
   14030: 93 09 24 01  	addi	s3, s0, 18
   14034: 63 99 b6 1c  	bne	a3, a1, 0x14206 <emul+0x2c4>
   14038: 83 57 04 00  	lhu	a5, 0(s0)
   1403c: 09 04        	addi	s0, s0, 2
   1403e: 63 9a 07 2e  	bnez	a5, 0x14332 <emul+0x3f0>
   14042: e3 9b 89 fe  	bne	s3, s0, 0x14038 <emul+0xf6>
   14046: 3d 81        	srli	a0, a0, 15
   14048: a1 67        	lui	a5, 8
   1404a: fd 17        	addi	a5, a5, -1
   1404c: 63 05 f7 16  	beq	a4, a5, 0x141b6 <emul+0x274>
   14050: 3d 82        	srli	a2, a2, 15
   14052: b3 47 c5 00  	xor	a5, a0, a2
   14056: 9b 97 f7 00  	slliw	a5, a5, 15
   1405a: 57 f7 84 c4  	<unknown>
   1405e: d7 76 80 04  	<unknown>
   14062: 23 99 f4 00  	sh	a5, 18(s1)
   14066: 57 3c 00 5e  	<unknown>
   1406a: a5 47        	addi	a5, zero, 9
   1406c: 57 f0 84 c4  	<unknown>
   14070: 27 dc 04 02  	<unknown>
   14074: b3 86 e7 40  	sub	a3, a5, a4
   14078: 63 89 e7 00  	beq	a5, a4, 0x1408a <emul+0x148>
   1407c: f3 27 20 c2  	csrr	a5, vlenb
   14080: a6 97        	add	a5, a5, s1
   14082: 57 f0 86 04  	<unknown>
   14086: 27 dc 07 02  	<unknown>
   1408a: 83 d7 24 01  	lhu	a5, 18(s1)
   1408e: 21 67        	lui	a4, 8
   14090: 7d 17        	addi	a4, a4, -1
   14092: d9 8f        	or	a5, a5, a4
   14094: 23 99 f4 00  	sh	a5, 18(s1)
   14098: ea 70        	ld	ra, 184(sp)
   1409a: 4a 74        	ld	s0, 176(sp)
   1409c: aa 74        	ld	s1, 168(sp)
   1409e: 0a 79        	ld	s2, 160(sp)
   140a0: ea 69        	ld	s3, 152(sp)
   140a2: 4a 6a        	ld	s4, 144(sp)
   140a4: aa 6a        	ld	s5, 136(sp)
   140a6: 0a 6b        	ld	s6, 128(sp)
   140a8: e6 7b        	ld	s7, 120(sp)
   140aa: 46 7c        	ld	s8, 112(sp)
   140ac: a6 7c        	ld	s9, 104(sp)
   140ae: 06 7d        	ld	s10, 96(sp)
   140b0: 29 61        	addi	sp, sp, 192
   140b2: 82 80        	ret
   140b4: 03 d6 25 01  	lhu	a2, 18(a1)
   140b8: b3 77 e6 00  	and	a5, a2, a4
   140bc: 63 83 e7 06  	beq	a5, a4, 0x14122 <emul+0x1e0>
   140c0: 22 85        	mv	a0, s0
   140c2: 8a 85        	mv	a1, sp
   140c4: ef f0 4f 96  	jal	0x13228 <emovi>
   140c8: 93 09 01 02  	addi	s3, sp, 32
   140cc: ce 85        	mv	a1, s3
   140ce: 4a 85        	mv	a0, s2
   140d0: ef f0 8f 95  	jal	0x13228 <emovi>
   140d4: 03 54 21 00  	lhu	s0, 2(sp)
   140d8: 03 5a 21 02  	lhu	s4, 34(sp)
   140dc: 63 18 04 10  	bnez	s0, 0x141ec <emul+0x2aa>
   140e0: 93 07 21 00  	addi	a5, sp, 2
   140e4: 34 08        	addi	a3, sp, 24
   140e6: 03 d7 07 00  	lhu	a4, 0(a5)
   140ea: 89 07        	addi	a5, a5, 2
   140ec: 63 19 07 24  	bnez	a4, 0x1433e <emul+0x3fc>
   140f0: e3 9b d7 fe  	bne	a5, a3, 0x140e6 <emul+0x1a4>
   140f4: d7 77 85 c4  	<unknown>
   140f8: 57 77 80 04  	<unknown>
   140fc: 29 47        	addi	a4, zero, 10
   140fe: 57 3c 00 5e  	<unknown>
   14102: 57 70 85 c4  	<unknown>
   14106: 27 dc 04 02  	<unknown>
   1410a: b3 06 f7 40  	sub	a3, a4, a5
   1410e: e3 05 f7 f8  	beq	a4, a5, 0x14098 <emul+0x156>
   14112: f3 27 20 c2  	csrr	a5, vlenb
   14116: a6 97        	add	a5, a5, s1
   14118: 57 f0 86 04  	<unknown>
   1411c: 27 dc 07 02  	<unknown>
   14120: a5 bf        	j	0x14098 <emul+0x156>
   14122: 93 07 29 01  	addi	a5, s2, 18
   14126: 4a 87        	mv	a4, s2
   14128: 83 56 07 00  	lhu	a3, 0(a4)
   1412c: 09 07        	addi	a4, a4, 2
   1412e: 8d ea        	bnez	a3, 0x14160 <emul+0x21e>
   14130: e3 1c f7 fe  	bne	a4, a5, 0x14128 <emul+0x1e6>
   14134: 21 67        	lui	a4, 8
   14136: 7d 17        	addi	a4, a4, -1
   14138: 93 09 24 01  	addi	s3, s0, 18
   1413c: e3 0e e8 e4  	beq	a6, a4, 0x13f98 <emul+0x56>
   14140: b3 76 e6 00  	and	a3, a2, a4
   14144: e3 85 e6 e8  	beq	a3, a4, 0x13fce <emul+0x8c>
   14148: a5 bf        	j	0x140c0 <emul+0x17e>
   1414a: 13 07 44 01  	addi	a4, s0, 20
   1414e: 83 57 04 00  	lhu	a5, 0(s0)
   14152: 09 04        	addi	s0, s0, 2
   14154: 89 04        	addi	s1, s1, 2
   14156: 23 9f f4 fe  	sh	a5, -2(s1)
   1415a: e3 1a e4 fe  	bne	s0, a4, 0x1414e <emul+0x20c>
   1415e: 2d bf        	j	0x14098 <emul+0x156>
   14160: 13 07 49 01  	addi	a4, s2, 20
   14164: 83 57 09 00  	lhu	a5, 0(s2)
   14168: 09 09        	addi	s2, s2, 2
   1416a: 89 04        	addi	s1, s1, 2
   1416c: 23 9f f4 fe  	sh	a5, -2(s1)
   14170: e3 1a e9 fe  	bne	s2, a4, 0x14164 <emul+0x222>
   14174: 15 b7        	j	0x14098 <emul+0x156>
   14176: 03 55 24 01  	lhu	a0, 18(s0)
   1417a: a1 65        	lui	a1, 8
   1417c: fd 15        	addi	a1, a1, -1
   1417e: 33 77 b5 00  	and	a4, a0, a1
   14182: ba 86        	mv	a3, a4
   14184: e3 1b b7 e8  	bne	a4, a1, 0x1401a <emul+0xd8>
   14188: 93 09 24 01  	addi	s3, s0, 18
   1418c: a2 87        	mv	a5, s0
   1418e: 03 d7 07 00  	lhu	a4, 0(a5)
   14192: 89 07        	addi	a5, a5, 2
   14194: 63 11 07 1a  	bnez	a4, 0x14336 <emul+0x3f4>
   14198: e3 9b f9 fe  	bne	s3, a5, 0x1418e <emul+0x24c>
   1419c: 1b 87 06 00  	sext.w	a4, a3
   141a0: 61 bd        	j	0x14038 <emul+0xf6>
   141a2: a1 65        	lui	a1, 8
   141a4: fd 15        	addi	a1, a1, -1
   141a6: 33 77 b6 00  	and	a4, a2, a1
   141aa: ba 86        	mv	a3, a4
   141ac: 93 07 29 01  	addi	a5, s2, 18
   141b0: e3 0f b7 e0  	beq	a4, a1, 0x13fce <emul+0x8c>
   141b4: e1 bf        	j	0x1418c <emul+0x24a>
   141b6: 93 07 29 01  	addi	a5, s2, 18
   141ba: 03 57 09 00  	lhu	a4, 0(s2)
   141be: 09 09        	addi	s2, s2, 2
   141c0: 63 17 07 16  	bnez	a4, 0x1432e <emul+0x3ec>
   141c4: e3 9b 27 ff  	bne	a5, s2, 0x141ba <emul+0x278>
   141c8: 61 b5        	j	0x14050 <emul+0x10e>
   141ca: d7 77 80 04  	<unknown>
   141ce: f1 77        	lui	a5, 1048572
   141d0: 57 3c 00 5e  	<unknown>
   141d4: 57 70 84 c4  	<unknown>
   141d8: 27 dc 04 02  	<unknown>
   141dc: 23 98 f4 00  	sh	a5, 16(s1)
   141e0: e1 77        	lui	a5, 1048568
   141e2: 93 c7 f7 ff  	not	a5, a5
   141e6: 23 99 f4 00  	sh	a5, 18(s1)
   141ea: 7d b5        	j	0x14098 <emul+0x156>
   141ec: d2 87        	mv	a5, s4
   141ee: 52 89        	mv	s2, s4
   141f0: 85 e7        	bnez	a5, 0x14218 <emul+0x2d6>
   141f2: 93 07 21 02  	addi	a5, sp, 34
   141f6: 34 18        	addi	a3, sp, 56
   141f8: 03 d7 07 00  	lhu	a4, 0(a5)
   141fc: 89 07        	addi	a5, a5, 2
   141fe: 11 e7        	bnez	a4, 0x1420a <emul+0x2c8>
   14200: e3 9c f6 fe  	bne	a3, a5, 0x141f8 <emul+0x2b6>
   14204: c5 bd        	j	0x140f4 <emul+0x1b2>
   14206: 3d 81        	srli	a0, a0, 15
   14208: 4d bf        	j	0x141ba <emul+0x278>
   1420a: 4e 85        	mv	a0, s3
   1420c: ef e0 3f d8  	jal	0x12f8e <enormlz>
   14210: 83 57 21 02  	lhu	a5, 34(sp)
   14214: 33 09 aa 40  	sub	s2, s4, a0
   14218: 83 56 01 02  	lhu	a3, 32(sp)
   1421c: 57 f7 85 c4  	<unknown>
   14220: 23 9b fa 02  	sh	a5, 54(s5)
   14224: 23 9a da 02  	sh	a3, 52(s5)
   14228: 57 76 80 04  	<unknown>
   1422c: 13 8a 8a 03  	addi	s4, s5, 56
   14230: 57 3c 00 5e  	<unknown>
   14234: ad 47        	addi	a5, zero, 11
   14236: 57 f0 85 c4  	<unknown>
   1423a: 27 5c 0a 02  	<unknown>
   1423e: b3 86 e7 40  	sub	a3, a5, a4
   14242: 63 8b e7 00  	beq	a5, a4, 0x14258 <emul+0x316>
   14246: f3 27 20 c2  	csrr	a5, vlenb
   1424a: 93 87 87 03  	addi	a5, a5, 56
   1424e: d6 97        	add	a5, a5, s5
   14250: 57 f0 86 04  	<unknown>
   14254: 27 dc 07 02  	<unknown>
   14258: 93 8c ca 04  	addi	s9, s5, 76
   1425c: 01 4c        	mv	s8, zero
   1425e: 13 0b 81 01  	addi	s6, sp, 24
   14262: 13 0d 41 00  	addi	s10, sp, 4
   14266: 93 0b 21 04  	addi	s7, sp, 66
   1426a: 03 55 0b 00  	lhu	a0, 0(s6)
   1426e: 79 1b        	addi	s6, s6, -2
   14270: 41 e9        	bnez	a0, 0x14300 <emul+0x3be>
   14272: 83 d7 ca 04  	lhu	a5, 76(s5)
   14276: 66 87        	mv	a4, s9
   14278: 33 6c fc 00  	or	s8, s8, a5
   1427c: 03 56 e7 ff  	lhu	a2, -2(a4)
   14280: 79 17        	addi	a4, a4, -2
   14282: 23 11 c7 00  	sh	a2, 2(a4)
   14286: e3 1b ea fe  	bne	s4, a4, 0x1427c <emul+0x33a>
   1428a: 23 9c 0a 02  	sh	zero, 56(s5)
   1428e: e3 1e ab fd  	bne	s6, s10, 0x1426a <emul+0x328>
   14292: 93 87 4a 03  	addi	a5, s5, 52
   14296: 57 f7 86 c4  	<unknown>
   1429a: b5 46        	addi	a3, zero, 13
   1429c: 07 dc 07 02  	<unknown>
   142a0: 27 dc 09 02  	<unknown>
   142a4: 33 86 e6 40  	sub	a2, a3, a4
   142a8: 63 80 e6 02  	beq	a3, a4, 0x142c8 <emul+0x386>
   142ac: f3 27 20 c2  	csrr	a5, vlenb
   142b0: 13 87 47 03  	addi	a4, a5, 52
   142b4: 93 87 07 02  	addi	a5, a5, 32
   142b8: 56 97        	add	a4, a4, s5
   142ba: 8a 97        	add	a5, a5, sp
   142bc: 57 70 86 04  	<unknown>
   142c0: 07 5c 07 02  	<unknown>
   142c4: 27 dc 07 02  	<unknown>
   142c8: f1 76        	lui	a3, 1048572
   142ca: 4a 94        	add	s0, s0, s2
   142cc: 89 06        	addi	a3, a3, 2
   142ce: e2 85        	mv	a1, s8
   142d0: 4e 85        	mv	a0, s3
   142d2: d6 87        	mv	a5, s5
   142d4: 13 07 00 04  	addi	a4, zero, 64
   142d8: a2 96        	add	a3, a3, s0
   142da: 01 46        	mv	a2, zero
   142dc: ef f0 cf 8e  	jal	0x133c8 <emdnorm>
   142e0: 03 57 01 02  	lhu	a4, 32(sp)
   142e4: 83 57 01 00  	lhu	a5, 0(sp)
   142e8: a6 85        	mv	a1, s1
   142ea: 4e 85        	mv	a0, s3
   142ec: 99 8f        	sub	a5, a5, a4
   142ee: b3 37 f0 00  	snez	a5, a5
   142f2: bb 07 f0 40  	negw	a5, a5
   142f6: 23 10 f1 02  	sh	a5, 32(sp)
   142fa: ef e0 df be  	jal	0x12ee6 <emovo.constprop.0>
   142fe: 69 bb        	j	0x14098 <emul+0x156>
   14300: 90 00        	addi	a2, sp, 64
   14302: ce 85        	mv	a1, s3
   14304: ef e0 9f b5  	jal	0x12e5c <m16m>
   14308: 66 85        	mv	a0, s9
   1430a: 81 45        	mv	a1, zero
   1430c: b0 08        	addi	a2, sp, 88
   1430e: 83 57 05 00  	lhu	a5, 0(a0)
   14312: 03 57 06 00  	lhu	a4, 0(a2)
   14316: 79 15        	addi	a0, a0, -2
   14318: 79 16        	addi	a2, a2, -2
   1431a: 3e 97        	add	a4, a4, a5
   1431c: 2e 97        	add	a4, a4, a1
   1431e: 93 55 07 01  	srli	a1, a4, 16
   14322: 23 11 e5 00  	sh	a4, 2(a0)
   14326: 85 89        	andi	a1, a1, 1
   14328: e3 13 76 ff  	bne	a2, s7, 0x1430e <emul+0x3cc>
   1432c: 99 b7        	j	0x14272 <emul+0x330>
   1432e: 01 46        	mv	a2, zero
   14330: 0d b3        	j	0x14052 <emul+0x110>
   14332: 01 45        	mv	a0, zero
   14334: 11 bb        	j	0x14048 <emul+0x106>
   14336: 36 87        	mv	a4, a3
   14338: a1 66        	lui	a3, 8
   1433a: fd 16        	addi	a3, a3, -1
   1433c: c1 b9        	j	0x1400c <emul+0xca>
   1433e: 0a 85        	mv	a0, sp
   14340: ef e0 ff c4  	jal	0x12f8e <enormlz>
   14344: 83 57 21 02  	lhu	a5, 34(sp)
   14348: 33 04 a0 40  	neg	s0, a0
   1434c: 4d b5        	j	0x141ee <emul+0x2ac>
   1434e: 03 55 24 01  	lhu	a0, 18(s0)
   14352: b3 77 b5 00  	and	a5, a0, a1
   14356: e3 8b b7 e2  	beq	a5, a1, 0x1418c <emul+0x24a>
   1435a: 9d b3        	j	0x140c0 <emul+0x17e>

000000000001435c <_ldtoa_r>:
   1435c: 13 01 01 d2  	addi	sp, sp, -736
   14360: 13 03 10 09  	addi	t1, zero, 145
   14364: b2 ed        	sd	a2, 216(sp)
   14366: 03 3e 05 06  	ld	t3, 96(a0)
   1436a: 13 16 03 02  	slli	a2, t1, 32
   1436e: 7d 16        	addi	a2, a2, -1
   14370: 23 38 81 2c  	sd	s0, 720(sp)
   14374: 23 3c 11 2c  	sd	ra, 728(sp)
   14378: 23 34 91 2c  	sd	s1, 712(sp)
   1437c: 23 30 21 2d  	sd	s2, 704(sp)
   14380: 23 3c 31 2b  	sd	s3, 696(sp)
   14384: 23 38 41 2b  	sd	s4, 688(sp)
   14388: 23 34 51 2b  	sd	s5, 680(sp)
   1438c: 23 30 61 2b  	sd	s6, 672(sp)
   14390: 23 3c 71 29  	sd	s7, 664(sp)
   14394: 23 38 81 29  	sd	s8, 656(sp)
   14398: 23 34 91 29  	sd	s9, 648(sp)
   1439c: 23 30 a1 29  	sd	s10, 640(sp)
   143a0: 23 3c b1 27  	sd	s11, 632(sp)
   143a4: 2a e8        	sd	a0, 16(sp)
   143a6: ae e9        	sd	a1, 208(sp)
   143a8: 23 30 c1 22  	sd	a2, 544(sp)
   143ac: 36 e4        	sd	a3, 8(sp)
   143ae: 3a f0        	sd	a4, 32(sp)
   143b0: 3e f8        	sd	a5, 48(sp)
   143b2: c6 e4        	sd	a7, 72(sp)
   143b4: 42 84        	mv	s0, a6
   143b6: 63 00 0e 02  	beqz	t3, 0x143d6 <_ldtoa_r+0x7a>
   143ba: 38 55        	lw	a4, 104(a0)
   143bc: 85 47        	addi	a5, zero, 1
   143be: aa 84        	mv	s1, a0
   143c0: bb 97 e7 00  	sllw	a5, a5, a4
   143c4: f2 85        	mv	a1, t3
   143c6: 23 24 ee 00  	sw	a4, 8(t3)
   143ca: 23 26 fe 00  	sw	a5, 12(t3)
   143ce: ef 10 50 28  	jal	0x15e52 <_Bfree>
   143d2: 23 b0 04 06  	sd	zero, 96(s1)
   143d6: 8c 11        	addi	a1, sp, 224
   143d8: 88 09        	addi	a0, sp, 208
   143da: ef f0 6f d1  	jal	0x138f0 <e113toe.isra.0>
   143de: 03 57 21 0f  	lhu	a4, 242(sp)
   143e2: a1 67        	lui	a5, 8
   143e4: fd 17        	addi	a5, a5, -1
   143e6: b3 76 f7 00  	and	a3, a4, a5
   143ea: 36 f4        	sd	a3, 40(sp)
   143ec: 63 9c f6 00  	bne	a3, a5, 0x14404 <_ldtoa_r+0xa8>
   143f0: 9c 11        	addi	a5, sp, 224
   143f2: 13 06 21 0f  	addi	a2, sp, 242
   143f6: 83 d6 07 00  	lhu	a3, 0(a5)
   143fa: 89 07        	addi	a5, a5, 2
   143fc: 63 9d 06 3c  	bnez	a3, 0x147d6 <_ldtoa_r+0x47a>
   14400: e3 9b c7 fe  	bne	a5, a2, 0x143f6 <_ldtoa_r+0x9a>
   14404: 9b 17 07 01  	slliw	a5, a4, 16
   14408: 9b d7 07 41  	sraiw	a5, a5, 16
   1440c: 22 67        	ld	a4, 8(sp)
   1440e: 9b d7 f7 01  	srliw	a5, a5, 31
   14412: 1c c0        	sw	a5, 0(s0)
   14414: 8d 47        	addi	a5, zero, 3
   14416: 63 02 f7 7a  	beq	a4, a5, 0x14bba <_ldtoa_r+0x85e>
   1441a: d1 47        	addi	a5, zero, 20
   1441c: 3e ec        	sd	a5, 24(sp)
   1441e: a2 67        	ld	a5, 8(sp)
   14420: 89 cf        	beqz	a5, 0x1443a <_ldtoa_r+0xde>
   14422: 82 77        	ld	a5, 32(sp)
   14424: fd 37        	addiw	a5, a5, -1
   14426: 93 06 a0 02  	addi	a3, zero, 42
   1442a: 3e 87        	mv	a4, a5
   1442c: 63 d4 f6 00  	bge	a3, a5, 0x14434 <_ldtoa_r+0xd8>
   14430: 13 07 a0 02  	addi	a4, zero, 42
   14434: 9b 07 07 00  	sext.w	a5, a4
   14438: 3e ec        	sd	a5, 24(sp)
   1443a: 03 27 41 22  	lw	a4, 548(sp)
   1443e: a1 67        	lui	a5, 8
   14440: fd 17        	addi	a5, a5, -1
   14442: 3a fc        	sd	a4, 56(sp)
   14444: 22 77        	ld	a4, 40(sp)
   14446: 63 0b f7 3a  	beq	a4, a5, 0x147fc <_ldtoa_r+0x4a0>
   1444a: 93 07 00 09  	addi	a5, zero, 144
   1444e: 23 22 f1 22  	sw	a5, 548(sp)
   14452: 00 02        	addi	s0, sp, 256
   14454: 90 11        	addi	a2, sp, 224
   14456: d7 77 85 c4  	<unknown>
   1445a: 73 27 20 c2  	csrr	a4, vlenb
   1445e: a9 46        	addi	a3, zero, 10
   14460: 07 5c 06 02  	<unknown>
   14464: 27 5c 04 02  	<unknown>
   14468: 33 0c e4 00  	add	s8, s0, a4
   1446c: 33 8d f6 40  	sub	s10, a3, a5
   14470: 63 8c f6 3e  	beq	a3, a5, 0x14868 <_ldtoa_r+0x50c>
   14474: 94 09        	addi	a3, sp, 208
   14476: 93 07 07 01  	addi	a5, a4, 16
   1447a: b6 97        	add	a5, a5, a3
   1447c: 57 70 8d 04  	<unknown>
   14480: 07 dc 07 02  	<unknown>
   14484: 27 5c 0c 02  	<unknown>
   14488: 83 56 21 11  	lhu	a3, 274(sp)
   1448c: b6 e0        	sd	a3, 64(sp)
   1448e: 93 d7 f6 00  	srli	a5, a3, 15
   14492: e3 97 07 20  	bnez	a5, 0x14ea0 <_ldtoa_r+0xb44>
   14496: f9 67        	lui	a5, 30
   14498: 93 8d 87 cd  	addi	s11, a5, -808
   1449c: 13 09 01 12  	addi	s2, sp, 288
   144a0: 33 86 ed 00  	add	a2, s11, a4
   144a4: 4a 97        	add	a4, a4, s2
   144a6: b2 f0        	sd	a2, 96(sp)
   144a8: ba e8        	sd	a4, 80(sp)
   144aa: 57 70 85 c4  	<unknown>
   144ae: 07 dc 0d 02  	<unknown>
   144b2: 27 5c 09 02  	<unknown>
   144b6: 57 70 8d 04  	<unknown>
   144ba: 07 5c 06 02  	<unknown>
   144be: 27 5c 07 02  	<unknown>
   144c2: 63 9a 06 3c  	bnez	a3, 0x14896 <_ldtoa_r+0x53a>
   144c6: 22 87        	mv	a4, s0
   144c8: 13 06 21 11  	addi	a2, sp, 274
   144cc: 83 56 07 00  	lhu	a3, 0(a4)
   144d0: 63 95 06 58  	bnez	a3, 0x14a5a <_ldtoa_r+0x6fe>
   144d4: 09 07        	addi	a4, a4, 2
   144d6: e3 1b e6 fe  	bne	a2, a4, 0x144cc <_ldtoa_r+0x170>
   144da: d7 77 86 c4  	<unknown>
   144de: 81 44        	mv	s1, zero
   144e0: 93 0b 01 1e  	addi	s7, sp, 480
   144e4: 13 0a 01 16  	addi	s4, sp, 352
   144e8: 93 09 01 22  	addi	s3, sp, 544
   144ec: be ec        	sd	a5, 88(sp)
   144ee: d2 85        	mv	a1, s4
   144f0: 4a 85        	mv	a0, s2
   144f2: ef e0 7f d3  	jal	0x13228 <emovi>
   144f6: 57 70 86 c4  	<unknown>
   144fa: 07 5c 0a 02  	<unknown>
   144fe: 27 5c 09 02  	<unknown>
   14502: 66 67        	ld	a4, 88(sp)
   14504: b1 47        	addi	a5, zero, 12
   14506: b3 8d e7 40  	sub	s11, a5, a4
   1450a: 63 99 e7 66  	bne	a5, a4, 0x14b7c <_ldtoa_r+0x820>
   1450e: d2 85        	mv	a1, s4
   14510: 22 85        	mv	a0, s0
   14512: 23 1c 01 12  	sh	zero, 312(sp)
   14516: ef e0 3f d1  	jal	0x13228 <emovi>
   1451a: 57 70 86 c4  	<unknown>
   1451e: 07 5c 0a 02  	<unknown>
   14522: 27 5c 04 02  	<unknown>
   14526: 4e 86        	mv	a2, s3
   14528: a2 85        	mv	a1, s0
   1452a: 4a 85        	mv	a0, s2
   1452c: 23 1c 01 10  	sh	zero, 280(sp)
   14530: ef f0 ef aa  	jal	0x137de <eiremain>
   14534: 83 56 c1 26  	lhu	a3, 620(sp)
   14538: 63 92 06 12  	bnez	a3, 0x1465c <_ldtoa_r+0x300>
   1453c: 93 0a 01 14  	addi	s5, sp, 320
   14540: 73 2a 20 c2  	csrr	s4, vlenb
   14544: 56 9a        	add	s4, s4, s5
   14546: 79 6b        	lui	s6, 30
   14548: 13 0d 21 10  	addi	s10, sp, 258
   1454c: 93 0c 21 14  	addi	s9, sp, 322
   14550: b6 e8        	sd	a3, 80(sp)
   14552: 93 05 0b cf  	addi	a1, s6, -784
   14556: 22 85        	mv	a0, s0
   14558: ef e0 9f d8  	jal	0x132e0 <ecmp>
   1455c: 63 0d 05 64  	beqz	a0, 0x14bb6 <_ldtoa_r+0x85a>
   14560: 01 47        	mv	a4, zero
   14562: 34 0a        	addi	a3, sp, 280
   14564: 83 d7 06 00  	lhu	a5, 0(a3)
   14568: 13 d6 f7 00  	srli	a2, a5, 15
   1456c: 19 c2        	beqz	a2, 0x14572 <_ldtoa_r+0x216>
   1456e: 13 67 17 00  	ori	a4, a4, 1
   14572: 9b 97 17 00  	slliw	a5, a5, 1
   14576: c2 17        	slli	a5, a5, 48
   14578: 13 76 27 00  	andi	a2, a4, 2
   1457c: c1 93        	srli	a5, a5, 48
   1457e: 19 c2        	beqz	a2, 0x14584 <_ldtoa_r+0x228>
   14580: 93 e7 17 00  	ori	a5, a5, 1
   14584: 1b 17 17 00  	slliw	a4, a4, 1
   14588: 23 90 f6 00  	sh	a5, 0(a3)
   1458c: 42 17        	slli	a4, a4, 48
   1458e: f9 16        	addi	a3, a3, -2
   14590: 41 93        	srli	a4, a4, 48
   14592: e3 99 a6 fd  	bne	a3, s10, 0x14564 <_ldtoa_r+0x208>
   14596: 57 70 86 c4  	<unknown>
   1459a: 07 5c 04 02  	<unknown>
   1459e: 27 dc 0a 02  	<unknown>
   145a2: 63 88 0d 00  	beqz	s11, 0x145b2 <_ldtoa_r+0x256>
   145a6: 57 f0 8d 04  	<unknown>
   145aa: 07 5c 0c 02  	<unknown>
   145ae: 27 5c 0a 02  	<unknown>
   145b2: 23 1c 01 14  	sh	zero, 344(sp)
   145b6: 01 47        	mv	a4, zero
   145b8: b4 0a        	addi	a3, sp, 344
   145ba: 83 d7 06 00  	lhu	a5, 0(a3)
   145be: 13 d6 f7 00  	srli	a2, a5, 15
   145c2: 19 c2        	beqz	a2, 0x145c8 <_ldtoa_r+0x26c>
   145c4: 13 67 17 00  	ori	a4, a4, 1
   145c8: 9b 97 17 00  	slliw	a5, a5, 1
   145cc: c2 17        	slli	a5, a5, 48
   145ce: 13 76 27 00  	andi	a2, a4, 2
   145d2: c1 93        	srli	a5, a5, 48
   145d4: 19 c2        	beqz	a2, 0x145da <_ldtoa_r+0x27e>
   145d6: 93 e7 17 00  	ori	a5, a5, 1
   145da: 1b 17 17 00  	slliw	a4, a4, 1
   145de: 23 90 f6 00  	sh	a5, 0(a3)
   145e2: 42 17        	slli	a4, a4, 48
   145e4: f9 16        	addi	a3, a3, -2
   145e6: 41 93        	srli	a4, a4, 48
   145e8: e3 99 96 fd  	bne	a3, s9, 0x145ba <_ldtoa_r+0x25e>
   145ec: 01 47        	mv	a4, zero
   145ee: b4 0a        	addi	a3, sp, 344
   145f0: 83 d7 06 00  	lhu	a5, 0(a3)
   145f4: 13 d6 f7 00  	srli	a2, a5, 15
   145f8: 19 c2        	beqz	a2, 0x145fe <_ldtoa_r+0x2a2>
   145fa: 13 67 17 00  	ori	a4, a4, 1
   145fe: 9b 97 17 00  	slliw	a5, a5, 1
   14602: c2 17        	slli	a5, a5, 48
   14604: 13 76 27 00  	andi	a2, a4, 2
   14608: c1 93        	srli	a5, a5, 48
   1460a: 19 c2        	beqz	a2, 0x14610 <_ldtoa_r+0x2b4>
   1460c: 93 e7 17 00  	ori	a5, a5, 1
   14610: 1b 17 17 00  	slliw	a4, a4, 1
   14614: 23 90 f6 00  	sh	a5, 0(a3)
   14618: 42 17        	slli	a4, a4, 48
   1461a: f9 16        	addi	a3, a3, -2
   1461c: 41 93        	srli	a4, a4, 48
   1461e: e3 99 96 fd  	bne	a3, s9, 0x145f0 <_ldtoa_r+0x294>
   14622: 01 46        	mv	a2, zero
   14624: 34 0a        	addi	a3, sp, 280
   14626: b8 0a        	addi	a4, sp, 344
   14628: 83 d5 06 00  	lhu	a1, 0(a3)
   1462c: 83 57 07 00  	lhu	a5, 0(a4)
   14630: f9 16        	addi	a3, a3, -2
   14632: 79 17        	addi	a4, a4, -2
   14634: ae 97        	add	a5, a5, a1
   14636: b2 97        	add	a5, a5, a2
   14638: 13 d6 07 01  	srli	a2, a5, 16
   1463c: 23 91 f6 00  	sh	a5, 2(a3)
   14640: 05 8a        	andi	a2, a2, 1
   14642: e3 13 97 ff  	bne	a4, s9, 0x14628 <_ldtoa_r+0x2cc>
   14646: 4e 86        	mv	a2, s3
   14648: a2 85        	mv	a1, s0
   1464a: 4a 85        	mv	a0, s2
   1464c: ef f0 2f 99  	jal	0x137de <eiremain>
   14650: 83 57 c1 26  	lhu	a5, 620(sp)
   14654: fd 34        	addiw	s1, s1, -1
   14656: e3 8e 07 ee  	beqz	a5, 0x14552 <_ldtoa_r+0x1f6>
   1465a: be 86        	mv	a3, a5
   1465c: 06 67        	ld	a4, 64(sp)
   1465e: 93 07 00 02  	addi	a5, zero, 32
   14662: 19 c3        	beqz	a4, 0x14668 <_ldtoa_r+0x30c>
   14664: 93 07 d0 02  	addi	a5, zero, 45
   14668: 22 67        	ld	a4, 8(sp)
   1466a: 23 00 f1 1e  	sb	a5, 480(sp)
   1466e: 0d 46        	addi	a2, zero, 3
   14670: e2 67        	ld	a5, 24(sp)
   14672: 63 13 c7 00  	bne	a4, a2, 0x14678 <_ldtoa_r+0x31c>
   14676: a5 9f        	addw	a5, a5, s1
   14678: 93 05 a0 02  	addi	a1, zero, 42
   1467c: 1b 86 07 00  	sext.w	a2, a5
   14680: 63 d4 f5 00  	bge	a1, a5, 0x14688 <_ldtoa_r+0x32c>
   14684: 13 06 a0 02  	addi	a2, zero, 42
   14688: a9 45        	addi	a1, zero, 10
   1468a: 63 8c b6 54  	beq	a3, a1, 0x14be2 <_ldtoa_r+0x886>
   1468e: 9b 86 06 03  	addiw	a3, a3, 48
   14692: a3 00 d1 1e  	sb	a3, 481(sp)
   14696: 93 06 e0 02  	addi	a3, zero, 46
   1469a: 23 01 d1 1e  	sb	a3, 482(sp)
   1469e: e3 cd 07 1c  	bltz	a5, 0x15078 <_ldtoa_r+0xd1c>
   146a2: 13 0a 31 1e  	addi	s4, sp, 483
   146a6: 93 0a 01 14  	addi	s5, sp, 320
   146aa: f3 27 20 c2  	csrr	a5, vlenb
   146ae: 01 4d        	mv	s10, zero
   146b0: d6 97        	add	a5, a5, s5
   146b2: a6 e0        	sd	s1, 64(sp)
   146b4: de e8        	sd	s7, 80(sp)
   146b6: d2 84        	mv	s1, s4
   146b8: ee 8b        	mv	s7, s11
   146ba: 6a 8a        	mv	s4, s10
   146bc: e2 8d        	mv	s11, s8
   146be: 93 0c 21 10  	addi	s9, sp, 258
   146c2: 4e 8c        	mv	s8, s3
   146c4: 13 0b 21 14  	addi	s6, sp, 322
   146c8: 3e 8d        	mv	s10, a5
   146ca: b2 89        	mv	s3, a2
   146cc: 01 46        	mv	a2, zero
   146ce: 2c 0a        	addi	a1, sp, 280
   146d0: 83 d7 05 00  	lhu	a5, 0(a1)
   146d4: 13 d5 f7 00  	srli	a0, a5, 15
   146d8: 19 c1        	beqz	a0, 0x146de <_ldtoa_r+0x382>
   146da: 13 66 16 00  	ori	a2, a2, 1
   146de: 9b 97 17 00  	slliw	a5, a5, 1
   146e2: c2 17        	slli	a5, a5, 48
   146e4: 13 75 26 00  	andi	a0, a2, 2
   146e8: c1 93        	srli	a5, a5, 48
   146ea: 19 c1        	beqz	a0, 0x146f0 <_ldtoa_r+0x394>
   146ec: 93 e7 17 00  	ori	a5, a5, 1
   146f0: 1b 16 16 00  	slliw	a2, a2, 1
   146f4: 23 90 f5 00  	sh	a5, 0(a1)
   146f8: 42 16        	slli	a2, a2, 48
   146fa: f9 15        	addi	a1, a1, -2
   146fc: 41 92        	srli	a2, a2, 48
   146fe: e3 99 95 fd  	bne	a1, s9, 0x146d0 <_ldtoa_r+0x374>
   14702: 57 70 86 c4  	<unknown>
   14706: 07 5c 04 02  	<unknown>
   1470a: 27 dc 0a 02  	<unknown>
   1470e: 63 88 0b 00  	beqz	s7, 0x1471e <_ldtoa_r+0x3c2>
   14712: 57 f0 8b 04  	<unknown>
   14716: 07 dc 0d 02  	<unknown>
   1471a: 27 5c 0d 02  	<unknown>
   1471e: 23 1c 01 14  	sh	zero, 344(sp)
   14722: 01 46        	mv	a2, zero
   14724: ac 0a        	addi	a1, sp, 344
   14726: 83 d7 05 00  	lhu	a5, 0(a1)
   1472a: 13 d5 f7 00  	srli	a0, a5, 15
   1472e: 19 c1        	beqz	a0, 0x14734 <_ldtoa_r+0x3d8>
   14730: 13 66 16 00  	ori	a2, a2, 1
   14734: 9b 97 17 00  	slliw	a5, a5, 1
   14738: c2 17        	slli	a5, a5, 48
   1473a: 13 75 26 00  	andi	a0, a2, 2
   1473e: c1 93        	srli	a5, a5, 48
   14740: 19 c1        	beqz	a0, 0x14746 <_ldtoa_r+0x3ea>
   14742: 93 e7 17 00  	ori	a5, a5, 1
   14746: 1b 16 16 00  	slliw	a2, a2, 1
   1474a: 23 90 f5 00  	sh	a5, 0(a1)
   1474e: 42 16        	slli	a2, a2, 48
   14750: f9 15        	addi	a1, a1, -2
   14752: 41 92        	srli	a2, a2, 48
   14754: e3 99 65 fd  	bne	a1, s6, 0x14726 <_ldtoa_r+0x3ca>
   14758: 01 46        	mv	a2, zero
   1475a: ac 0a        	addi	a1, sp, 344
   1475c: 83 d7 05 00  	lhu	a5, 0(a1)
   14760: 13 d5 f7 00  	srli	a0, a5, 15
   14764: 19 c1        	beqz	a0, 0x1476a <_ldtoa_r+0x40e>
   14766: 13 66 16 00  	ori	a2, a2, 1
   1476a: 9b 97 17 00  	slliw	a5, a5, 1
   1476e: c2 17        	slli	a5, a5, 48
   14770: 13 75 26 00  	andi	a0, a2, 2
   14774: c1 93        	srli	a5, a5, 48
   14776: 19 c1        	beqz	a0, 0x1477c <_ldtoa_r+0x420>
   14778: 93 e7 17 00  	ori	a5, a5, 1
   1477c: 1b 16 16 00  	slliw	a2, a2, 1
   14780: 23 90 f5 00  	sh	a5, 0(a1)
   14784: 42 16        	slli	a2, a2, 48
   14786: f9 15        	addi	a1, a1, -2
   14788: 41 92        	srli	a2, a2, 48
   1478a: e3 99 65 fd  	bne	a1, s6, 0x1475c <_ldtoa_r+0x400>
   1478e: 01 45        	mv	a0, zero
   14790: 2c 0a        	addi	a1, sp, 280
   14792: b0 0a        	addi	a2, sp, 344
   14794: 03 d8 05 00  	lhu	a6, 0(a1)
   14798: 83 57 06 00  	lhu	a5, 0(a2)
   1479c: f9 15        	addi	a1, a1, -2
   1479e: 79 16        	addi	a2, a2, -2
   147a0: c2 97        	add	a5, a5, a6
   147a2: aa 97        	add	a5, a5, a0
   147a4: 13 d5 07 01  	srli	a0, a5, 16
   147a8: 23 91 f5 00  	sh	a5, 2(a1)
   147ac: 05 89        	andi	a0, a0, 1
   147ae: e3 13 66 ff  	bne	a2, s6, 0x14794 <_ldtoa_r+0x438>
   147b2: 62 86        	mv	a2, s8
   147b4: a2 85        	mv	a1, s0
   147b6: 4a 85        	mv	a0, s2
   147b8: ef f0 6f 82  	jal	0x137de <eiremain>
   147bc: 83 55 c1 26  	lhu	a1, 620(sp)
   147c0: 93 87 14 00  	addi	a5, s1, 1
   147c4: 05 2a        	addiw	s4, s4, 1
   147c6: 1b 86 05 03  	addiw	a2, a1, 48
   147ca: a3 8f c7 fe  	sb	a2, -1(a5)
   147ce: 63 cc 49 1d  	blt	s3, s4, 0x149a6 <_ldtoa_r+0x64a>
   147d2: be 84        	mv	s1, a5
   147d4: e5 bd        	j	0x146cc <_ldtoa_r+0x370>
   147d6: 22 67        	ld	a4, 8(sp)
   147d8: 23 20 04 00  	sw	zero, 0(s0)
   147dc: 8d 47        	addi	a5, zero, 3
   147de: e3 1e f7 c2  	bne	a4, a5, 0x1441a <_ldtoa_r+0xbe>
   147e2: 82 76        	ld	a3, 32(sp)
   147e4: 13 07 a0 02  	addi	a4, zero, 42
   147e8: 9b 87 06 00  	sext.w	a5, a3
   147ec: 63 54 d7 00  	bge	a4, a3, 0x147f4 <_ldtoa_r+0x498>
   147f0: 93 07 a0 02  	addi	a5, zero, 42
   147f4: 03 27 41 22  	lw	a4, 548(sp)
   147f8: 3e ec        	sd	a5, 24(sp)
   147fa: 3a fc        	sd	a4, 56(sp)
   147fc: 9c 11        	addi	a5, sp, 224
   147fe: 93 06 21 0f  	addi	a3, sp, 242
   14802: 19 a0        	j	0x14808 <_ldtoa_r+0x4ac>
   14804: e3 83 d7 c4  	beq	a5, a3, 0x1444a <_ldtoa_r+0xee>
   14808: 03 d7 07 00  	lhu	a4, 0(a5)
   1480c: 89 07        	addi	a5, a5, 2
   1480e: 7d db        	beqz	a4, 0x14804 <_ldtoa_r+0x4a8>
   14810: 93 0b 01 1e  	addi	s7, sp, 480
   14814: f9 65        	lui	a1, 30
   14816: 93 85 85 a9  	addi	a1, a1, -1384
   1481a: 5e 85        	mv	a0, s7
   1481c: ef 20 a0 3a  	jal	0x16bc6 <sprintf>
   14820: e2 77        	ld	a5, 56(sp)
   14822: 89 64        	lui	s1, 2
   14824: 93 84 f4 70  	addi	s1, s1, 1807
   14828: 23 22 f1 22  	sw	a5, 548(sp)
   1482c: 23 28 91 22  	sw	s1, 560(sp)
   14830: 9c 11        	addi	a5, sp, 224
   14832: 13 06 21 0f  	addi	a2, sp, 242
   14836: 03 d7 07 00  	lhu	a4, 0(a5)
   1483a: 89 07        	addi	a5, a5, 2
   1483c: 63 17 07 7e  	bnez	a4, 0x1502a <_ldtoa_r+0xcce>
   14840: e3 9b c7 fe  	bne	a5, a2, 0x14836 <_ldtoa_r+0x4da>
   14844: 42 77        	ld	a4, 48(sp)
   14846: 89 67        	lui	a5, 2
   14848: 93 87 f7 70  	addi	a5, a5, 1807
   1484c: 1c c3        	sw	a5, 0(a4)
   1484e: de 87        	mv	a5, s7
   14850: 13 06 00 02  	addi	a2, zero, 32
   14854: 93 05 d0 02  	addi	a1, zero, 45
   14858: 03 c7 07 00  	lbu	a4, 0(a5)
   1485c: 63 04 c7 00  	beq	a4, a2, 0x14864 <_ldtoa_r+0x508>
   14860: 63 10 b7 06  	bne	a4, a1, 0x148c0 <_ldtoa_r+0x564>
   14864: 85 07        	addi	a5, a5, 1
   14866: cd bf        	j	0x14858 <_ldtoa_r+0x4fc>
   14868: 83 56 21 11  	lhu	a3, 274(sp)
   1486c: b6 e0        	sd	a3, 64(sp)
   1486e: 93 d7 f6 00  	srli	a5, a3, 15
   14872: 63 9b 07 5e  	bnez	a5, 0x14e68 <_ldtoa_r+0xb0c>
   14876: f9 67        	lui	a5, 30
   14878: 93 8d 87 cd  	addi	s11, a5, -808
   1487c: 13 09 01 12  	addi	s2, sp, 288
   14880: 07 dc 0d 02  	<unknown>
   14884: 33 86 ed 00  	add	a2, s11, a4
   14888: 27 5c 09 02  	<unknown>
   1488c: 4a 97        	add	a4, a4, s2
   1488e: b2 f0        	sd	a2, 96(sp)
   14890: ba e8        	sd	a4, 80(sp)
   14892: e3 8a 06 c2  	beqz	a3, 0x144c6 <_ldtoa_r+0x16a>
   14896: 21 67        	lui	a4, 8
   14898: 7d 17        	addi	a4, a4, -1
   1489a: 63 86 e6 56  	beq	a3, a4, 0x14e06 <_ldtoa_r+0xaaa>
   1489e: 82 e0        	sd	zero, 64(sp)
   148a0: 03 17 01 11  	lh	a4, 272(sp)
   148a4: 63 4b 07 1a  	bltz	a4, 0x14a5a <_ldtoa_r+0x6fe>
   148a8: 93 0b 01 1e  	addi	s7, sp, 480
   148ac: f9 65        	lui	a1, 30
   148ae: 93 85 05 a1  	addi	a1, a1, -1520
   148b2: 5e 85        	mv	a0, s7
   148b4: ef 20 20 31  	jal	0x16bc6 <sprintf>
   148b8: 89 64        	lui	s1, 2
   148ba: 93 84 f4 70  	addi	s1, s1, 1807
   148be: a1 a2        	j	0x14a06 <_ldtoa_r+0x6aa>
   148c0: 5e 86        	mv	a2, s7
   148c2: 03 c7 07 00  	lbu	a4, 0(a5)
   148c6: 32 84        	mv	s0, a2
   148c8: 85 07        	addi	a5, a5, 1
   148ca: 23 00 e6 00  	sb	a4, 0(a2)
   148ce: 05 06        	addi	a2, a2, 1
   148d0: 6d fb        	bnez	a4, 0x148c2 <_ldtoa_r+0x566>
   148d2: 22 67        	ld	a4, 8(sp)
   148d4: 89 47        	addi	a5, zero, 2
   148d6: 03 46 f4 ff  	lbu	a2, -1(s0)
   148da: 63 07 f7 28  	beq	a4, a5, 0x14b68 <_ldtoa_r+0x80c>
   148de: 62 67        	ld	a4, 24(sp)
   148e0: a6 87        	mv	a5, s1
   148e2: 63 d3 e4 00  	bge	s1, a4, 0x148e8 <_ldtoa_r+0x58c>
   148e6: ba 87        	mv	a5, a4
   148e8: 93 05 00 03  	addi	a1, zero, 48
   148ec: 9b 86 07 00  	sext.w	a3, a5
   148f0: 63 14 b6 02  	bne	a2, a1, 0x14918 <_ldtoa_r+0x5bc>
   148f4: b3 07 74 41  	sub	a5, s0, s7
   148f8: 63 d0 f6 02  	bge	a3, a5, 0x14918 <_ldtoa_r+0x5bc>
   148fc: 13 06 00 03  	addi	a2, zero, 48
   14900: 29 a0        	j	0x1490a <_ldtoa_r+0x5ae>
   14902: b3 07 74 41  	sub	a5, s0, s7
   14906: 63 d9 f6 00  	bge	a3, a5, 0x14918 <_ldtoa_r+0x5bc>
   1490a: 83 47 e4 ff  	lbu	a5, -2(s0)
   1490e: a3 0f 04 fe  	sb	zero, -1(s0)
   14912: 7d 14        	addi	s0, s0, -1
   14914: e3 87 c7 fe  	beq	a5, a2, 0x14902 <_ldtoa_r+0x5a6>
   14918: 22 67        	ld	a4, 8(sp)
   1491a: 8d 47        	addi	a5, zero, 3
   1491c: 63 00 f7 22  	beq	a4, a5, 0x14b3c <_ldtoa_r+0x7e0>
   14920: 82 77        	ld	a5, 32(sp)
   14922: 7d 46        	addi	a2, zero, 31
   14924: 85 46        	addi	a3, zero, 1
   14926: 1b 85 97 00  	addiw	a0, a5, 9
   1492a: 91 47        	addi	a5, zero, 4
   1492c: 63 79 a6 22  	bgeu	a2, a0, 0x14b5e <_ldtoa_r+0x802>
   14930: 9b 97 17 00  	slliw	a5, a5, 1
   14934: 13 87 c7 01  	addi	a4, a5, 28
   14938: 9b 85 06 00  	sext.w	a1, a3
   1493c: 85 26        	addiw	a3, a3, 1
   1493e: e3 79 e5 fe  	bgeu	a0, a4, 0x14930 <_ldtoa_r+0x5d4>
   14942: c2 67        	ld	a5, 16(sp)
   14944: 81 25        	sext.w	a1, a1
   14946: ac d7        	sw	a1, 104(a5)
   14948: 42 65        	ld	a0, 16(sp)
   1494a: ef 10 60 4a  	jal	0x15df0 <_Balloc>
   1494e: aa 84        	mv	s1, a0
   14950: e3 01 05 1e  	beqz	a0, 0x15332 <_ldtoa_r+0xfd6>
   14954: c2 67        	ld	a5, 16(sp)
   14956: de 85        	mv	a1, s7
   14958: a8 f3        	sd	a0, 96(a5)
   1495a: ef 20 a0 48  	jal	0x16de4 <strcpy>
   1495e: 26 67        	ld	a4, 72(sp)
   14960: 09 c7        	beqz	a4, 0x1496a <_ldtoa_r+0x60e>
   14962: b3 07 74 41  	sub	a5, s0, s7
   14966: a6 97        	add	a5, a5, s1
   14968: 1c e3        	sd	a5, 0(a4)
   1496a: 83 30 81 2d  	ld	ra, 728(sp)
   1496e: 03 34 01 2d  	ld	s0, 720(sp)
   14972: 03 39 01 2c  	ld	s2, 704(sp)
   14976: 83 39 81 2b  	ld	s3, 696(sp)
   1497a: 03 3a 01 2b  	ld	s4, 688(sp)
   1497e: 83 3a 81 2a  	ld	s5, 680(sp)
   14982: 03 3b 01 2a  	ld	s6, 672(sp)
   14986: 83 3b 81 29  	ld	s7, 664(sp)
   1498a: 03 3c 01 29  	ld	s8, 656(sp)
   1498e: 83 3c 81 28  	ld	s9, 648(sp)
   14992: 03 3d 01 28  	ld	s10, 640(sp)
   14996: 83 3d 81 27  	ld	s11, 632(sp)
   1499a: 26 85        	mv	a0, s1
   1499c: 83 34 81 2c  	ld	s1, 712(sp)
   149a0: 13 01 01 2e  	addi	sp, sp, 736
   149a4: 82 80        	ret
   149a6: 26 8a        	mv	s4, s1
   149a8: c6 6b        	ld	s7, 80(sp)
   149aa: 86 64        	ld	s1, 64(sp)
   149ac: 4e 86        	mv	a2, s3
   149ae: 11 45        	addi	a0, zero, 4
   149b0: 63 54 b5 04  	bge	a0, a1, 0x149f8 <_ldtoa_r+0x69c>
   149b4: 15 45        	addi	a0, zero, 5
   149b6: 63 89 a5 70  	beq	a1, a0, 0x150c8 <_ldtoa_r+0xd6c>
   149ba: 83 c5 e7 ff  	lbu	a1, -2(a5)
   149be: 13 85 e7 ff  	addi	a0, a5, -2
   149c2: 93 f5 f5 07  	andi	a1, a1, 127
   149c6: 63 44 06 02  	bltz	a2, 0x149ee <_ldtoa_r+0x692>
   149ca: 93 07 e0 02  	addi	a5, zero, 46
   149ce: 13 06 80 03  	addi	a2, zero, 56
   149d2: 13 08 00 03  	addi	a6, zero, 48
   149d6: 63 8e f5 62  	beq	a1, a5, 0x15012 <_ldtoa_r+0xcb6>
   149da: 63 78 b6 62  	bgeu	a2, a1, 0x1500a <_ldtoa_r+0xcae>
   149de: 83 45 f5 ff  	lbu	a1, -1(a0)
   149e2: 23 00 05 01  	sb	a6, 0(a0)
   149e6: 7d 15        	addi	a0, a0, -1
   149e8: 93 f5 f5 07  	andi	a1, a1, 127
   149ec: ed b7        	j	0x149d6 <_ldtoa_r+0x67a>
   149ee: 13 06 10 03  	addi	a2, zero, 49
   149f2: 23 8f c7 fe  	sb	a2, -2(a5)
   149f6: 85 24        	addiw	s1, s1, 1
   149f8: f9 65        	lui	a1, 30
   149fa: 26 86        	mv	a2, s1
   149fc: 93 85 85 a1  	addi	a1, a1, -1512
   14a00: 52 85        	mv	a0, s4
   14a02: ef 20 40 1c  	jal	0x16bc6 <sprintf>
   14a06: e2 77        	ld	a5, 56(sp)
   14a08: 22 77        	ld	a4, 40(sp)
   14a0a: 23 28 91 22  	sw	s1, 560(sp)
   14a0e: 23 22 f1 22  	sw	a5, 548(sp)
   14a12: a1 67        	lui	a5, 8
   14a14: fd 17        	addi	a5, a5, -1
   14a16: e3 0d f7 e0  	beq	a4, a5, 0x14830 <_ldtoa_r+0x4d4>
   14a1a: 42 77        	ld	a4, 48(sp)
   14a1c: 83 46 01 1e  	lbu	a3, 480(sp)
   14a20: 9b 87 14 00  	addiw	a5, s1, 1
   14a24: 1c c3        	sw	a5, 0(a4)
   14a26: 63 8d 06 42  	beqz	a3, 0x14e60 <_ldtoa_r+0xb04>
   14a2a: de 87        	mv	a5, s7
   14a2c: 13 06 e0 02  	addi	a2, zero, 46
   14a30: 63 87 c6 18  	beq	a3, a2, 0x14bbe <_ldtoa_r+0x862>
   14a34: 83 c6 17 00  	lbu	a3, 1(a5)
   14a38: 85 07        	addi	a5, a5, 1
   14a3a: fd fa        	bnez	a3, 0x14a30 <_ldtoa_r+0x6d4>
   14a3c: 13 06 50 04  	addi	a2, zero, 69
   14a40: 63 e5 fb 00  	bltu	s7, a5, 0x14a4a <_ldtoa_r+0x6ee>
   14a44: 01 a8        	j	0x14a54 <_ldtoa_r+0x6f8>
   14a46: 63 8d 77 41  	beq	a5, s7, 0x14e60 <_ldtoa_r+0xb04>
   14a4a: 03 c7 f7 ff  	lbu	a4, -1(a5)
   14a4e: fd 17        	addi	a5, a5, -1
   14a50: e3 1b c7 fe  	bne	a4, a2, 0x14a46 <_ldtoa_r+0x6ea>
   14a54: 23 80 07 00  	sb	zero, 0(a5)
   14a58: dd bb        	j	0x1484e <_ldtoa_r+0x4f2>
   14a5a: a2 85        	mv	a1, s0
   14a5c: 13 85 87 cd  	addi	a0, a5, -808
   14a60: be ec        	sd	a5, 88(sp)
   14a62: ef e0 ff 87  	jal	0x132e0 <ecmp>
   14a66: e3 0a 05 a6  	beqz	a0, 0x144da <_ldtoa_r+0x17e>
   14a6a: e6 67        	ld	a5, 88(sp)
   14a6c: 63 4e 05 18  	bltz	a0, 0x14c08 <_ldtoa_r+0x8ac>
   14a70: 03 57 21 11  	lhu	a4, 274(sp)
   14a74: 63 1f 07 6c  	bnez	a4, 0x15152 <_ldtoa_r+0xdf6>
   14a78: 03 17 01 11  	lh	a4, 272(sp)
   14a7c: 81 44        	mv	s1, zero
   14a7e: e3 43 07 00  	bltz	a4, 0x15284 <_ldtoa_r+0xf28>
   14a82: 93 09 01 22  	addi	s3, sp, 544
   14a86: f9 6c        	lui	s9, 30
   14a88: 3e 8a        	mv	s4, a5
   14a8a: ce 86        	mv	a3, s3
   14a8c: 22 86        	mv	a2, s0
   14a8e: a2 85        	mv	a1, s0
   14a90: 13 85 0c cc  	addi	a0, s9, -832
   14a94: ef f0 ef ca  	jal	0x13f42 <emul>
   14a98: 83 17 01 11  	lh	a5, 272(sp)
   14a9c: fd 34        	addiw	s1, s1, -1
   14a9e: e3 d6 07 fe  	bgez	a5, 0x14a8a <_ldtoa_r+0x72e>
   14aa2: d2 87        	mv	a5, s4
   14aa4: 57 77 86 c4  	<unknown>
   14aa8: 13 0a 01 16  	addi	s4, sp, 352
   14aac: ba ec        	sd	a4, 88(sp)
   14aae: 57 70 85 c4  	<unknown>
   14ab2: 07 5c 04 02  	<unknown>
   14ab6: 27 5c 0a 02  	<unknown>
   14aba: 63 14 0d 58  	bnez	s10, 0x15042 <_ldtoa_r+0xce6>
   14abe: 07 dc 0d 02  	<unknown>
   14ac2: 27 5c 09 02  	<unknown>
   14ac6: 79 6d        	lui	s10, 30
   14ac8: f9 6c        	lui	s9, 30
   14aca: f9 6d        	lui	s11, 30
   14acc: 7d 7b        	lui	s6, 1048575
   14ace: 13 0d 0d bd  	addi	s10, s10, -1072
   14ad2: 93 8c 8c ac  	addi	s9, s9, -1336
   14ad6: 93 8d 8d bb  	addi	s11, s11, -1096
   14ada: be 8a        	mv	s5, a5
   14adc: 19 a0        	j	0x14ae2 <_ldtoa_r+0x786>
   14ade: d1 0c        	addi	s9, s9, 20
   14ae0: 51 0d        	addi	s10, s10, 20
   14ae2: d2 85        	mv	a1, s4
   14ae4: 13 85 8a cd  	addi	a0, s5, -808
   14ae8: ef e0 8f ff  	jal	0x132e0 <ecmp>
   14aec: 63 5e a0 02  	blez	a0, 0x14b28 <_ldtoa_r+0x7cc>
   14af0: d2 85        	mv	a1, s4
   14af2: 66 85        	mv	a0, s9
   14af4: ef e0 cf fe  	jal	0x132e0 <ecmp>
   14af8: 63 40 05 02  	bltz	a0, 0x14b18 <_ldtoa_r+0x7bc>
   14afc: ce 86        	mv	a3, s3
   14afe: 52 86        	mv	a2, s4
   14b00: d2 85        	mv	a1, s4
   14b02: 6a 85        	mv	a0, s10
   14b04: ef f0 ef c3  	jal	0x13f42 <emul>
   14b08: ce 86        	mv	a3, s3
   14b0a: 4a 86        	mv	a2, s2
   14b0c: ca 85        	mv	a1, s2
   14b0e: 6a 85        	mv	a0, s10
   14b10: ef f0 2f c3  	jal	0x13f42 <emul>
   14b14: bb 04 9b 00  	addw	s1, s6, s1
   14b18: 9b 57 fb 01  	srliw	a5, s6, 31
   14b1c: bb 87 67 01  	addw	a5, a5, s6
   14b20: 1b db 17 40  	sraiw	s6, a5, 1
   14b24: e3 9d bc fb  	bne	s9, s11, 0x14ade <_ldtoa_r+0x782>
   14b28: ce 86        	mv	a3, s3
   14b2a: 4a 86        	mv	a2, s2
   14b2c: 93 85 8a cd  	addi	a1, s5, -808
   14b30: 4a 85        	mv	a0, s2
   14b32: ef e0 3f f1  	jal	0x13a44 <ediv>
   14b36: 93 0b 01 1e  	addi	s7, sp, 480
   14b3a: 55 ba        	j	0x144ee <_ldtoa_r+0x192>
   14b3c: e2 67        	ld	a5, 24(sp)
   14b3e: bd 9c        	addw	s1, s1, a5
   14b40: 63 ce 04 4a  	bltz	s1, 0x14ffc <_ldtoa_r+0xca0>
   14b44: c2 77        	ld	a5, 48(sp)
   14b46: 02 77        	ld	a4, 32(sp)
   14b48: 9c 43        	lw	a5, 0(a5)
   14b4a: b9 9f        	addw	a5, a5, a4
   14b4c: 3e f0        	sd	a5, 32(sp)
   14b4e: 82 77        	ld	a5, 32(sp)
   14b50: 7d 46        	addi	a2, zero, 31
   14b52: 85 46        	addi	a3, zero, 1
   14b54: 1b 85 37 00  	addiw	a0, a5, 3
   14b58: 91 47        	addi	a5, zero, 4
   14b5a: e3 6b a6 dc  	bltu	a2, a0, 0x14930 <_ldtoa_r+0x5d4>
   14b5e: c2 67        	ld	a5, 16(sp)
   14b60: 81 45        	mv	a1, zero
   14b62: 23 a4 07 06  	sw	zero, 104(a5)
   14b66: cd b3        	j	0x14948 <_ldtoa_r+0x5ec>
   14b68: 93 07 00 03  	addi	a5, zero, 48
   14b6c: e3 1a f6 da  	bne	a2, a5, 0x14920 <_ldtoa_r+0x5c4>
   14b70: b3 07 74 41  	sub	a5, s0, s7
   14b74: 85 46        	addi	a3, zero, 1
   14b76: e3 c3 f6 d8  	blt	a3, a5, 0x148fc <_ldtoa_r+0x5a0>
   14b7a: 5d b3        	j	0x14920 <_ldtoa_r+0x5c4>
   14b7c: c6 67        	ld	a5, 80(sp)
   14b7e: 73 2b 20 c2  	csrr	s6, vlenb
   14b82: 52 9b        	add	s6, s6, s4
   14b84: 57 f0 8d 04  	<unknown>
   14b88: d2 85        	mv	a1, s4
   14b8a: 07 5c 0b 02  	<unknown>
   14b8e: 22 85        	mv	a0, s0
   14b90: 27 dc 07 02  	<unknown>
   14b94: 23 1c 01 12  	sh	zero, 312(sp)
   14b98: ef e0 0f e9  	jal	0x13228 <emovi>
   14b9c: 57 70 86 c4  	<unknown>
   14ba0: 07 5c 0a 02  	<unknown>
   14ba4: 27 5c 04 02  	<unknown>
   14ba8: 57 f0 8d 04  	<unknown>
   14bac: 07 5c 0b 02  	<unknown>
   14bb0: 27 5c 0c 02  	<unknown>
   14bb4: 8d ba        	j	0x14526 <_ldtoa_r+0x1ca>
   14bb6: c6 66        	ld	a3, 80(sp)
   14bb8: 55 b4        	j	0x1465c <_ldtoa_r+0x300>
   14bba: 82 77        	ld	a5, 32(sp)
   14bbc: ad b0        	j	0x14426 <_ldtoa_r+0xca>
   14bbe: 83 c6 07 00  	lbu	a3, 0(a5)
   14bc2: e3 8d 06 e6  	beqz	a3, 0x14a3c <_ldtoa_r+0x6e0>
   14bc6: 03 c7 17 00  	lbu	a4, 1(a5)
   14bca: 85 07        	addi	a5, a5, 1
   14bcc: a3 8f e7 fe  	sb	a4, -1(a5)
   14bd0: e3 06 07 e6  	beqz	a4, 0x14a3c <_ldtoa_r+0x6e0>
   14bd4: 03 c7 17 00  	lbu	a4, 1(a5)
   14bd8: 85 07        	addi	a5, a5, 1
   14bda: a3 8f e7 fe  	sb	a4, -1(a5)
   14bde: 65 f7        	bnez	a4, 0x14bc6 <_ldtoa_r+0x86a>
   14be0: b1 bd        	j	0x14a3c <_ldtoa_r+0x6e0>
   14be2: 93 06 10 03  	addi	a3, zero, 49
   14be6: a3 00 d1 1e  	sb	a3, 481(sp)
   14bea: 93 06 e0 02  	addi	a3, zero, 46
   14bee: 23 01 d1 1e  	sb	a3, 482(sp)
   14bf2: 85 24        	addiw	s1, s1, 1
   14bf4: 63 53 f0 20  	blez	a5, 0x14dfa <_ldtoa_r+0xa9e>
   14bf8: 93 07 00 03  	addi	a5, zero, 48
   14bfc: a3 01 f1 1e  	sb	a5, 483(sp)
   14c00: 7d 36        	addiw	a2, a2, -1
   14c02: 13 0a 41 1e  	addi	s4, sp, 484
   14c06: 45 b4        	j	0x146a6 <_ldtoa_r+0x34a>
   14c08: 93 0a 01 14  	addi	s5, sp, 320
   14c0c: 57 70 85 c4  	<unknown>
   14c10: f3 2e 20 c2  	csrr	t4, vlenb
   14c14: 07 5c 04 02  	<unknown>
   14c18: 27 dc 0a 02  	<unknown>
   14c1c: d6 9e        	add	t4, t4, s5
   14c1e: 63 1f 0d 50  	bnez	s10, 0x1513c <_ldtoa_r+0xde0>
   14c22: f3 26 20 c2  	csrr	a3, vlenb
   14c26: 13 0a 01 16  	addi	s4, sp, 352
   14c2a: 33 07 da 00  	add	a4, s4, a3
   14c2e: 93 0b 01 1e  	addi	s7, sp, 480
   14c32: 3a e5        	sd	a4, 136(sp)
   14c34: 33 87 db 00  	add	a4, s7, a3
   14c38: 3a e1        	sd	a4, 128(sp)
   14c3a: 98 03        	addi	a4, sp, 448
   14c3c: 36 97        	add	a4, a4, a3
   14c3e: 3a f5        	sd	a4, 168(sp)
   14c40: 18 03        	addi	a4, sp, 384
   14c42: 36 97        	add	a4, a4, a3
   14c44: 3a f1        	sd	a4, 160(sp)
   14c46: 18 13        	addi	a4, sp, 416
   14c48: 36 97        	add	a4, a4, a3
   14c4a: f9 66        	lui	a3, 30
   14c4c: 93 86 46 cd  	addi	a3, a3, -812
   14c50: b6 f4        	sd	a3, 104(sp)
   14c52: f9 66        	lui	a3, 30
   14c54: 93 86 06 aa  	addi	a3, a3, -1376
   14c58: 36 e9        	sd	a3, 144(sp)
   14c5a: a1 66        	lui	a3, 8
   14c5c: fd 16        	addi	a3, a3, -1
   14c5e: 11 66        	lui	a2, 4
   14c60: 79 6e        	lui	t3, 30
   14c62: b6 f8        	sd	a3, 112(sp)
   14c64: f1 76        	lui	a3, 1048572
   14c66: 81 44        	mv	s1, zero
   14c68: 13 0b 0e c7  	addi	s6, t3, -912
   14c6c: 93 09 01 22  	addi	s3, sp, 544
   14c70: 89 26        	addiw	a3, a3, 2
   14c72: 13 06 e6 08  	addi	a2, a2, 142
   14c76: 3a ed        	sd	a4, 152(sp)
   14c78: 41 47        	addi	a4, zero, 16
   14c7a: b6 de        	sw	a3, 124(sp)
   14c7c: 62 f9        	sd	s8, 176(sp)
   14c7e: 6e fd        	sd	s11, 184(sp)
   14c80: 4a 8c        	mv	s8, s2
   14c82: d2 8d        	mv	s11, s4
   14c84: 26 89        	mv	s2, s1
   14c86: 4e 8a        	mv	s4, s3
   14c88: d7 76 86 c4  	<unknown>
   14c8c: ea 89        	mv	s3, s10
   14c8e: 23 19 c1 14  	sh	a2, 338(sp)
   14c92: 5a 8d        	mv	s10, s6
   14c94: 93 0c 21 13  	addi	s9, sp, 306
   14c98: b6 ec        	sd	a3, 88(sp)
   14c9a: be e1        	sd	a5, 192(sp)
   14c9c: ba 84        	mv	s1, a4
   14c9e: 76 8b        	mv	s6, t4
   14ca0: d2 86        	mv	a3, s4
   14ca2: 62 86        	mv	a2, s8
   14ca4: d6 85        	mv	a1, s5
   14ca6: 6a 85        	mv	a0, s10
   14ca8: ef e0 df d9  	jal	0x13a44 <ediv>
   14cac: 57 70 85 c4  	<unknown>
   14cb0: 07 5c 0c 02  	<unknown>
   14cb4: 27 dc 0b 02  	<unknown>
   14cb8: 63 90 09 16  	bnez	s3, 0x14e18 <_ldtoa_r+0xabc>
   14cbc: c6 77        	ld	a5, 112(sp)
   14cbe: 03 57 21 1f  	lhu	a4, 498(sp)
   14cc2: b3 76 f7 00  	and	a3, a4, a5
   14cc6: f6 57        	lw	a5, 124(sp)
   14cc8: b5 9f        	addw	a5, a5, a3
   14cca: be 86        	mv	a3, a5
   14ccc: 63 5d f0 3a  	blez	a5, 0x15086 <_ldtoa_r+0xd2a>
   14cd0: 93 07 00 09  	addi	a5, zero, 144
   14cd4: 95 9f        	subw	a5, a5, a3
   14cd6: 07 dc 0b 02  	<unknown>
   14cda: 27 dc 0d 02  	<unknown>
   14cde: 63 55 f0 04  	blez	a5, 0x14d28 <_ldtoa_r+0x9cc>
   14ce2: bd 46        	addi	a3, zero, 15
   14ce4: 63 db f6 56  	bge	a3, a5, 0x1525a <_ldtoa_r+0xefe>
   14ce8: c1 37        	addiw	a5, a5, -16
   14cea: 1b d6 47 00  	srliw	a2, a5, 4
   14cee: 9b 05 16 00  	addiw	a1, a2, 1
   14cf2: 93 96 15 00  	slli	a3, a1, 1
   14cf6: 1b 16 46 00  	slliw	a2, a2, 4
   14cfa: 57 75 80 04  	<unknown>
   14cfe: ee 96        	add	a3, a3, s11
   14d00: 91 9f        	subw	a5, a5, a2
   14d02: 57 3c 00 5e  	<unknown>
   14d06: 57 f0 85 04  	<unknown>
   14d0a: 27 dc 0d 02  	<unknown>
   14d0e: 4a 66        	ld	a2, 144(sp)
   14d10: 86 07        	slli	a5, a5, 1
   14d12: b2 97        	add	a5, a5, a2
   14d14: 83 d7 07 00  	lhu	a5, 0(a5)
   14d18: 03 d6 06 00  	lhu	a2, 0(a3)
   14d1c: f1 8f        	and	a5, a5, a2
   14d1e: 23 90 f6 00  	sh	a5, 0(a3)
   14d22: 3d 83        	srli	a4, a4, 15
   14d24: 63 10 07 1c  	bnez	a4, 0x14ee4 <_ldtoa_r+0xb88>
   14d28: e2 87        	mv	a5, s8
   14d2a: 6e 87        	mv	a4, s11
   14d2c: 03 d6 07 00  	lhu	a2, 0(a5)
   14d30: 83 56 07 00  	lhu	a3, 0(a4)
   14d34: 63 17 d6 02  	bne	a2, a3, 0x14d62 <_ldtoa_r+0xa06>
   14d38: 89 07        	addi	a5, a5, 2
   14d3a: 09 07        	addi	a4, a4, 2
   14d3c: e3 98 97 ff  	bne	a5, s9, 0x14d2c <_ldtoa_r+0x9d0>
   14d40: 57 70 85 c4  	<unknown>
   14d44: 07 5c 0c 02  	<unknown>
   14d48: 27 dc 0a 02  	<unknown>
   14d4c: 63 89 09 00  	beqz	s3, 0x14d5e <_ldtoa_r+0xa02>
   14d50: c6 67        	ld	a5, 80(sp)
   14d52: 57 f0 89 04  	<unknown>
   14d56: 07 dc 07 02  	<unknown>
   14d5a: 27 5c 0b 02  	<unknown>
   14d5e: 3b 89 24 01  	addw	s2, s1, s2
   14d62: a6 77        	ld	a5, 104(sp)
   14d64: 51 0d        	addi	s10, s10, 20
   14d66: 85 80        	srli	s1, s1, 1
   14d68: e3 1c fd f2  	bne	s10, a5, 0x14ca0 <_ldtoa_r+0x944>
   14d6c: 83 57 21 15  	lhu	a5, 338(sp)
   14d70: 83 56 21 11  	lhu	a3, 274(sp)
   14d74: 57 70 85 c4  	<unknown>
   14d78: ca 84        	mv	s1, s2
   14d7a: b5 9f        	addw	a5, a5, a3
   14d7c: f1 76        	lui	a3, 1048572
   14d7e: 9b 86 26 f7  	addiw	a3, a3, -142
   14d82: b5 9f        	addw	a5, a5, a3
   14d84: 23 19 f1 14  	sh	a5, 338(sp)
   14d88: 4e 8d        	mv	s10, s3
   14d8a: 62 89        	mv	s2, s8
   14d8c: d2 89        	mv	s3, s4
   14d8e: 4a 7c        	ld	s8, 176(sp)
   14d90: 6e 8a        	mv	s4, s11
   14d92: 07 dc 0a 02  	<unknown>
   14d96: ea 7d        	ld	s11, 184(sp)
   14d98: 27 5c 04 02  	<unknown>
   14d9c: 63 1b 0d 36  	bnez	s10, 0x15112 <_ldtoa_r+0xdb6>
   14da0: 07 dc 0d 02  	<unknown>
   14da4: 27 5c 09 02  	<unknown>
   14da8: 79 6b        	lui	s6, 30
   14daa: 05 6d        	lui	s10, 1
   14dac: 13 0b 0b bd  	addi	s6, s6, -1072
   14db0: f9 6c        	lui	s9, 30
   14db2: 01 a8        	j	0x14dc2 <_ldtoa_r+0xa66>
   14db4: 93 87 0c cc  	addi	a5, s9, -832
   14db8: 13 5d 1d 00  	srli	s10, s10, 1
   14dbc: 63 09 fb f2  	beq	s6, a5, 0x144ee <_ldtoa_r+0x192>
   14dc0: 51 0b        	addi	s6, s6, 20
   14dc2: d6 85        	mv	a1, s5
   14dc4: 13 85 0c cc  	addi	a0, s9, -832
   14dc8: ef e0 8f d1  	jal	0x132e0 <ecmp>
   14dcc: 63 41 a0 f2  	bgtz	a0, 0x144ee <_ldtoa_r+0x192>
   14dd0: d6 85        	mv	a1, s5
   14dd2: 5a 85        	mv	a0, s6
   14dd4: ef e0 cf d0  	jal	0x132e0 <ecmp>
   14dd8: e3 4e a0 fc  	bgtz	a0, 0x14db4 <_ldtoa_r+0xa58>
   14ddc: ce 86        	mv	a3, s3
   14dde: 56 86        	mv	a2, s5
   14de0: d6 85        	mv	a1, s5
   14de2: 5a 85        	mv	a0, s6
   14de4: ef e0 1f c6  	jal	0x13a44 <ediv>
   14de8: ce 86        	mv	a3, s3
   14dea: 4a 86        	mv	a2, s2
   14dec: ca 85        	mv	a1, s2
   14dee: 5a 85        	mv	a0, s6
   14df0: ef f0 2f 95  	jal	0x13f42 <emul>
   14df4: bb 04 9d 00  	addw	s1, s10, s1
   14df8: 75 bf        	j	0x14db4 <_ldtoa_r+0xa58>
   14dfa: 13 0a 31 1e  	addi	s4, sp, 483
   14dfe: 01 46        	mv	a2, zero
   14e00: e3 9c 07 be  	bnez	a5, 0x149f8 <_ldtoa_r+0x69c>
   14e04: 4d b0        	j	0x146a6 <_ldtoa_r+0x34a>
   14e06: 93 0b 01 1e  	addi	s7, sp, 480
   14e0a: f9 65        	lui	a1, 30
   14e0c: 93 85 05 a0  	addi	a1, a1, -1536
   14e10: 5e 85        	mv	a0, s7
   14e12: ef 10 50 5b  	jal	0x16bc6 <sprintf>
   14e16: 4d b4        	j	0x148b8 <_ldtoa_r+0x55c>
   14e18: c6 67        	ld	a5, 80(sp)
   14e1a: 57 f0 89 04  	<unknown>
   14e1e: 07 dc 07 02  	<unknown>
   14e22: 8a 67        	ld	a5, 128(sp)
   14e24: 27 dc 07 02  	<unknown>
   14e28: c6 77        	ld	a5, 112(sp)
   14e2a: 03 57 21 1f  	lhu	a4, 498(sp)
   14e2e: b3 76 f7 00  	and	a3, a4, a5
   14e32: f6 57        	lw	a5, 124(sp)
   14e34: b5 9f        	addw	a5, a5, a3
   14e36: be 86        	mv	a3, a5
   14e38: 63 50 f0 26  	blez	a5, 0x15098 <_ldtoa_r+0xd3c>
   14e3c: 93 07 00 09  	addi	a5, zero, 144
   14e40: 57 70 85 c4  	<unknown>
   14e44: 95 9f        	subw	a5, a5, a3
   14e46: 07 dc 0b 02  	<unknown>
   14e4a: 27 dc 0d 02  	<unknown>
   14e4e: 8a 66        	ld	a3, 128(sp)
   14e50: 57 f0 89 04  	<unknown>
   14e54: 07 dc 06 02  	<unknown>
   14e58: aa 66        	ld	a3, 136(sp)
   14e5a: 27 dc 06 02  	<unknown>
   14e5e: 41 b5        	j	0x14cde <_ldtoa_r+0x982>
   14e60: de 87        	mv	a5, s7
   14e62: 23 80 07 00  	sb	zero, 0(a5)
   14e66: e5 b2        	j	0x1484e <_ldtoa_r+0x4f2>
   14e68: 86 67        	ld	a5, 64(sp)
   14e6a: a1 66        	lui	a3, 8
   14e6c: fd 16        	addi	a3, a3, -1
   14e6e: 33 f6 d7 00  	and	a2, a5, a3
   14e72: f9 67        	lui	a5, 30
   14e74: 23 19 c1 10  	sh	a2, 274(sp)
   14e78: 93 8d 87 cd  	addi	s11, a5, -808
   14e7c: 13 09 01 12  	addi	s2, sp, 288
   14e80: 07 dc 0d 02  	<unknown>
   14e84: b3 85 ed 00  	add	a1, s11, a4
   14e88: 27 5c 09 02  	<unknown>
   14e8c: 4a 97        	add	a4, a4, s2
   14e8e: ae f0        	sd	a1, 96(sp)
   14e90: ba e8        	sd	a4, 80(sp)
   14e92: 21 c6        	beqz	a2, 0x14eda <_ldtoa_r+0xb7e>
   14e94: 63 00 d6 22  	beq	a2, a3, 0x150b4 <_ldtoa_r+0xd58>
   14e98: 41 67        	lui	a4, 16
   14e9a: 7d 17        	addi	a4, a4, -1
   14e9c: ba e0        	sd	a4, 64(sp)
   14e9e: 09 b4        	j	0x148a0 <_ldtoa_r+0x544>
   14ea0: 06 66        	ld	a2, 64(sp)
   14ea2: a1 66        	lui	a3, 8
   14ea4: f9 67        	lui	a5, 30
   14ea6: 93 8d 87 cd  	addi	s11, a5, -808
   14eaa: 13 09 01 12  	addi	s2, sp, 288
   14eae: fd 16        	addi	a3, a3, -1
   14eb0: b3 85 ed 00  	add	a1, s11, a4
   14eb4: 75 8e        	and	a2, a2, a3
   14eb6: 4a 97        	add	a4, a4, s2
   14eb8: 23 19 c1 10  	sh	a2, 274(sp)
   14ebc: ae f0        	sd	a1, 96(sp)
   14ebe: ba e8        	sd	a4, 80(sp)
   14ec0: 57 70 85 c4  	<unknown>
   14ec4: 07 dc 0d 02  	<unknown>
   14ec8: 27 5c 09 02  	<unknown>
   14ecc: 57 70 8d 04  	<unknown>
   14ed0: 07 dc 05 02  	<unknown>
   14ed4: 27 5c 07 02  	<unknown>
   14ed8: 55 fe        	bnez	a2, 0x14e94 <_ldtoa_r+0xb38>
   14eda: 41 67        	lui	a4, 16
   14edc: 7d 17        	addi	a4, a4, -1
   14ede: ba e0        	sd	a4, 64(sp)
   14ee0: 6f f0 6f de  	j	0x144c6 <_ldtoa_r+0x16a>
   14ee4: de 87        	mv	a5, s7
   14ee6: 6e 87        	mv	a4, s11
   14ee8: 39 a0        	j	0x14ef6 <_ldtoa_r+0xb9a>
   14eea: 89 07        	addi	a5, a5, 2
   14eec: 93 06 21 1f  	addi	a3, sp, 498
   14ef0: 09 07        	addi	a4, a4, 2
   14ef2: e3 8b f6 e2  	beq	a3, a5, 0x14d28 <_ldtoa_r+0x9cc>
   14ef6: 03 d6 07 00  	lhu	a2, 0(a5)
   14efa: 83 56 07 00  	lhu	a3, 0(a4)
   14efe: e3 06 d6 fe  	beq	a2, a3, 0x14eea <_ldtoa_r+0xb8e>
   14f02: 83 57 21 17  	lhu	a5, 370(sp)
   14f06: 93 c7 f7 ff  	not	a5, a5
   14f0a: 13 97 17 03  	slli	a4, a5, 49
   14f0e: 63 08 07 34  	beqz	a4, 0x1525e <_ldtoa_r+0xf02>
   14f12: 8e 67        	ld	a5, 192(sp)
   14f14: 0c 03        	addi	a1, sp, 384
   14f16: 13 85 87 cd  	addi	a0, a5, -808
   14f1a: ef e0 ef b0  	jal	0x13228 <emovi>
   14f1e: 0c 13        	addi	a1, sp, 416
   14f20: 6e 85        	mv	a0, s11
   14f22: ef e0 6f b0  	jal	0x13228 <emovi>
   14f26: 83 57 01 18  	lhu	a5, 384(sp)
   14f2a: 03 57 21 1a  	lhu	a4, 418(sp)
   14f2e: 03 56 21 18  	lhu	a2, 386(sp)
   14f32: 93 c7 f7 ff  	not	a5, a5
   14f36: 23 10 f1 18  	sh	a5, 384(sp)
   14f3a: b3 05 e6 40  	sub	a1, a2, a4
   14f3e: ba 86        	mv	a3, a4
   14f40: 63 57 b0 04  	blez	a1, 0x14f8e <_ldtoa_r+0xc32>
   14f44: 1c 13        	addi	a5, sp, 416
   14f46: 57 70 86 c4  	<unknown>
   14f4a: 07 dc 07 02  	<unknown>
   14f4e: 9c 03        	addi	a5, sp, 448
   14f50: 27 dc 07 02  	<unknown>
   14f54: e6 65        	ld	a1, 88(sp)
   14f56: b1 46        	addi	a3, zero, 12
   14f58: b3 87 b6 40  	sub	a5, a3, a1
   14f5c: 63 91 b6 38  	bne	a3, a1, 0x152de <_ldtoa_r+0xf82>
   14f60: 23 1c 01 1c  	sh	zero, 472(sp)
   14f64: 1c 03        	addi	a5, sp, 384
   14f66: 07 dc 07 02  	<unknown>
   14f6a: 1c 13        	addi	a5, sp, 416
   14f6c: 27 dc 07 02  	<unknown>
   14f70: 23 1c 01 1a  	sh	zero, 440(sp)
   14f74: 9c 03        	addi	a5, sp, 448
   14f76: 07 dc 07 02  	<unknown>
   14f7a: 1c 03        	addi	a5, sp, 384
   14f7c: 27 dc 07 02  	<unknown>
   14f80: 83 56 21 1a  	lhu	a3, 418(sp)
   14f84: b3 05 c7 40  	sub	a1, a4, a2
   14f88: 23 1c 01 18  	sh	zero, 408(sp)
   14f8c: 36 87        	mv	a4, a3
   14f8e: 63 84 05 30  	beqz	a1, 0x15296 <_ldtoa_r+0xf3a>
   14f92: b6 e5        	sd	a3, 200(sp)
   14f94: 93 07 f0 f6  	addi	a5, zero, -145
   14f98: 63 cd f5 04  	blt	a1, a5, 0x14ff2 <_ldtoa_r+0xc96>
   14f9c: 81 25        	sext.w	a1, a1
   14f9e: 08 03        	addi	a0, sp, 384
   14fa0: ef e0 ef 91  	jal	0x130be <eshift.part.0>
   14fa4: ae 66        	ld	a3, 200(sp)
   14fa6: aa 85        	mv	a1, a0
   14fa8: 03 57 01 18  	lhu	a4, 384(sp)
   14fac: 83 57 01 1a  	lhu	a5, 416(sp)
   14fb0: 63 0d f7 38  	beq	a4, a5, 0x1534a <_ldtoa_r+0xfee>
   14fb4: 81 47        	mv	a5, zero
   14fb6: 93 0e 81 19  	addi	t4, sp, 408
   14fba: 28 1b        	addi	a0, sp, 440
   14fbc: 13 06 21 1a  	addi	a2, sp, 418
   14fc0: 03 57 05 00  	lhu	a4, 0(a0)
   14fc4: 03 d8 0e 00  	lhu	a6, 0(t4)
   14fc8: 79 15        	addi	a0, a0, -2
   14fca: f9 1e        	addi	t4, t4, -2
   14fcc: 33 07 07 41  	sub	a4, a4, a6
   14fd0: b3 07 f7 40  	sub	a5, a4, a5
   14fd4: 13 d7 07 01  	srli	a4, a5, 16
   14fd8: 23 11 f5 00  	sh	a5, 2(a0)
   14fdc: 93 77 17 00  	andi	a5, a4, 1
   14fe0: e3 10 c5 fe  	bne	a0, a2, 0x14fc0 <_ldtoa_r+0xc64>
   14fe4: 05 46        	addi	a2, zero, 1
   14fe6: d2 87        	mv	a5, s4
   14fe8: 13 07 00 04  	addi	a4, zero, 64
   14fec: 08 13        	addi	a0, sp, 416
   14fee: ef e0 af bd  	jal	0x133c8 <emdnorm>
   14ff2: ee 85        	mv	a1, s11
   14ff4: 08 13        	addi	a0, sp, 416
   14ff6: ef d0 1f ef  	jal	0x12ee6 <emovo.constprop.0>
   14ffa: 3d b3        	j	0x14d28 <_ldtoa_r+0x9cc>
   14ffc: c2 77        	ld	a5, 48(sp)
   14ffe: 23 00 01 1e  	sb	zero, 480(sp)
   15002: 5e 84        	mv	s0, s7
   15004: 23 a0 07 00  	sw	zero, 0(a5)
   15008: 99 b6        	j	0x14b4e <_ldtoa_r+0x7f2>
   1500a: 85 25        	addiw	a1, a1, 1
   1500c: 23 00 b5 00  	sb	a1, 0(a0)
   15010: e5 b2        	j	0x149f8 <_ldtoa_r+0x69c>
   15012: 83 47 f5 ff  	lbu	a5, -1(a0)
   15016: 13 06 80 03  	addi	a2, zero, 56
   1501a: 63 78 f6 12  	bgeu	a2, a5, 0x1514a <_ldtoa_r+0xdee>
   1501e: 93 07 10 03  	addi	a5, zero, 49
   15022: 85 24        	addiw	s1, s1, 1
   15024: a3 0f f5 fe  	sb	a5, -1(a0)
   15028: c1 ba        	j	0x149f8 <_ldtoa_r+0x69c>
   1502a: 9c 11        	addi	a5, sp, 224
   1502c: 13 06 21 0f  	addi	a2, sp, 242
   15030: 19 a0        	j	0x15036 <_ldtoa_r+0xcda>
   15032: e3 84 c7 9e  	beq	a5, a2, 0x14a1a <_ldtoa_r+0x6be>
   15036: 03 d7 07 00  	lhu	a4, 0(a5)
   1503a: 89 07        	addi	a5, a5, 2
   1503c: 7d db        	beqz	a4, 0x15032 <_ldtoa_r+0xcd6>
   1503e: 6f f0 7f 80  	j	0x14844 <_ldtoa_r+0x4e8>
   15042: 73 27 20 c2  	csrr	a4, vlenb
   15046: 13 07 07 09  	addi	a4, a4, 144
   1504a: 94 09        	addi	a3, sp, 208
   1504c: 57 70 8d 04  	<unknown>
   15050: 36 97        	add	a4, a4, a3
   15052: 07 5c 0c 02  	<unknown>
   15056: 27 5c 07 02  	<unknown>
   1505a: 57 70 85 c4  	<unknown>
   1505e: 07 dc 0d 02  	<unknown>
   15062: 27 5c 09 02  	<unknown>
   15066: 06 77        	ld	a4, 96(sp)
   15068: 57 70 8d 04  	<unknown>
   1506c: 07 5c 07 02  	<unknown>
   15070: 46 67        	ld	a4, 80(sp)
   15072: 27 5c 07 02  	<unknown>
   15076: 81 bc        	j	0x14ac6 <_ldtoa_r+0x76a>
   15078: 83 55 c1 26  	lhu	a1, 620(sp)
   1507c: 13 0a 21 1e  	addi	s4, sp, 482
   15080: 93 07 31 1e  	addi	a5, sp, 483
   15084: 2d b2        	j	0x149ae <_ldtoa_r+0x652>
   15086: d7 77 80 04  	<unknown>
   1508a: 57 3c 00 5e  	<unknown>
   1508e: 57 70 85 c4  	<unknown>
   15092: 27 dc 0d 02  	<unknown>
   15096: 71 b1        	j	0x14d22 <_ldtoa_r+0x9c6>
   15098: d7 77 80 04  	<unknown>
   1509c: 57 3c 00 5e  	<unknown>
   150a0: 57 70 85 c4  	<unknown>
   150a4: 27 dc 0d 02  	<unknown>
   150a8: aa 67        	ld	a5, 136(sp)
   150aa: 57 f0 89 04  	<unknown>
   150ae: 27 dc 07 02  	<unknown>
   150b2: 85 b9        	j	0x14d22 <_ldtoa_r+0x9c6>
   150b4: 93 0b 01 1e  	addi	s7, sp, 480
   150b8: f9 65        	lui	a1, 30
   150ba: 93 85 05 9f  	addi	a1, a1, -1552
   150be: 5e 85        	mv	a0, s7
   150c0: ef 10 70 30  	jal	0x16bc6 <sprintf>
   150c4: 6f f0 4f ff  	j	0x148b8 <_ldtoa_r+0x55c>
   150c8: ca 85        	mv	a1, s2
   150ca: 22 85        	mv	a0, s0
   150cc: be e8        	sd	a5, 80(sp)
   150ce: b2 e0        	sd	a2, 64(sp)
   150d0: ef d0 7f e1  	jal	0x12ee6 <emovo.constprop.0>
   150d4: f9 65        	lui	a1, 30
   150d6: 93 85 05 cf  	addi	a1, a1, -784
   150da: 4a 85        	mv	a0, s2
   150dc: ef e0 4f a0  	jal	0x132e0 <ecmp>
   150e0: 06 66        	ld	a2, 64(sp)
   150e2: c6 67        	ld	a5, 80(sp)
   150e4: e3 1b 05 8c  	bnez	a0, 0x149ba <_ldtoa_r+0x65e>
   150e8: e3 48 06 90  	bltz	a2, 0x149f8 <_ldtoa_r+0x69c>
   150ec: 83 c5 e7 ff  	lbu	a1, -2(a5)
   150f0: 13 85 e7 ff  	addi	a0, a5, -2
   150f4: 13 86 25 fd  	addi	a2, a1, -46
   150f8: 13 36 16 00  	seqz	a2, a2
   150fc: 13 46 f6 ff  	not	a2, a2
   15100: 52 96        	add	a2, a2, s4
   15102: 03 46 06 00  	lbu	a2, 0(a2)
   15106: 93 f5 f5 07  	andi	a1, a1, 127
   1510a: 05 8a        	andi	a2, a2, 1
   1510c: e3 1f 06 8a  	bnez	a2, 0x149ca <_ldtoa_r+0x66e>
   15110: e5 b0        	j	0x149f8 <_ldtoa_r+0x69c>
   15112: 57 70 8d 04  	<unknown>
   15116: 07 5c 0b 02  	<unknown>
   1511a: 27 5c 0c 02  	<unknown>
   1511e: 57 70 85 c4  	<unknown>
   15122: 07 dc 0d 02  	<unknown>
   15126: 27 5c 09 02  	<unknown>
   1512a: 86 77        	ld	a5, 96(sp)
   1512c: 57 70 8d 04  	<unknown>
   15130: 07 dc 07 02  	<unknown>
   15134: c6 67        	ld	a5, 80(sp)
   15136: 27 dc 07 02  	<unknown>
   1513a: bd b1        	j	0x14da8 <_ldtoa_r+0xa4c>
   1513c: 57 70 8d 04  	<unknown>
   15140: 07 5c 0c 02  	<unknown>
   15144: 27 dc 0e 02  	<unknown>
   15148: e9 bc        	j	0x14c22 <_ldtoa_r+0x8c6>
   1514a: 85 27        	addiw	a5, a5, 1
   1514c: a3 0f f5 fe  	sb	a5, -1(a0)
   15150: 65 b0        	j	0x149f8 <_ldtoa_r+0x69c>
   15152: 13 0a 01 16  	addi	s4, sp, 352
   15156: d2 85        	mv	a1, s4
   15158: 22 85        	mv	a0, s0
   1515a: be f4        	sd	a5, 104(sp)
   1515c: ef e0 cf 8c  	jal	0x13228 <emovi>
   15160: 73 27 20 c2  	csrr	a4, vlenb
   15164: 93 0a 01 14  	addi	s5, sp, 320
   15168: b3 86 ea 00  	add	a3, s5, a4
   1516c: b3 0c ea 00  	add	s9, s4, a4
   15170: b1 45        	addi	a1, zero, 12
   15172: 11 67        	lui	a4, 4
   15174: d7 77 86 c4  	<unknown>
   15178: ca f8        	sd	s2, 112(sp)
   1517a: be ec        	sd	a5, 88(sp)
   1517c: 81 44        	mv	s1, zero
   1517e: 33 8b f5 40  	sub	s6, a1, a5
   15182: 93 09 21 16  	addi	s3, sp, 354
   15186: 93 0b e7 ff  	addi	s7, a4, -2
   1518a: 36 89        	mv	s2, a3
   1518c: 83 57 81 17  	lhu	a5, 376(sp)
   15190: 9d 8b        	andi	a5, a5, 7
   15192: c5 eb        	bnez	a5, 0x15242 <_ldtoa_r+0xee6>
   15194: 57 70 86 c4  	<unknown>
   15198: 07 5c 0a 02  	<unknown>
   1519c: 27 dc 0a 02  	<unknown>
   151a0: 63 08 0b 00  	beqz	s6, 0x151b0 <_ldtoa_r+0xe54>
   151a4: 57 70 8b 04  	<unknown>
   151a8: 07 dc 0c 02  	<unknown>
   151ac: 27 5c 09 02  	<unknown>
   151b0: 56 85        	mv	a0, s5
   151b2: 23 1c 01 14  	sh	zero, 344(sp)
   151b6: ef d0 df c2  	jal	0x12de2 <eshdn1>
   151ba: 56 85        	mv	a0, s5
   151bc: ef d0 7f c2  	jal	0x12de2 <eshdn1>
   151c0: 01 47        	mv	a4, zero
   151c2: b0 0a        	addi	a2, sp, 344
   151c4: b4 1a        	addi	a3, sp, 376
   151c6: 83 55 06 00  	lhu	a1, 0(a2)
   151ca: 83 d7 06 00  	lhu	a5, 0(a3)
   151ce: 79 16        	addi	a2, a2, -2
   151d0: f9 16        	addi	a3, a3, -2
   151d2: ae 97        	add	a5, a5, a1
   151d4: ba 97        	add	a5, a5, a4
   151d6: 13 d7 07 01  	srli	a4, a5, 16
   151da: 23 11 f6 00  	sh	a5, 2(a2)
   151de: 05 8b        	andi	a4, a4, 1
   151e0: e3 93 36 ff  	bne	a3, s3, 0x151c6 <_ldtoa_r+0xe6a>
   151e4: 83 57 21 14  	lhu	a5, 322(sp)
   151e8: 03 57 41 14  	lhu	a4, 324(sp)
   151ec: 8d 27        	addiw	a5, a5, 3
   151ee: 23 11 f1 14  	sh	a5, 322(sp)
   151f2: 01 cf        	beqz	a4, 0x1520a <_ldtoa_r+0xeae>
   151f4: 56 85        	mv	a0, s5
   151f6: ef d0 df be  	jal	0x12de2 <eshdn1>
   151fa: 83 57 21 14  	lhu	a5, 322(sp)
   151fe: 03 57 41 14  	lhu	a4, 324(sp)
   15202: 85 27        	addiw	a5, a5, 1
   15204: 23 11 f1 14  	sh	a5, 322(sp)
   15208: 75 f7        	bnez	a4, 0x151f4 <_ldtoa_r+0xe98>
   1520a: 83 57 81 15  	lhu	a5, 344(sp)
   1520e: 95 eb        	bnez	a5, 0x15242 <_ldtoa_r+0xee6>
   15210: 83 57 21 14  	lhu	a5, 322(sp)
   15214: 63 e7 fb 02  	bltu	s7, a5, 0x15242 <_ldtoa_r+0xee6>
   15218: 57 70 86 c4  	<unknown>
   1521c: 07 dc 0a 02  	<unknown>
   15220: 27 5c 0a 02  	<unknown>
   15224: 63 08 0b 00  	beqz	s6, 0x15234 <_ldtoa_r+0xed8>
   15228: 57 70 8b 04  	<unknown>
   1522c: 07 5c 09 02  	<unknown>
   15230: 27 dc 0c 02  	<unknown>
   15234: 23 1c 01 16  	sh	zero, 376(sp)
   15238: fd 34        	addiw	s1, s1, -1
   1523a: 93 07 50 fd  	addi	a5, zero, -43
   1523e: e3 97 f4 f4  	bne	s1, a5, 0x1518c <_ldtoa_r+0xe30>
   15242: a6 77        	ld	a5, 104(sp)
   15244: a2 85        	mv	a1, s0
   15246: 52 85        	mv	a0, s4
   15248: be f4        	sd	a5, 104(sp)
   1524a: 46 79        	ld	s2, 112(sp)
   1524c: ef d0 bf c9  	jal	0x12ee6 <emovo.constprop.0>
   15250: a6 77        	ld	a5, 104(sp)
   15252: 93 09 01 22  	addi	s3, sp, 544
   15256: 6f f0 9f 85  	j	0x14aae <_ldtoa_r+0x752>
   1525a: ee 86        	mv	a3, s11
   1525c: 4d bc        	j	0x14d0e <_ldtoa_r+0x9b2>
   1525e: ee 87        	mv	a5, s11
   15260: 13 07 21 17  	addi	a4, sp, 370
   15264: 83 d6 07 00  	lhu	a3, 0(a5)
   15268: 89 07        	addi	a5, a5, 2
   1526a: e3 9f 06 aa  	bnez	a3, 0x14d28 <_ldtoa_r+0x9cc>
   1526e: e3 9b e7 fe  	bne	a5, a4, 0x15264 <_ldtoa_r+0xf08>
   15272: ee 87        	mv	a5, s11
   15274: 83 d6 07 00  	lhu	a3, 0(a5)
   15278: 89 07        	addi	a5, a5, 2
   1527a: e3 9c 06 c8  	bnez	a3, 0x14f12 <_ldtoa_r+0xbb6>
   1527e: e3 9b e7 fe  	bne	a5, a4, 0x15274 <_ldtoa_r+0xf18>
   15282: 5d b4        	j	0x14d28 <_ldtoa_r+0x9cc>
   15284: 57 77 86 c4  	<unknown>
   15288: 13 0a 01 16  	addi	s4, sp, 352
   1528c: 93 09 01 22  	addi	s3, sp, 544
   15290: ba ec        	sd	a4, 88(sp)
   15292: 6f f0 df 81  	j	0x14aae <_ldtoa_r+0x752>
   15296: 50 13        	addi	a2, sp, 420
   15298: 5c 03        	addi	a5, sp, 388
   1529a: 93 0e a1 19  	addi	t4, sp, 410
   1529e: 03 d5 07 00  	lhu	a0, 0(a5)
   152a2: 83 55 06 00  	lhu	a1, 0(a2)
   152a6: 89 07        	addi	a5, a5, 2
   152a8: 09 06        	addi	a2, a2, 2
   152aa: 63 17 b5 0c  	bne	a0, a1, 0x15378 <_ldtoa_r+0x101c>
   152ae: e3 98 d7 ff  	bne	a5, t4, 0x1529e <_ldtoa_r+0xf42>
   152b2: 03 56 01 18  	lhu	a2, 384(sp)
   152b6: 83 57 01 1a  	lhu	a5, 416(sp)
   152ba: 63 03 f6 10  	beq	a2, a5, 0x153c0 <_ldtoa_r+0x1064>
   152be: d7 77 80 04  	<unknown>
   152c2: 57 3c 00 5e  	<unknown>
   152c6: 57 70 85 c4  	<unknown>
   152ca: 27 dc 0d 02  	<unknown>
   152ce: e3 8d 09 a4  	beqz	s3, 0x14d28 <_ldtoa_r+0x9cc>
   152d2: aa 67        	ld	a5, 136(sp)
   152d4: 57 f0 89 04  	<unknown>
   152d8: 27 dc 07 02  	<unknown>
   152dc: b1 b4        	j	0x14d28 <_ldtoa_r+0x9cc>
   152de: ea 66        	ld	a3, 152(sp)
   152e0: aa 75        	ld	a1, 168(sp)
   152e2: 57 f0 87 04  	<unknown>
   152e6: 07 dc 06 02  	<unknown>
   152ea: 27 dc 05 02  	<unknown>
   152ee: 23 1c 01 1c  	sh	zero, 472(sp)
   152f2: 08 03        	addi	a0, sp, 384
   152f4: 57 70 86 c4  	<unknown>
   152f8: 07 5c 05 02  	<unknown>
   152fc: 08 13        	addi	a0, sp, 416
   152fe: 27 5c 05 02  	<unknown>
   15302: 0a 75        	ld	a0, 160(sp)
   15304: 57 f0 87 04  	<unknown>
   15308: 07 5c 05 02  	<unknown>
   1530c: 27 dc 06 02  	<unknown>
   15310: 23 1c 01 1a  	sh	zero, 440(sp)
   15314: 94 03        	addi	a3, sp, 448
   15316: 57 70 86 c4  	<unknown>
   1531a: 07 dc 06 02  	<unknown>
   1531e: 14 03        	addi	a3, sp, 384
   15320: 27 dc 06 02  	<unknown>
   15324: 57 f0 87 04  	<unknown>
   15328: 07 dc 05 02  	<unknown>
   1532c: 27 5c 05 02  	<unknown>
   15330: 81 b9        	j	0x14f80 <_ldtoa_r+0xc24>
   15332: f9 66        	lui	a3, 30
   15334: 85 65        	lui	a1, 1
   15336: 79 65        	lui	a0, 30
   15338: 93 86 06 a2  	addi	a3, a3, -1504
   1533c: 01 46        	mv	a2, zero
   1533e: 93 85 e5 b6  	addi	a1, a1, -1170
   15342: 13 05 85 a3  	addi	a0, a0, -1480
   15346: ef 40 00 62  	jal	0x19966 <__assert_func>
   1534a: 81 47        	mv	a5, zero
   1534c: 28 1b        	addi	a0, sp, 440
   1534e: 30 0b        	addi	a2, sp, 408
   15350: 93 0e 21 18  	addi	t4, sp, 386
   15354: 03 58 05 00  	lhu	a6, 0(a0)
   15358: 03 57 06 00  	lhu	a4, 0(a2)
   1535c: 79 15        	addi	a0, a0, -2
   1535e: 79 16        	addi	a2, a2, -2
   15360: 42 97        	add	a4, a4, a6
   15362: ba 97        	add	a5, a5, a4
   15364: 13 d7 07 01  	srli	a4, a5, 16
   15368: 23 11 f5 00  	sh	a5, 2(a0)
   1536c: 93 77 17 00  	andi	a5, a4, 1
   15370: e3 12 d6 ff  	bne	a2, t4, 0x15354 <_ldtoa_r+0xff8>
   15374: 01 46        	mv	a2, zero
   15376: 85 b9        	j	0x14fe6 <_ldtoa_r+0xc8a>
   15378: 63 f2 a5 04  	bgeu	a1, a0, 0x153bc <_ldtoa_r+0x1060>
   1537c: 1c 13        	addi	a5, sp, 416
   1537e: 57 70 86 c4  	<unknown>
   15382: 07 dc 07 02  	<unknown>
   15386: 9c 03        	addi	a5, sp, 448
   15388: 27 dc 07 02  	<unknown>
   1538c: 66 66        	ld	a2, 88(sp)
   1538e: 31 47        	addi	a4, zero, 12
   15390: b3 07 c7 40  	sub	a5, a4, a2
   15394: 63 18 c7 04  	bne	a4, a2, 0x153e4 <_ldtoa_r+0x1088>
   15398: 23 1c 01 1c  	sh	zero, 472(sp)
   1539c: 1c 03        	addi	a5, sp, 384
   1539e: 07 dc 07 02  	<unknown>
   153a2: 1c 13        	addi	a5, sp, 416
   153a4: 27 dc 07 02  	<unknown>
   153a8: 23 1c 01 1a  	sh	zero, 440(sp)
   153ac: 9c 03        	addi	a5, sp, 448
   153ae: 07 dc 07 02  	<unknown>
   153b2: 1c 03        	addi	a5, sp, 384
   153b4: 27 dc 07 02  	<unknown>
   153b8: 23 1c 01 18  	sh	zero, 408(sp)
   153bc: 81 45        	mv	a1, zero
   153be: ed b6        	j	0x14fa8 <_ldtoa_r+0xc4c>
   153c0: b6 87        	mv	a5, a3
   153c2: 89 e6        	bnez	a3, 0x153cc <_ldtoa_r+0x1070>
   153c4: 03 16 61 1a  	lh	a2, 422(sp)
   153c8: 63 5a 06 06  	bgez	a2, 0x1543c <_ldtoa_r+0x10e0>
   153cc: 93 07 21 1a  	addi	a5, sp, 418
   153d0: 30 1b        	addi	a2, sp, 440
   153d2: 83 d5 07 00  	lhu	a1, 0(a5)
   153d6: c5 e1        	bnez	a1, 0x15476 <_ldtoa_r+0x111a>
   153d8: 89 07        	addi	a5, a5, 2
   153da: e3 1c f6 fe  	bne	a2, a5, 0x153d2 <_ldtoa_r+0x1076>
   153de: 23 11 e1 1a  	sh	a4, 418(sp)
   153e2: 01 b9        	j	0x14ff2 <_ldtoa_r+0xc96>
   153e4: 6a 67        	ld	a4, 152(sp)
   153e6: 2a 76        	ld	a2, 168(sp)
   153e8: 57 f0 87 04  	<unknown>
   153ec: 07 5c 07 02  	<unknown>
   153f0: 27 5c 06 02  	<unknown>
   153f4: 23 1c 01 1c  	sh	zero, 472(sp)
   153f8: 0c 03        	addi	a1, sp, 384
   153fa: 57 70 86 c4  	<unknown>
   153fe: 07 dc 05 02  	<unknown>
   15402: 0c 13        	addi	a1, sp, 416
   15404: 27 dc 05 02  	<unknown>
   15408: 8a 75        	ld	a1, 160(sp)
   1540a: 57 f0 87 04  	<unknown>
   1540e: 07 dc 05 02  	<unknown>
   15412: 27 5c 07 02  	<unknown>
   15416: 23 1c 01 1a  	sh	zero, 440(sp)
   1541a: 98 03        	addi	a4, sp, 448
   1541c: 57 70 86 c4  	<unknown>
   15420: 07 5c 07 02  	<unknown>
   15424: 18 03        	addi	a4, sp, 384
   15426: 27 5c 07 02  	<unknown>
   1542a: 57 f0 87 04  	<unknown>
   1542e: 07 5c 06 02  	<unknown>
   15432: 27 dc 05 02  	<unknown>
   15436: 23 1c 01 18  	sh	zero, 408(sp)
   1543a: 49 b7        	j	0x153bc <_ldtoa_r+0x1060>
   1543c: 34 1b        	addi	a3, sp, 440
   1543e: 13 06 21 1a  	addi	a2, sp, 418
   15442: 03 d7 06 00  	lhu	a4, 0(a3)
   15446: 93 55 f7 00  	srli	a1, a4, 15
   1544a: 99 c1        	beqz	a1, 0x15450 <_ldtoa_r+0x10f4>
   1544c: 93 e7 17 00  	ori	a5, a5, 1
   15450: 1b 17 17 00  	slliw	a4, a4, 1
   15454: 42 17        	slli	a4, a4, 48
   15456: 93 f5 27 00  	andi	a1, a5, 2
   1545a: 41 93        	srli	a4, a4, 48
   1545c: 99 c1        	beqz	a1, 0x15462 <_ldtoa_r+0x1106>
   1545e: 13 67 17 00  	ori	a4, a4, 1
   15462: 9b 97 17 00  	slliw	a5, a5, 1
   15466: 23 90 e6 00  	sh	a4, 0(a3)
   1546a: c2 17        	slli	a5, a5, 48
   1546c: f9 16        	addi	a3, a3, -2
   1546e: c1 93        	srli	a5, a5, 48
   15470: e3 99 c6 fc  	bne	a3, a2, 0x15442 <_ldtoa_r+0x10e6>
   15474: bd be        	j	0x14ff2 <_ldtoa_r+0xc96>
   15476: 13 87 16 00  	addi	a4, a3, 1
   1547a: 23 11 e1 1a  	sh	a4, 418(sp)
   1547e: 95 be        	j	0x14ff2 <_ldtoa_r+0xc96>

0000000000015480 <_ldcheck>:
   15480: 18 61        	ld	a4, 0(a0)
   15482: 1c 65        	ld	a5, 8(a0)
   15484: 39 71        	addi	sp, sp, -64
   15486: 0a 85        	mv	a0, sp
   15488: 0c 08        	addi	a1, sp, 16
   1548a: 3a e0        	sd	a4, 0(sp)
   1548c: 3e e4        	sd	a5, 8(sp)
   1548e: 06 fc        	sd	ra, 56(sp)
   15490: ef e0 0f c6  	jal	0x138f0 <e113toe.isra.0>
   15494: 83 57 21 02  	lhu	a5, 34(sp)
   15498: 01 45        	mv	a0, zero
   1549a: 93 c7 f7 ff  	not	a5, a5
   1549e: 13 97 17 03  	slli	a4, a5, 49
   154a2: 19 eb        	bnez	a4, 0x154b8 <_ldcheck+0x38>
   154a4: 1c 08        	addi	a5, sp, 16
   154a6: 93 06 21 02  	addi	a3, sp, 34
   154aa: 03 d7 07 00  	lhu	a4, 0(a5)
   154ae: 89 07        	addi	a5, a5, 2
   154b0: 19 e7        	bnez	a4, 0x154be <_ldcheck+0x3e>
   154b2: e3 9c d7 fe  	bne	a5, a3, 0x154aa <_ldcheck+0x2a>
   154b6: 09 45        	addi	a0, zero, 2
   154b8: e2 70        	ld	ra, 56(sp)
   154ba: 21 61        	addi	sp, sp, 64
   154bc: 82 80        	ret
   154be: 05 45        	addi	a0, zero, 1
   154c0: e5 bf        	j	0x154b8 <_ldcheck+0x38>

00000000000154c2 <__localeconv_l>:
   154c2: 13 05 05 10  	addi	a0, a0, 256
   154c6: 82 80        	ret

00000000000154c8 <_localeconv_r>:
   154c8: 37 05 02 00  	lui	a0, 32
   154cc: 13 05 85 e9  	addi	a0, a0, -360
   154d0: 82 80        	ret

00000000000154d2 <localeconv>:
   154d2: 37 05 02 00  	lui	a0, 32
   154d6: 13 05 85 e9  	addi	a0, a0, -360
   154da: 82 80        	ret

00000000000154dc <_setlocale_r>:
   154dc: 01 11        	addi	sp, sp, -32
   154de: 06 ec        	sd	ra, 24(sp)
   154e0: 22 e8        	sd	s0, 16(sp)
   154e2: 26 e4        	sd	s1, 8(sp)
   154e4: 0d c2        	beqz	a2, 0x15506 <_setlocale_r+0x2a>
   154e6: f9 65        	lui	a1, 30
   154e8: 93 85 05 d1  	addi	a1, a1, -752
   154ec: 32 85        	mv	a0, a2
   154ee: 32 84        	mv	s0, a2
   154f0: ef 10 b0 00  	jal	0x16cfa <strcmp>
   154f4: f9 64        	lui	s1, 30
   154f6: 11 e9        	bnez	a0, 0x1550a <_setlocale_r+0x2e>
   154f8: 13 85 84 d0  	addi	a0, s1, -760
   154fc: e2 60        	ld	ra, 24(sp)
   154fe: 42 64        	ld	s0, 16(sp)
   15500: a2 64        	ld	s1, 8(sp)
   15502: 05 61        	addi	sp, sp, 32
   15504: 82 80        	ret
   15506: f9 64        	lui	s1, 30
   15508: c5 bf        	j	0x154f8 <_setlocale_r+0x1c>
   1550a: 93 85 84 d0  	addi	a1, s1, -760
   1550e: 22 85        	mv	a0, s0
   15510: ef 10 a0 7e  	jal	0x16cfa <strcmp>
   15514: 75 d1        	beqz	a0, 0x154f8 <_setlocale_r+0x1c>
   15516: f5 65        	lui	a1, 29
   15518: 93 85 05 77  	addi	a1, a1, 1904
   1551c: 22 85        	mv	a0, s0
   1551e: ef 10 c0 7d  	jal	0x16cfa <strcmp>
   15522: 79 d9        	beqz	a0, 0x154f8 <_setlocale_r+0x1c>
   15524: 01 45        	mv	a0, zero
   15526: d9 bf        	j	0x154fc <_setlocale_r+0x20>

0000000000015528 <__locale_mb_cur_max>:
   15528: b7 07 02 00  	lui	a5, 32
   1552c: 03 c5 87 ef  	lbu	a0, -264(a5)
   15530: 82 80        	ret

0000000000015532 <setlocale>:
   15532: 01 11        	addi	sp, sp, -32
   15534: 06 ec        	sd	ra, 24(sp)
   15536: 22 e8        	sd	s0, 16(sp)
   15538: 26 e4        	sd	s1, 8(sp)
   1553a: 8d c1        	beqz	a1, 0x1555c <setlocale+0x2a>
   1553c: 2e 84        	mv	s0, a1
   1553e: f9 65        	lui	a1, 30
   15540: 93 85 05 d1  	addi	a1, a1, -752
   15544: 22 85        	mv	a0, s0
   15546: ef 10 40 7b  	jal	0x16cfa <strcmp>
   1554a: f9 64        	lui	s1, 30
   1554c: 11 e9        	bnez	a0, 0x15560 <setlocale+0x2e>
   1554e: 13 85 84 d0  	addi	a0, s1, -760
   15552: e2 60        	ld	ra, 24(sp)
   15554: 42 64        	ld	s0, 16(sp)
   15556: a2 64        	ld	s1, 8(sp)
   15558: 05 61        	addi	sp, sp, 32
   1555a: 82 80        	ret
   1555c: f9 64        	lui	s1, 30
   1555e: c5 bf        	j	0x1554e <setlocale+0x1c>
   15560: 93 85 84 d0  	addi	a1, s1, -760
   15564: 22 85        	mv	a0, s0
   15566: ef 10 40 79  	jal	0x16cfa <strcmp>
   1556a: 75 d1        	beqz	a0, 0x1554e <setlocale+0x1c>
   1556c: f5 65        	lui	a1, 29
   1556e: 93 85 05 77  	addi	a1, a1, 1904
   15572: 22 85        	mv	a0, s0
   15574: ef 10 60 78  	jal	0x16cfa <strcmp>
   15578: 79 d9        	beqz	a0, 0x1554e <setlocale+0x1c>
   1557a: 01 45        	mv	a0, zero
   1557c: d9 bf        	j	0x15552 <setlocale+0x20>

000000000001557e <__smakebuf_r>:
   1557e: 83 97 05 01  	lh	a5, 16(a1)
   15582: 35 71        	addi	sp, sp, -160
   15584: 22 e9        	sd	s0, 144(sp)
   15586: 06 ed        	sd	ra, 152(sp)
   15588: 26 e5        	sd	s1, 136(sp)
   1558a: 4a e1        	sd	s2, 128(sp)
   1558c: ce fc        	sd	s3, 120(sp)
   1558e: d2 f8        	sd	s4, 112(sp)
   15590: 13 f7 27 00  	andi	a4, a5, 2
   15594: 2e 84        	mv	s0, a1
   15596: 19 cf        	beqz	a4, 0x155b4 <__smakebuf_r+0x36>
   15598: 93 87 75 07  	addi	a5, a1, 119
   1559c: 9c e1        	sd	a5, 0(a1)
   1559e: 9c ed        	sd	a5, 24(a1)
   155a0: 85 47        	addi	a5, zero, 1
   155a2: 9c d1        	sw	a5, 32(a1)
   155a4: ea 60        	ld	ra, 152(sp)
   155a6: 4a 64        	ld	s0, 144(sp)
   155a8: aa 64        	ld	s1, 136(sp)
   155aa: 0a 69        	ld	s2, 128(sp)
   155ac: e6 79        	ld	s3, 120(sp)
   155ae: 46 7a        	ld	s4, 112(sp)
   155b0: 0d 61        	addi	sp, sp, 160
   155b2: 82 80        	ret
   155b4: 83 95 25 01  	lh	a1, 18(a1)
   155b8: aa 84        	mv	s1, a0
   155ba: 63 c8 05 06  	bltz	a1, 0x1562a <__smakebuf_r+0xac>
   155be: 30 00        	addi	a2, sp, 8
   155c0: ef 40 00 70  	jal	0x19cc0 <_fstat_r>
   155c4: 63 41 05 06  	bltz	a0, 0x15626 <__smakebuf_r+0xa8>
   155c8: 32 49        	lw	s2, 12(sp)
   155ca: bd 67        	lui	a5, 15
   155cc: 85 69        	lui	s3, 1
   155ce: 33 79 f9 00  	and	s2, s2, a5
   155d2: 89 67        	lui	a5, 2
   155d4: 33 09 f9 40  	sub	s2, s2, a5
   155d8: 13 39 19 00  	seqz	s2, s2
   155dc: 13 0a 00 40  	addi	s4, zero, 1024
   155e0: 93 89 09 80  	addi	s3, s3, -2048
   155e4: d2 85        	mv	a1, s4
   155e6: 26 85        	mv	a0, s1
   155e8: ef 00 20 13  	jal	0x1571a <_malloc_r>
   155ec: 83 17 04 01  	lh	a5, 16(s0)
   155f0: 39 c9        	beqz	a0, 0x15646 <__smakebuf_r+0xc8>
   155f2: 49 67        	lui	a4, 18
   155f4: 13 07 27 71  	addi	a4, a4, 1810
   155f8: b8 ec        	sd	a4, 88(s1)
   155fa: 93 e7 07 08  	ori	a5, a5, 128
   155fe: 23 18 f4 00  	sh	a5, 16(s0)
   15602: 08 e0        	sd	a0, 0(s0)
   15604: 08 ec        	sd	a0, 24(s0)
   15606: 23 20 44 03  	sw	s4, 32(s0)
   1560a: 63 11 09 06  	bnez	s2, 0x1566c <__smakebuf_r+0xee>
   1560e: b3 e7 37 01  	or	a5, a5, s3
   15612: ea 60        	ld	ra, 152(sp)
   15614: 23 18 f4 00  	sh	a5, 16(s0)
   15618: 4a 64        	ld	s0, 144(sp)
   1561a: aa 64        	ld	s1, 136(sp)
   1561c: 0a 69        	ld	s2, 128(sp)
   1561e: e6 79        	ld	s3, 120(sp)
   15620: 46 7a        	ld	s4, 112(sp)
   15622: 0d 61        	addi	sp, sp, 160
   15624: 82 80        	ret
   15626: 83 17 04 01  	lh	a5, 16(s0)
   1562a: 93 f7 07 08  	andi	a5, a5, 128
   1562e: 01 49        	mv	s2, zero
   15630: 95 cb        	beqz	a5, 0x15664 <__smakebuf_r+0xe6>
   15632: 13 0a 00 04  	addi	s4, zero, 64
   15636: d2 85        	mv	a1, s4
   15638: 26 85        	mv	a0, s1
   1563a: ef 00 00 0e  	jal	0x1571a <_malloc_r>
   1563e: 83 17 04 01  	lh	a5, 16(s0)
   15642: 81 49        	mv	s3, zero
   15644: 5d f5        	bnez	a0, 0x155f2 <__smakebuf_r+0x74>
   15646: 13 f7 07 20  	andi	a4, a5, 512
   1564a: 29 ff        	bnez	a4, 0x155a4 <__smakebuf_r+0x26>
   1564c: f1 9b        	andi	a5, a5, -4
   1564e: 93 e7 27 00  	ori	a5, a5, 2
   15652: 13 07 74 07  	addi	a4, s0, 119
   15656: 23 18 f4 00  	sh	a5, 16(s0)
   1565a: 85 47        	addi	a5, zero, 1
   1565c: 18 e0        	sd	a4, 0(s0)
   1565e: 18 ec        	sd	a4, 24(s0)
   15660: 1c d0        	sw	a5, 32(s0)
   15662: 89 b7        	j	0x155a4 <__smakebuf_r+0x26>
   15664: 13 0a 00 40  	addi	s4, zero, 1024
   15668: 81 49        	mv	s3, zero
   1566a: ad bf        	j	0x155e4 <__smakebuf_r+0x66>
   1566c: 83 15 24 01  	lh	a1, 18(s0)
   15670: 26 85        	mv	a0, s1
   15672: ef 40 70 1b  	jal	0x1a028 <_isatty_r>
   15676: 01 e5        	bnez	a0, 0x1567e <__smakebuf_r+0x100>
   15678: 83 17 04 01  	lh	a5, 16(s0)
   1567c: 49 bf        	j	0x1560e <__smakebuf_r+0x90>
   1567e: 83 57 04 01  	lhu	a5, 16(s0)
   15682: f1 9b        	andi	a5, a5, -4
   15684: 93 e7 17 00  	ori	a5, a5, 1
   15688: 9b 97 07 01  	slliw	a5, a5, 16
   1568c: 9b d7 07 41  	sraiw	a5, a5, 16
   15690: bd bf        	j	0x1560e <__smakebuf_r+0x90>

0000000000015692 <__swhatbuf_r>:
   15692: 75 71        	addi	sp, sp, -144
   15694: 22 e1        	sd	s0, 128(sp)
   15696: 2e 84        	mv	s0, a1
   15698: 83 95 25 01  	lh	a1, 18(a1)
   1569c: a6 fc        	sd	s1, 120(sp)
   1569e: ca f8        	sd	s2, 112(sp)
   156a0: 06 e5        	sd	ra, 136(sp)
   156a2: b2 84        	mv	s1, a2
   156a4: 36 89        	mv	s2, a3
   156a6: 63 cc 05 02  	bltz	a1, 0x156de <__swhatbuf_r+0x4c>
   156aa: 30 00        	addi	a2, sp, 8
   156ac: ef 40 40 61  	jal	0x19cc0 <_fstat_r>
   156b0: 63 47 05 02  	bltz	a0, 0x156de <__swhatbuf_r+0x4c>
   156b4: b2 47        	lw	a5, 12(sp)
   156b6: 3d 67        	lui	a4, 15
   156b8: aa 60        	ld	ra, 136(sp)
   156ba: f9 8f        	and	a5, a5, a4
   156bc: 09 67        	lui	a4, 2
   156be: 99 8f        	sub	a5, a5, a4
   156c0: 0a 64        	ld	s0, 128(sp)
   156c2: 93 b7 17 00  	seqz	a5, a5
   156c6: 23 20 f9 00  	sw	a5, 0(s2)
   156ca: 13 07 00 40  	addi	a4, zero, 1024
   156ce: 98 e0        	sd	a4, 0(s1)
   156d0: 05 65        	lui	a0, 1
   156d2: e6 74        	ld	s1, 120(sp)
   156d4: 46 79        	ld	s2, 112(sp)
   156d6: 13 05 05 80  	addi	a0, a0, -2048
   156da: 49 61        	addi	sp, sp, 144
   156dc: 82 80        	ret
   156de: 83 57 04 01  	lhu	a5, 16(s0)
   156e2: 93 f7 07 08  	andi	a5, a5, 128
   156e6: 91 cf        	beqz	a5, 0x15702 <__swhatbuf_r+0x70>
   156e8: aa 60        	ld	ra, 136(sp)
   156ea: 0a 64        	ld	s0, 128(sp)
   156ec: 81 47        	mv	a5, zero
   156ee: 23 20 f9 00  	sw	a5, 0(s2)
   156f2: 13 07 00 04  	addi	a4, zero, 64
   156f6: 98 e0        	sd	a4, 0(s1)
   156f8: 46 79        	ld	s2, 112(sp)
   156fa: e6 74        	ld	s1, 120(sp)
   156fc: 01 45        	mv	a0, zero
   156fe: 49 61        	addi	sp, sp, 144
   15700: 82 80        	ret
   15702: aa 60        	ld	ra, 136(sp)
   15704: 0a 64        	ld	s0, 128(sp)
   15706: 23 20 f9 00  	sw	a5, 0(s2)
   1570a: 13 07 00 40  	addi	a4, zero, 1024
   1570e: 98 e0        	sd	a4, 0(s1)
   15710: 46 79        	ld	s2, 112(sp)
   15712: e6 74        	ld	s1, 120(sp)
   15714: 01 45        	mv	a0, zero
   15716: 49 61        	addi	sp, sp, 144
   15718: 82 80        	ret

000000000001571a <_malloc_r>:
   1571a: 1d 71        	addi	sp, sp, -96
   1571c: a6 e4        	sd	s1, 72(sp)
   1571e: ca e0        	sd	s2, 64(sp)
   15720: 86 ec        	sd	ra, 88(sp)
   15722: a2 e8        	sd	s0, 80(sp)
   15724: 4e fc        	sd	s3, 56(sp)
   15726: 52 f8        	sd	s4, 48(sp)
   15728: 56 f4        	sd	s5, 40(sp)
   1572a: 5a f0        	sd	s6, 32(sp)
   1572c: 5e ec        	sd	s7, 24(sp)
   1572e: 62 e8        	sd	s8, 16(sp)
   15730: 66 e4        	sd	s9, 8(sp)
   15732: 93 84 75 01  	addi	s1, a1, 23
   15736: 93 07 e0 02  	addi	a5, zero, 46
   1573a: 2a 89        	mv	s2, a0
   1573c: 63 e4 97 06  	bltu	a5, s1, 0x157a4 <_malloc_r+0x8a>
   15740: 93 07 00 02  	addi	a5, zero, 32
   15744: 63 e1 b7 1a  	bltu	a5, a1, 0x158e6 <_malloc_r+0x1cc>
   15748: ef 00 40 6a  	jal	0x15dec <__malloc_lock>
   1574c: 93 04 00 02  	addi	s1, zero, 32
   15750: 93 07 00 05  	addi	a5, zero, 80
   15754: 11 46        	addi	a2, zero, 4
   15756: b7 09 02 00  	lui	s3, 32
   1575a: 93 89 09 f4  	addi	s3, s3, -192
   1575e: ce 97        	add	a5, a5, s3
   15760: 80 67        	ld	s0, 8(a5)
   15762: 13 87 07 ff  	addi	a4, a5, -16
   15766: 63 0e e4 24  	beq	s0, a4, 0x159c2 <_malloc_r+0x2a8>
   1576a: 1c 64        	ld	a5, 8(s0)
   1576c: 14 6c        	ld	a3, 24(s0)
   1576e: 10 68        	ld	a2, 16(s0)
   15770: f1 9b        	andi	a5, a5, -4
   15772: a2 97        	add	a5, a5, s0
   15774: 98 67        	ld	a4, 8(a5)
   15776: 14 ee        	sd	a3, 24(a2)
   15778: 90 ea        	sd	a2, 16(a3)
   1577a: 13 67 17 00  	ori	a4, a4, 1
   1577e: 4a 85        	mv	a0, s2
   15780: 98 e7        	sd	a4, 8(a5)
   15782: ef 00 c0 66  	jal	0x15dee <__malloc_unlock>
   15786: 13 05 04 01  	addi	a0, s0, 16
   1578a: e6 60        	ld	ra, 88(sp)
   1578c: 46 64        	ld	s0, 80(sp)
   1578e: a6 64        	ld	s1, 72(sp)
   15790: 06 69        	ld	s2, 64(sp)
   15792: e2 79        	ld	s3, 56(sp)
   15794: 42 7a        	ld	s4, 48(sp)
   15796: a2 7a        	ld	s5, 40(sp)
   15798: 02 7b        	ld	s6, 32(sp)
   1579a: e2 6b        	ld	s7, 24(sp)
   1579c: 42 6c        	ld	s8, 16(sp)
   1579e: a2 6c        	ld	s9, 8(sp)
   157a0: 25 61        	addi	sp, sp, 96
   157a2: 82 80        	ret
   157a4: b7 07 00 80  	lui	a5, 524288
   157a8: c1 98        	andi	s1, s1, -16
   157aa: 93 c7 f7 ff  	not	a5, a5
   157ae: 63 ec 97 12  	bltu	a5, s1, 0x158e6 <_malloc_r+0x1cc>
   157b2: 63 ea b4 12  	bltu	s1, a1, 0x158e6 <_malloc_r+0x1cc>
   157b6: ef 00 60 63  	jal	0x15dec <__malloc_lock>
   157ba: 93 07 70 1f  	addi	a5, zero, 503
   157be: 63 fd 97 2e  	bgeu	a5, s1, 0x15ab8 <_malloc_r+0x39e>
   157c2: 93 d7 94 00  	srli	a5, s1, 9
   157c6: 63 85 07 12  	beqz	a5, 0x158f0 <_malloc_r+0x1d6>
   157ca: 11 47        	addi	a4, zero, 4
   157cc: 63 62 f7 24  	bltu	a4, a5, 0x15a10 <_malloc_r+0x2f6>
   157d0: 93 d7 64 00  	srli	a5, s1, 6
   157d4: 1b 86 97 03  	addiw	a2, a5, 57
   157d8: 9b 16 16 00  	slliw	a3, a2, 1
   157dc: 1b 85 87 03  	addiw	a0, a5, 56
   157e0: 8e 06        	slli	a3, a3, 3
   157e2: b7 09 02 00  	lui	s3, 32
   157e6: 93 89 09 f4  	addi	s3, s3, -192
   157ea: ce 96        	add	a3, a3, s3
   157ec: 80 66        	ld	s0, 8(a3)
   157ee: c1 16        	addi	a3, a3, -16
   157f0: 63 80 86 02  	beq	a3, s0, 0x15810 <_malloc_r+0xf6>
   157f4: fd 45        	addi	a1, zero, 31
   157f6: 31 a0        	j	0x15802 <_malloc_r+0xe8>
   157f8: 63 52 07 1c  	bgez	a4, 0x159bc <_malloc_r+0x2a2>
   157fc: 00 6c        	ld	s0, 24(s0)
   157fe: 63 89 86 00  	beq	a3, s0, 0x15810 <_malloc_r+0xf6>
   15802: 1c 64        	ld	a5, 8(s0)
   15804: f1 9b        	andi	a5, a5, -4
   15806: 33 87 97 40  	sub	a4, a5, s1
   1580a: e3 d7 e5 fe  	bge	a1, a4, 0x157f8 <_malloc_r+0xde>
   1580e: 2a 86        	mv	a2, a0
   15810: 03 b4 09 02  	ld	s0, 32(s3)
   15814: 37 08 02 00  	lui	a6, 32
   15818: 13 08 08 f5  	addi	a6, a6, -176
   1581c: 63 0d 04 19  	beq	s0, a6, 0x159b6 <_malloc_r+0x29c>
   15820: 1c 64        	ld	a5, 8(s0)
   15822: fd 46        	addi	a3, zero, 31
   15824: f1 9b        	andi	a5, a5, -4
   15826: 33 87 97 40  	sub	a4, a5, s1
   1582a: 63 c0 e6 26  	blt	a3, a4, 0x15a8a <_malloc_r+0x370>
   1582e: 23 b4 09 03  	sd	a6, 40(s3)
   15832: 23 b0 09 03  	sd	a6, 32(s3)
   15836: 63 5f 07 22  	bgez	a4, 0x15a74 <_malloc_r+0x35a>
   1583a: 13 07 f0 1f  	addi	a4, zero, 511
   1583e: 03 b5 89 00  	ld	a0, 8(s3)
   15842: 63 65 f7 18  	bltu	a4, a5, 0x159cc <_malloc_r+0x2b2>
   15846: 8d 83        	srli	a5, a5, 3
   15848: 1b 87 17 00  	addiw	a4, a5, 1
   1584c: 1b 17 17 00  	slliw	a4, a4, 1
   15850: 0e 07        	slli	a4, a4, 3
   15852: 4e 97        	add	a4, a4, s3
   15854: 0c 63        	ld	a1, 0(a4)
   15856: 9b d7 27 40  	sraiw	a5, a5, 2
   1585a: 85 46        	addi	a3, zero, 1
   1585c: b3 97 f6 00  	sll	a5, a3, a5
   15860: 5d 8d        	or	a0, a0, a5
   15862: 93 07 07 ff  	addi	a5, a4, -16
   15866: 1c ec        	sd	a5, 24(s0)
   15868: 0c e8        	sd	a1, 16(s0)
   1586a: 23 b4 a9 00  	sd	a0, 8(s3)
   1586e: 00 e3        	sd	s0, 0(a4)
   15870: 80 ed        	sd	s0, 24(a1)
   15872: 9b 57 26 40  	sraiw	a5, a2, 2
   15876: 85 45        	addi	a1, zero, 1
   15878: b3 95 f5 00  	sll	a1, a1, a5
   1587c: 63 61 b5 08  	bltu	a0, a1, 0x158fe <_malloc_r+0x1e4>
   15880: b3 f7 a5 00  	and	a5, a1, a0
   15884: 81 ef        	bnez	a5, 0x1589c <_malloc_r+0x182>
   15886: 86 05        	slli	a1, a1, 1
   15888: 71 9a        	andi	a2, a2, -4
   1588a: b3 f7 a5 00  	and	a5, a1, a0
   1588e: 11 26        	addiw	a2, a2, 4
   15890: 91 e7        	bnez	a5, 0x1589c <_malloc_r+0x182>
   15892: 86 05        	slli	a1, a1, 1
   15894: b3 f7 a5 00  	and	a5, a1, a0
   15898: 11 26        	addiw	a2, a2, 4
   1589a: e5 df        	beqz	a5, 0x15892 <_malloc_r+0x178>
   1589c: fd 48        	addi	a7, zero, 31
   1589e: 1b 03 16 00  	addiw	t1, a2, 1
   158a2: 1b 13 13 00  	slliw	t1, t1, 1
   158a6: 0e 03        	slli	t1, t1, 3
   158a8: 41 13        	addi	t1, t1, -16
   158aa: 4e 93        	add	t1, t1, s3
   158ac: 1a 85        	mv	a0, t1
   158ae: 1c 6d        	ld	a5, 24(a0)
   158b0: 32 8e        	mv	t3, a2
   158b2: 63 00 f5 18  	beq	a0, a5, 0x15a32 <_malloc_r+0x318>
   158b6: 98 67        	ld	a4, 8(a5)
   158b8: 3e 84        	mv	s0, a5
   158ba: 9c 6f        	ld	a5, 24(a5)
   158bc: 71 9b        	andi	a4, a4, -4
   158be: b3 06 97 40  	sub	a3, a4, s1
   158c2: 63 cf d8 16  	blt	a7, a3, 0x15a40 <_malloc_r+0x326>
   158c6: e3 c6 06 fe  	bltz	a3, 0x158b2 <_malloc_r+0x198>
   158ca: 22 97        	add	a4, a4, s0
   158cc: 14 67        	ld	a3, 8(a4)
   158ce: 10 68        	ld	a2, 16(s0)
   158d0: 4a 85        	mv	a0, s2
   158d2: 93 e6 16 00  	ori	a3, a3, 1
   158d6: 14 e7        	sd	a3, 8(a4)
   158d8: 1c ee        	sd	a5, 24(a2)
   158da: 90 eb        	sd	a2, 16(a5)
   158dc: ef 00 20 51  	jal	0x15dee <__malloc_unlock>
   158e0: 13 05 04 01  	addi	a0, s0, 16
   158e4: 5d b5        	j	0x1578a <_malloc_r+0x70>
   158e6: b1 47        	addi	a5, zero, 12
   158e8: 23 20 f9 00  	sw	a5, 0(s2)
   158ec: 01 45        	mv	a0, zero
   158ee: 71 bd        	j	0x1578a <_malloc_r+0x70>
   158f0: 93 06 00 40  	addi	a3, zero, 1024
   158f4: 13 06 00 04  	addi	a2, zero, 64
   158f8: 13 05 f0 03  	addi	a0, zero, 63
   158fc: dd b5        	j	0x157e2 <_malloc_r+0xc8>
   158fe: 03 b4 09 01  	ld	s0, 16(s3)
   15902: 1c 64        	ld	a5, 8(s0)
   15904: 93 fa c7 ff  	andi	s5, a5, -4
   15908: 63 e7 9a 00  	bltu	s5, s1, 0x15916 <_malloc_r+0x1fc>
   1590c: 33 87 9a 40  	sub	a4, s5, s1
   15910: fd 47        	addi	a5, zero, 31
   15912: 63 c3 e7 08  	blt	a5, a4, 0x15998 <_malloc_r+0x27e>
   15916: 03 ba 01 89  	ld	s4, -1904(gp)
   1591a: 03 b7 01 86  	ld	a4, -1952(gp)
   1591e: fd 57        	addi	a5, zero, -1
   15920: 33 0c 54 01  	add	s8, s0, s5
   15924: 26 9a        	add	s4, s4, s1
   15926: 63 05 f7 2e  	beq	a4, a5, 0x15c10 <_malloc_r+0x4f6>
   1592a: 85 67        	lui	a5, 1
   1592c: fd 07        	addi	a5, a5, 31
   1592e: 3e 9a        	add	s4, s4, a5
   15930: fd 77        	lui	a5, 1048575
   15932: 33 7a fa 00  	and	s4, s4, a5
   15936: d2 85        	mv	a1, s4
   15938: 4a 85        	mv	a0, s2
   1593a: ef 10 a0 19  	jal	0x16ad4 <_sbrk_r>
   1593e: fd 57        	addi	a5, zero, -1
   15940: 2a 8b        	mv	s6, a0
   15942: 63 00 f5 1a  	beq	a0, a5, 0x15ae2 <_malloc_r+0x3c8>
   15946: 63 6c 85 19  	bltu	a0, s8, 0x15ade <_malloc_r+0x3c4>
   1594a: b7 3b 02 00  	lui	s7, 35
   1594e: 93 8b 0b a9  	addi	s7, s7, -1392
   15952: 83 a7 0b 00  	lw	a5, 0(s7)
   15956: 3b 87 47 01  	addw	a4, a5, s4
   1595a: 23 a0 eb 00  	sw	a4, 0(s7)
   1595e: ba 86        	mv	a3, a4
   15960: 63 12 ac 20  	bne	s8, a0, 0x15b64 <_malloc_r+0x44a>
   15964: 93 17 45 03  	slli	a5, a0, 52
   15968: 63 9e 07 1e  	bnez	a5, 0x15b64 <_malloc_r+0x44a>
   1596c: 03 bb 09 01  	ld	s6, 16(s3)
   15970: b3 87 4a 01  	add	a5, s5, s4
   15974: 93 e7 17 00  	ori	a5, a5, 1
   15978: 23 34 fb 00  	sd	a5, 8(s6)
   1597c: 03 b6 81 88  	ld	a2, -1912(gp)
   15980: 63 74 e6 00  	bgeu	a2, a4, 0x15988 <_malloc_r+0x26e>
   15984: 23 b4 e1 88  	sd	a4, -1912(gp)
   15988: 03 b6 01 88  	ld	a2, -1920(gp)
   1598c: 63 74 e6 00  	bgeu	a2, a4, 0x15994 <_malloc_r+0x27a>
   15990: 23 b0 e1 88  	sd	a4, -1920(gp)
   15994: 5a 84        	mv	s0, s6
   15996: 89 aa        	j	0x15ae8 <_malloc_r+0x3ce>
   15998: 93 e7 14 00  	ori	a5, s1, 1
   1599c: 1c e4        	sd	a5, 8(s0)
   1599e: a2 94        	add	s1, s1, s0
   159a0: 23 b8 99 00  	sd	s1, 16(s3)
   159a4: 13 67 17 00  	ori	a4, a4, 1
   159a8: 4a 85        	mv	a0, s2
   159aa: 98 e4        	sd	a4, 8(s1)
   159ac: ef 00 20 44  	jal	0x15dee <__malloc_unlock>
   159b0: 13 05 04 01  	addi	a0, s0, 16
   159b4: d9 bb        	j	0x1578a <_malloc_r+0x70>
   159b6: 03 b5 89 00  	ld	a0, 8(s3)
   159ba: 65 bd        	j	0x15872 <_malloc_r+0x158>
   159bc: 14 6c        	ld	a3, 24(s0)
   159be: 10 68        	ld	a2, 16(s0)
   159c0: 4d bb        	j	0x15772 <_malloc_r+0x58>
   159c2: 80 6f        	ld	s0, 24(a5)
   159c4: 09 26        	addiw	a2, a2, 2
   159c6: e3 85 87 e4  	beq	a5, s0, 0x15810 <_malloc_r+0xf6>
   159ca: 45 b3        	j	0x1576a <_malloc_r+0x50>
   159cc: 13 d7 97 00  	srli	a4, a5, 9
   159d0: 91 46        	addi	a3, zero, 4
   159d2: 63 fc e6 0e  	bgeu	a3, a4, 0x15aca <_malloc_r+0x3b0>
   159d6: d1 46        	addi	a3, zero, 20
   159d8: 63 e8 e6 26  	bltu	a3, a4, 0x15c48 <_malloc_r+0x52e>
   159dc: 9b 05 c7 05  	addiw	a1, a4, 92
   159e0: 9b 95 15 00  	slliw	a1, a1, 1
   159e4: 9b 06 b7 05  	addiw	a3, a4, 91
   159e8: 8e 05        	slli	a1, a1, 3
   159ea: ce 95        	add	a1, a1, s3
   159ec: 98 61        	ld	a4, 0(a1)
   159ee: c1 15        	addi	a1, a1, -16
   159f0: 63 96 e5 00  	bne	a1, a4, 0x159fc <_malloc_r+0x2e2>
   159f4: 0d a4        	j	0x15c16 <_malloc_r+0x4fc>
   159f6: 18 6b        	ld	a4, 16(a4)
   159f8: 63 86 e5 00  	beq	a1, a4, 0x15a04 <_malloc_r+0x2ea>
   159fc: 14 67        	ld	a3, 8(a4)
   159fe: f1 9a        	andi	a3, a3, -4
   15a00: e3 eb d7 fe  	bltu	a5, a3, 0x159f6 <_malloc_r+0x2dc>
   15a04: 0c 6f        	ld	a1, 24(a4)
   15a06: 0c ec        	sd	a1, 24(s0)
   15a08: 18 e8        	sd	a4, 16(s0)
   15a0a: 80 e9        	sd	s0, 16(a1)
   15a0c: 00 ef        	sd	s0, 24(a4)
   15a0e: 95 b5        	j	0x15872 <_malloc_r+0x158>
   15a10: 51 47        	addi	a4, zero, 20
   15a12: 63 78 f7 0e  	bgeu	a4, a5, 0x15b02 <_malloc_r+0x3e8>
   15a16: 13 07 40 05  	addi	a4, zero, 84
   15a1a: 63 65 f7 24  	bltu	a4, a5, 0x15c64 <_malloc_r+0x54a>
   15a1e: 93 d7 c4 00  	srli	a5, s1, 12
   15a22: 1b 86 f7 06  	addiw	a2, a5, 111
   15a26: 9b 16 16 00  	slliw	a3, a2, 1
   15a2a: 1b 85 e7 06  	addiw	a0, a5, 110
   15a2e: 8e 06        	slli	a3, a3, 3
   15a30: 4d bb        	j	0x157e2 <_malloc_r+0xc8>
   15a32: 05 2e        	addiw	t3, t3, 1
   15a34: 93 77 3e 00  	andi	a5, t3, 3
   15a38: 41 05        	addi	a0, a0, 16
   15a3a: ed c3        	beqz	a5, 0x15b1c <_malloc_r+0x402>
   15a3c: 1c 6d        	ld	a5, 24(a0)
   15a3e: 95 bd        	j	0x158b2 <_malloc_r+0x198>
   15a40: 10 68        	ld	a2, 16(s0)
   15a42: 93 e5 14 00  	ori	a1, s1, 1
   15a46: 0c e4        	sd	a1, 8(s0)
   15a48: 1c ee        	sd	a5, 24(a2)
   15a4a: 90 eb        	sd	a2, 16(a5)
   15a4c: a2 94        	add	s1, s1, s0
   15a4e: 23 b4 99 02  	sd	s1, 40(s3)
   15a52: 23 b0 99 02  	sd	s1, 32(s3)
   15a56: 93 e7 16 00  	ori	a5, a3, 1
   15a5a: 23 bc 04 01  	sd	a6, 24(s1)
   15a5e: 23 b8 04 01  	sd	a6, 16(s1)
   15a62: 9c e4        	sd	a5, 8(s1)
   15a64: 22 97        	add	a4, a4, s0
   15a66: 4a 85        	mv	a0, s2
   15a68: 14 e3        	sd	a3, 0(a4)
   15a6a: ef 00 40 38  	jal	0x15dee <__malloc_unlock>
   15a6e: 13 05 04 01  	addi	a0, s0, 16
   15a72: 21 bb        	j	0x1578a <_malloc_r+0x70>
   15a74: a2 97        	add	a5, a5, s0
   15a76: 98 67        	ld	a4, 8(a5)
   15a78: 4a 85        	mv	a0, s2
   15a7a: 13 67 17 00  	ori	a4, a4, 1
   15a7e: 98 e7        	sd	a4, 8(a5)
   15a80: ef 00 e0 36  	jal	0x15dee <__malloc_unlock>
   15a84: 13 05 04 01  	addi	a0, s0, 16
   15a88: 09 b3        	j	0x1578a <_malloc_r+0x70>
   15a8a: 93 e6 14 00  	ori	a3, s1, 1
   15a8e: 14 e4        	sd	a3, 8(s0)
   15a90: a2 94        	add	s1, s1, s0
   15a92: 23 b4 99 02  	sd	s1, 40(s3)
   15a96: 23 b0 99 02  	sd	s1, 32(s3)
   15a9a: 93 66 17 00  	ori	a3, a4, 1
   15a9e: 23 bc 04 01  	sd	a6, 24(s1)
   15aa2: 23 b8 04 01  	sd	a6, 16(s1)
   15aa6: 94 e4        	sd	a3, 8(s1)
   15aa8: a2 97        	add	a5, a5, s0
   15aaa: 4a 85        	mv	a0, s2
   15aac: 98 e3        	sd	a4, 0(a5)
   15aae: ef 00 00 34  	jal	0x15dee <__malloc_unlock>
   15ab2: 13 05 04 01  	addi	a0, s0, 16
   15ab6: d1 b9        	j	0x1578a <_malloc_r+0x70>
   15ab8: 13 d6 34 00  	srli	a2, s1, 3
   15abc: 9b 07 16 00  	addiw	a5, a2, 1
   15ac0: 9b 97 17 00  	slliw	a5, a5, 1
   15ac4: 01 26        	sext.w	a2, a2
   15ac6: 8e 07        	slli	a5, a5, 3
   15ac8: 79 b1        	j	0x15756 <_malloc_r+0x3c>
   15aca: 13 d7 67 00  	srli	a4, a5, 6
   15ace: 9b 05 97 03  	addiw	a1, a4, 57
   15ad2: 9b 95 15 00  	slliw	a1, a1, 1
   15ad6: 9b 06 87 03  	addiw	a3, a4, 56
   15ada: 8e 05        	slli	a1, a1, 3
   15adc: 39 b7        	j	0x159ea <_malloc_r+0x2d0>
   15ade: 63 09 34 07  	beq	s0, s3, 0x15b50 <_malloc_r+0x436>
   15ae2: 03 b4 09 01  	ld	s0, 16(s3)
   15ae6: 1c 64        	ld	a5, 8(s0)
   15ae8: f1 9b        	andi	a5, a5, -4
   15aea: 33 87 97 40  	sub	a4, a5, s1
   15aee: 63 e5 97 00  	bltu	a5, s1, 0x15af8 <_malloc_r+0x3de>
   15af2: fd 47        	addi	a5, zero, 31
   15af4: e3 c2 e7 ea  	blt	a5, a4, 0x15998 <_malloc_r+0x27e>
   15af8: 4a 85        	mv	a0, s2
   15afa: ef 00 40 2f  	jal	0x15dee <__malloc_unlock>
   15afe: 01 45        	mv	a0, zero
   15b00: 69 b1        	j	0x1578a <_malloc_r+0x70>
   15b02: 1b 86 c7 05  	addiw	a2, a5, 92
   15b06: 9b 16 16 00  	slliw	a3, a2, 1
   15b0a: 1b 85 b7 05  	addiw	a0, a5, 91
   15b0e: 8e 06        	slli	a3, a3, 3
   15b10: c9 b9        	j	0x157e2 <_malloc_r+0xc8>
   15b12: 83 37 03 01  	ld	a5, 16(t1)
   15b16: 7d 36        	addiw	a2, a2, -1
   15b18: 63 97 67 1e  	bne	a5, t1, 0x15d06 <_malloc_r+0x5ec>
   15b1c: 93 77 36 00  	andi	a5, a2, 3
   15b20: 41 13        	addi	t1, t1, -16
   15b22: e5 fb        	bnez	a5, 0x15b12 <_malloc_r+0x3f8>
   15b24: 03 b7 89 00  	ld	a4, 8(s3)
   15b28: 93 c7 f5 ff  	not	a5, a1
   15b2c: f9 8f        	and	a5, a5, a4
   15b2e: 23 b4 f9 00  	sd	a5, 8(s3)
   15b32: 86 05        	slli	a1, a1, 1
   15b34: e3 e5 b7 dc  	bltu	a5, a1, 0x158fe <_malloc_r+0x1e4>
   15b38: e3 83 05 dc  	beqz	a1, 0x158fe <_malloc_r+0x1e4>
   15b3c: 33 f7 f5 00  	and	a4, a1, a5
   15b40: 11 e7        	bnez	a4, 0x15b4c <_malloc_r+0x432>
   15b42: 86 05        	slli	a1, a1, 1
   15b44: 33 f7 f5 00  	and	a4, a1, a5
   15b48: 11 2e        	addiw	t3, t3, 4
   15b4a: 65 df        	beqz	a4, 0x15b42 <_malloc_r+0x428>
   15b4c: 72 86        	mv	a2, t3
   15b4e: 81 bb        	j	0x1589e <_malloc_r+0x184>
   15b50: b7 3b 02 00  	lui	s7, 35
   15b54: 93 8b 0b a9  	addi	s7, s7, -1392
   15b58: 83 a7 0b 00  	lw	a5, 0(s7)
   15b5c: bb 86 47 01  	addw	a3, a5, s4
   15b60: 23 a0 db 00  	sw	a3, 0(s7)
   15b64: 03 b7 01 86  	ld	a4, -1952(gp)
   15b68: fd 57        	addi	a5, zero, -1
   15b6a: 63 0b f7 10  	beq	a4, a5, 0x15c80 <_malloc_r+0x566>
   15b6e: b3 07 8b 41  	sub	a5, s6, s8
   15b72: b5 9f        	addw	a5, a5, a3
   15b74: 23 a0 fb 00  	sw	a5, 0(s7)
   15b78: 93 7c fb 00  	andi	s9, s6, 15
   15b7c: 63 86 0c 0a  	beqz	s9, 0x15c28 <_malloc_r+0x50e>
   15b80: 05 67        	lui	a4, 1
   15b82: 33 0b 9b 41  	sub	s6, s6, s9
   15b86: 13 0c 07 01  	addi	s8, a4, 16
   15b8a: 41 0b        	addi	s6, s6, 16
   15b8c: 33 0c 9c 41  	sub	s8, s8, s9
   15b90: 5a 9a        	add	s4, s4, s6
   15b92: 7d 17        	addi	a4, a4, -1
   15b94: 33 0c 4c 41  	sub	s8, s8, s4
   15b98: 33 7c ec 00  	and	s8, s8, a4
   15b9c: e2 85        	mv	a1, s8
   15b9e: 4a 85        	mv	a0, s2
   15ba0: ef 00 50 73  	jal	0x16ad4 <_sbrk_r>
   15ba4: fd 57        	addi	a5, zero, -1
   15ba6: 63 00 f5 12  	beq	a0, a5, 0x15cc6 <_malloc_r+0x5ac>
   15baa: 33 05 65 41  	sub	a0, a0, s6
   15bae: 9b 06 0c 00  	sext.w	a3, s8
   15bb2: 33 0a 85 01  	add	s4, a0, s8
   15bb6: 03 a7 0b 00  	lw	a4, 0(s7)
   15bba: 23 b8 69 01  	sd	s6, 16(s3)
   15bbe: 93 67 1a 00  	ori	a5, s4, 1
   15bc2: 35 9f        	addw	a4, a4, a3
   15bc4: 23 a0 eb 00  	sw	a4, 0(s7)
   15bc8: 23 34 fb 00  	sd	a5, 8(s6)
   15bcc: e3 08 34 db  	beq	s0, s3, 0x1597c <_malloc_r+0x262>
   15bd0: 7d 46        	addi	a2, zero, 31
   15bd2: 63 7a 56 0b  	bgeu	a2, s5, 0x15c86 <_malloc_r+0x56c>
   15bd6: 14 64        	ld	a3, 8(s0)
   15bd8: 93 87 8a fe  	addi	a5, s5, -24
   15bdc: c1 9b        	andi	a5, a5, -16
   15bde: 85 8a        	andi	a3, a3, 1
   15be0: dd 8e        	or	a3, a3, a5
   15be2: 14 e4        	sd	a3, 8(s0)
   15be4: a5 45        	addi	a1, zero, 9
   15be6: b3 06 f4 00  	add	a3, s0, a5
   15bea: 8c e6        	sd	a1, 8(a3)
   15bec: 8c ea        	sd	a1, 16(a3)
   15bee: 63 65 f6 00  	bltu	a2, a5, 0x15bf8 <_malloc_r+0x4de>
   15bf2: 83 37 8b 00  	ld	a5, 8(s6)
   15bf6: 59 b3        	j	0x1597c <_malloc_r+0x262>
   15bf8: 93 05 04 01  	addi	a1, s0, 16
   15bfc: 4a 85        	mv	a0, s2
   15bfe: ef c0 7f e9  	jal	0x12a94 <_free_r>
   15c02: 03 bb 09 01  	ld	s6, 16(s3)
   15c06: 03 a7 0b 00  	lw	a4, 0(s7)
   15c0a: 83 37 8b 00  	ld	a5, 8(s6)
   15c0e: bd b3        	j	0x1597c <_malloc_r+0x262>
   15c10: 13 0a 0a 02  	addi	s4, s4, 32
   15c14: 0d b3        	j	0x15936 <_malloc_r+0x21c>
   15c16: 9b d6 26 40  	sraiw	a3, a3, 2
   15c1a: 85 47        	addi	a5, zero, 1
   15c1c: b3 97 d7 00  	sll	a5, a5, a3
   15c20: 5d 8d        	or	a0, a0, a5
   15c22: 23 b4 a9 00  	sd	a0, 8(s3)
   15c26: c5 b3        	j	0x15a06 <_malloc_r+0x2ec>
   15c28: 33 0c 4b 01  	add	s8, s6, s4
   15c2c: 33 0c 80 41  	neg	s8, s8
   15c30: 52 1c        	slli	s8, s8, 52
   15c32: 13 5c 4c 03  	srli	s8, s8, 52
   15c36: e2 85        	mv	a1, s8
   15c38: 4a 85        	mv	a0, s2
   15c3a: ef 00 b0 69  	jal	0x16ad4 <_sbrk_r>
   15c3e: fd 57        	addi	a5, zero, -1
   15c40: 81 46        	mv	a3, zero
   15c42: e3 14 f5 f6  	bne	a0, a5, 0x15baa <_malloc_r+0x490>
   15c46: 85 bf        	j	0x15bb6 <_malloc_r+0x49c>
   15c48: 93 06 40 05  	addi	a3, zero, 84
   15c4c: 63 e1 e6 04  	bltu	a3, a4, 0x15c8e <_malloc_r+0x574>
   15c50: 13 d7 c7 00  	srli	a4, a5, 12
   15c54: 9b 05 f7 06  	addiw	a1, a4, 111
   15c58: 9b 95 15 00  	slliw	a1, a1, 1
   15c5c: 9b 06 e7 06  	addiw	a3, a4, 110
   15c60: 8e 05        	slli	a1, a1, 3
   15c62: 61 b3        	j	0x159ea <_malloc_r+0x2d0>
   15c64: 13 07 40 15  	addi	a4, zero, 340
   15c68: 63 61 f7 04  	bltu	a4, a5, 0x15caa <_malloc_r+0x590>
   15c6c: 93 d7 f4 00  	srli	a5, s1, 15
   15c70: 1b 86 87 07  	addiw	a2, a5, 120
   15c74: 9b 16 16 00  	slliw	a3, a2, 1
   15c78: 1b 85 77 07  	addiw	a0, a5, 119
   15c7c: 8e 06        	slli	a3, a3, 3
   15c7e: 95 b6        	j	0x157e2 <_malloc_r+0xc8>
   15c80: 23 b0 61 87  	sd	s6, -1952(gp)
   15c84: d5 bd        	j	0x15b78 <_malloc_r+0x45e>
   15c86: 85 47        	addi	a5, zero, 1
   15c88: 23 34 fb 00  	sd	a5, 8(s6)
   15c8c: b5 b5        	j	0x15af8 <_malloc_r+0x3de>
   15c8e: 93 06 40 15  	addi	a3, zero, 340
   15c92: 63 e0 e6 04  	bltu	a3, a4, 0x15cd2 <_malloc_r+0x5b8>
   15c96: 13 d7 f7 00  	srli	a4, a5, 15
   15c9a: 9b 05 87 07  	addiw	a1, a4, 120
   15c9e: 9b 95 15 00  	slliw	a1, a1, 1
   15ca2: 9b 06 77 07  	addiw	a3, a4, 119
   15ca6: 8e 05        	slli	a1, a1, 3
   15ca8: 89 b3        	j	0x159ea <_malloc_r+0x2d0>
   15caa: 13 07 40 55  	addi	a4, zero, 1364
   15cae: 63 60 f7 04  	bltu	a4, a5, 0x15cee <_malloc_r+0x5d4>
   15cb2: 93 d7 24 01  	srli	a5, s1, 18
   15cb6: 1b 86 d7 07  	addiw	a2, a5, 125
   15cba: 9b 16 16 00  	slliw	a3, a2, 1
   15cbe: 1b 85 c7 07  	addiw	a0, a5, 124
   15cc2: 8e 06        	slli	a3, a3, 3
   15cc4: 39 be        	j	0x157e2 <_malloc_r+0xc8>
   15cc6: c1 1c        	addi	s9, s9, -16
   15cc8: 66 9a        	add	s4, s4, s9
   15cca: 33 0a 6a 41  	sub	s4, s4, s6
   15cce: 81 46        	mv	a3, zero
   15cd0: dd b5        	j	0x15bb6 <_malloc_r+0x49c>
   15cd2: 93 06 40 55  	addi	a3, zero, 1364
   15cd6: 63 e3 e6 02  	bltu	a3, a4, 0x15cfc <_malloc_r+0x5e2>
   15cda: 13 d7 27 01  	srli	a4, a5, 18
   15cde: 9b 05 d7 07  	addiw	a1, a4, 125
   15ce2: 9b 95 15 00  	slliw	a1, a1, 1
   15ce6: 9b 06 c7 07  	addiw	a3, a4, 124
   15cea: 8e 05        	slli	a1, a1, 3
   15cec: fd b9        	j	0x159ea <_malloc_r+0x2d0>
   15cee: 93 06 00 7f  	addi	a3, zero, 2032
   15cf2: 13 06 f0 07  	addi	a2, zero, 127
   15cf6: 13 05 e0 07  	addi	a0, zero, 126
   15cfa: e5 b4        	j	0x157e2 <_malloc_r+0xc8>
   15cfc: 93 05 00 7f  	addi	a1, zero, 2032
   15d00: 93 06 e0 07  	addi	a3, zero, 126
   15d04: dd b1        	j	0x159ea <_malloc_r+0x2d0>
   15d06: 83 b7 89 00  	ld	a5, 8(s3)
   15d0a: 25 b5        	j	0x15b32 <_malloc_r+0x418>

0000000000015d0c <_mbtowc_r>:
   15d0c: b7 07 02 00  	lui	a5, 32
   15d10: 83 b7 07 e8  	ld	a5, -384(a5)
   15d14: 82 87        	jr	a5

0000000000015d16 <__ascii_mbtowc>:
   15d16: 99 c9        	beqz	a1, 0x15d2c <__ascii_mbtowc+0x16>
   15d18: 1d c6        	beqz	a2, 0x15d46 <__ascii_mbtowc+0x30>
   15d1a: 95 ca        	beqz	a3, 0x15d4e <__ascii_mbtowc+0x38>
   15d1c: 83 47 06 00  	lbu	a5, 0(a2)
   15d20: 9c c1        	sw	a5, 0(a1)
   15d22: 03 45 06 00  	lbu	a0, 0(a2)
   15d26: 33 35 a0 00  	snez	a0, a0
   15d2a: 82 80        	ret
   15d2c: 41 11        	addi	sp, sp, -16
   15d2e: 6c 00        	addi	a1, sp, 12
   15d30: 09 ce        	beqz	a2, 0x15d4a <__ascii_mbtowc+0x34>
   15d32: 85 c2        	beqz	a3, 0x15d52 <__ascii_mbtowc+0x3c>
   15d34: 83 47 06 00  	lbu	a5, 0(a2)
   15d38: 9c c1        	sw	a5, 0(a1)
   15d3a: 03 45 06 00  	lbu	a0, 0(a2)
   15d3e: 33 35 a0 00  	snez	a0, a0
   15d42: 41 01        	addi	sp, sp, 16
   15d44: 82 80        	ret
   15d46: 01 45        	mv	a0, zero
   15d48: 82 80        	ret
   15d4a: 01 45        	mv	a0, zero
   15d4c: dd bf        	j	0x15d42 <__ascii_mbtowc+0x2c>
   15d4e: 79 55        	addi	a0, zero, -2
   15d50: 82 80        	ret
   15d52: 79 55        	addi	a0, zero, -2
   15d54: fd b7        	j	0x15d42 <__ascii_mbtowc+0x2c>

0000000000015d56 <memchr>:
   15d56: 93 77 75 00  	andi	a5, a0, 7
   15d5a: 93 f6 f5 0f  	andi	a3, a1, 255
   15d5e: 85 c7        	beqz	a5, 0x15d86 <memchr+0x30>
   15d60: 93 07 f6 ff  	addi	a5, a2, -1
   15d64: 19 ce        	beqz	a2, 0x15d82 <memchr+0x2c>
   15d66: 7d 56        	addi	a2, zero, -1
   15d68: 01 a8        	j	0x15d78 <memchr+0x22>
   15d6a: 05 05        	addi	a0, a0, 1
   15d6c: 13 77 75 00  	andi	a4, a0, 7
   15d70: 01 cf        	beqz	a4, 0x15d88 <memchr+0x32>
   15d72: fd 17        	addi	a5, a5, -1
   15d74: 63 87 c7 00  	beq	a5, a2, 0x15d82 <memchr+0x2c>
   15d78: 03 47 05 00  	lbu	a4, 0(a0)
   15d7c: e3 17 d7 fe  	bne	a4, a3, 0x15d6a <memchr+0x14>
   15d80: 82 80        	ret
   15d82: 01 45        	mv	a0, zero
   15d84: 82 80        	ret
   15d86: b2 87        	mv	a5, a2
   15d88: 1d 47        	addi	a4, zero, 7
   15d8a: 63 75 f7 04  	bgeu	a4, a5, 0x15dd4 <memchr+0x7e>
   15d8e: 93 f5 f5 0f  	andi	a1, a1, 255
   15d92: 13 97 85 00  	slli	a4, a1, 8
   15d96: 2e 97        	add	a4, a4, a1
   15d98: 93 15 07 01  	slli	a1, a4, 16
   15d9c: ba 95        	add	a1, a1, a4
   15d9e: 37 07 02 00  	lui	a4, 32
   15da2: 83 38 07 75  	ld	a7, 1872(a4)
   15da6: 37 07 02 00  	lui	a4, 32
   15daa: 03 38 87 75  	ld	a6, 1880(a4)
   15dae: 13 97 05 02  	slli	a4, a1, 32
   15db2: ba 95        	add	a1, a1, a4
   15db4: 1d 43        	addi	t1, zero, 7
   15db6: 29 a0        	j	0x15dc0 <memchr+0x6a>
   15db8: e1 17        	addi	a5, a5, -8
   15dba: 21 05        	addi	a0, a0, 8
   15dbc: 63 7c f3 00  	bgeu	t1, a5, 0x15dd4 <memchr+0x7e>
   15dc0: 18 61        	ld	a4, 0(a0)
   15dc2: 2d 8f        	xor	a4, a4, a1
   15dc4: 33 06 17 01  	add	a2, a4, a7
   15dc8: 13 47 f7 ff  	not	a4, a4
   15dcc: 71 8f        	and	a4, a4, a2
   15dce: 33 77 07 01  	and	a4, a4, a6
   15dd2: 7d d3        	beqz	a4, 0x15db8 <memchr+0x62>
   15dd4: 33 07 f5 00  	add	a4, a0, a5
   15dd8: 89 e7        	bnez	a5, 0x15de2 <memchr+0x8c>
   15dda: 65 b7        	j	0x15d82 <memchr+0x2c>
   15ddc: 05 05        	addi	a0, a0, 1
   15dde: e3 02 a7 fa  	beq	a4, a0, 0x15d82 <memchr+0x2c>
   15de2: 83 47 05 00  	lbu	a5, 0(a0)
   15de6: e3 9b d7 fe  	bne	a5, a3, 0x15ddc <memchr+0x86>
   15dea: 82 80        	ret

0000000000015dec <__malloc_lock>:
   15dec: 82 80        	ret

0000000000015dee <__malloc_unlock>:
   15dee: 82 80        	ret

0000000000015df0 <_Balloc>:
   15df0: 3c 7d        	ld	a5, 120(a0)
   15df2: 01 11        	addi	sp, sp, -32
   15df4: 22 e8        	sd	s0, 16(sp)
   15df6: 26 e4        	sd	s1, 8(sp)
   15df8: 06 ec        	sd	ra, 24(sp)
   15dfa: 4a e0        	sd	s2, 0(sp)
   15dfc: 2a 84        	mv	s0, a0
   15dfe: ae 84        	mv	s1, a1
   15e00: 85 c3        	beqz	a5, 0x15e20 <_Balloc+0x30>
   15e02: 13 97 34 00  	slli	a4, s1, 3
   15e06: ba 97        	add	a5, a5, a4
   15e08: 88 63        	ld	a0, 0(a5)
   15e0a: 0d c5        	beqz	a0, 0x15e34 <_Balloc+0x44>
   15e0c: 18 61        	ld	a4, 0(a0)
   15e0e: 98 e3        	sd	a4, 0(a5)
   15e10: 23 38 05 00  	sd	zero, 16(a0)
   15e14: e2 60        	ld	ra, 24(sp)
   15e16: 42 64        	ld	s0, 16(sp)
   15e18: a2 64        	ld	s1, 8(sp)
   15e1a: 02 69        	ld	s2, 0(sp)
   15e1c: 05 61        	addi	sp, sp, 32
   15e1e: 82 80        	ret
   15e20: 13 06 10 04  	addi	a2, zero, 65
   15e24: a1 45        	addi	a1, zero, 8
   15e26: ef 30 50 39  	jal	0x199ba <_calloc_r>
   15e2a: 28 fc        	sd	a0, 120(s0)
   15e2c: aa 87        	mv	a5, a0
   15e2e: 71 f9        	bnez	a0, 0x15e02 <_Balloc+0x12>
   15e30: 01 45        	mv	a0, zero
   15e32: cd b7        	j	0x15e14 <_Balloc+0x24>
   15e34: 05 49        	addi	s2, zero, 1
   15e36: 3b 16 99 00  	sllw	a2, s2, s1
   15e3a: 32 89        	mv	s2, a2
   15e3c: 1d 06        	addi	a2, a2, 7
   15e3e: 0a 06        	slli	a2, a2, 2
   15e40: 85 45        	addi	a1, zero, 1
   15e42: 22 85        	mv	a0, s0
   15e44: ef 30 70 37  	jal	0x199ba <_calloc_r>
   15e48: 65 d5        	beqz	a0, 0x15e30 <_Balloc+0x40>
   15e4a: 04 c5        	sw	s1, 8(a0)
   15e4c: 23 26 25 01  	sw	s2, 12(a0)
   15e50: c1 b7        	j	0x15e10 <_Balloc+0x20>

0000000000015e52 <_Bfree>:
   15e52: 81 c9        	beqz	a1, 0x15e62 <_Bfree+0x10>
   15e54: 98 45        	lw	a4, 8(a1)
   15e56: 3c 7d        	ld	a5, 120(a0)
   15e58: 0e 07        	slli	a4, a4, 3
   15e5a: ba 97        	add	a5, a5, a4
   15e5c: 98 63        	ld	a4, 0(a5)
   15e5e: 98 e1        	sd	a4, 0(a1)
   15e60: 8c e3        	sd	a1, 0(a5)
   15e62: 82 80        	ret

0000000000015e64 <__multadd>:
   15e64: 79 71        	addi	sp, sp, -48
   15e66: 22 f0        	sd	s0, 32(sp)
   15e68: c0 49        	lw	s0, 20(a1)
   15e6a: 41 63        	lui	t1, 16
   15e6c: 4a e8        	sd	s2, 16(sp)
   15e6e: 4e e4        	sd	s3, 8(sp)
   15e70: 06 f4        	sd	ra, 40(sp)
   15e72: 26 ec        	sd	s1, 24(sp)
   15e74: 52 e0        	sd	s4, 0(sp)
   15e76: 2e 89        	mv	s2, a1
   15e78: aa 89        	mv	s3, a0
   15e7a: 13 88 85 01  	addi	a6, a1, 24
   15e7e: 81 48        	mv	a7, zero
   15e80: 7d 13        	addi	t1, t1, -1
   15e82: 83 27 08 00  	lw	a5, 0(a6)
   15e86: 11 08        	addi	a6, a6, 4
   15e88: 85 28        	addiw	a7, a7, 1
   15e8a: 33 f7 67 00  	and	a4, a5, t1
   15e8e: 3b 07 c7 02  	mulw	a4, a4, a2
   15e92: 9b d7 07 01  	srliw	a5, a5, 16
   15e96: bb 87 c7 02  	mulw	a5, a5, a2
   15e9a: 35 9f        	addw	a4, a4, a3
   15e9c: 9b 56 07 01  	srliw	a3, a4, 16
   15ea0: 33 77 e3 00  	and	a4, t1, a4
   15ea4: b5 9f        	addw	a5, a5, a3
   15ea6: 9b 96 07 01  	slliw	a3, a5, 16
   15eaa: 35 9f        	addw	a4, a4, a3
   15eac: 23 2e e8 fe  	sw	a4, -4(a6)
   15eb0: 9b d4 07 01  	srliw	s1, a5, 16
   15eb4: 9b d6 07 01  	srliw	a3, a5, 16
   15eb8: e3 c5 88 fc  	blt	a7, s0, 0x15e82 <__multadd+0x1e>
   15ebc: 89 ce        	beqz	a3, 0x15ed6 <__multadd+0x72>
   15ebe: 83 27 c9 00  	lw	a5, 12(s2)
   15ec2: 63 53 f4 02  	bge	s0, a5, 0x15ee8 <__multadd+0x84>
   15ec6: 93 07 44 00  	addi	a5, s0, 4
   15eca: 8a 07        	slli	a5, a5, 2
   15ecc: ca 97        	add	a5, a5, s2
   15ece: 84 c7        	sw	s1, 8(a5)
   15ed0: 05 24        	addiw	s0, s0, 1
   15ed2: 23 2a 89 00  	sw	s0, 20(s2)
   15ed6: a2 70        	ld	ra, 40(sp)
   15ed8: 02 74        	ld	s0, 32(sp)
   15eda: e2 64        	ld	s1, 24(sp)
   15edc: a2 69        	ld	s3, 8(sp)
   15ede: 02 6a        	ld	s4, 0(sp)
   15ee0: 4a 85        	mv	a0, s2
   15ee2: 42 69        	ld	s2, 16(sp)
   15ee4: 45 61        	addi	sp, sp, 48
   15ee6: 82 80        	ret
   15ee8: 83 25 89 00  	lw	a1, 8(s2)
   15eec: 4e 85        	mv	a0, s3
   15eee: 85 25        	addiw	a1, a1, 1
   15ef0: ef f0 1f f0  	jal	0x15df0 <_Balloc>
   15ef4: 2a 8a        	mv	s4, a0
   15ef6: 1d cd        	beqz	a0, 0x15f34 <__multadd+0xd0>
   15ef8: 03 26 49 01  	lw	a2, 20(s2)
   15efc: 93 05 09 01  	addi	a1, s2, 16
   15f00: 41 05        	addi	a0, a0, 16
   15f02: 09 06        	addi	a2, a2, 2
   15f04: 0a 06        	slli	a2, a2, 2
   15f06: ef a0 4f e5  	jal	0x1055a <memcpy>
   15f0a: 03 27 89 00  	lw	a4, 8(s2)
   15f0e: 83 b7 89 07  	ld	a5, 120(s3)
   15f12: 0e 07        	slli	a4, a4, 3
   15f14: ba 97        	add	a5, a5, a4
   15f16: 98 63        	ld	a4, 0(a5)
   15f18: 23 30 e9 00  	sd	a4, 0(s2)
   15f1c: 23 b0 27 01  	sd	s2, 0(a5)
   15f20: 93 07 44 00  	addi	a5, s0, 4
   15f24: 52 89        	mv	s2, s4
   15f26: 8a 07        	slli	a5, a5, 2
   15f28: ca 97        	add	a5, a5, s2
   15f2a: 84 c7        	sw	s1, 8(a5)
   15f2c: 05 24        	addiw	s0, s0, 1
   15f2e: 23 2a 89 00  	sw	s0, 20(s2)
   15f32: 55 b7        	j	0x15ed6 <__multadd+0x72>
   15f34: f9 66        	lui	a3, 30
   15f36: 79 65        	lui	a0, 30
   15f38: 93 86 06 a2  	addi	a3, a3, -1504
   15f3c: 01 46        	mv	a2, zero
   15f3e: 93 05 50 0b  	addi	a1, zero, 181
   15f42: 13 05 05 d2  	addi	a0, a0, -736
   15f46: ef 30 10 22  	jal	0x19966 <__assert_func>

0000000000015f4a <__s2b>:
   15f4a: 39 71        	addi	sp, sp, -64
   15f4c: 22 f8        	sd	s0, 48(sp)
   15f4e: 26 f4        	sd	s1, 40(sp)
   15f50: 4a f0        	sd	s2, 32(sp)
   15f52: 4e ec        	sd	s3, 24(sp)
   15f54: 52 e8        	sd	s4, 16(sp)
   15f56: b6 84        	mv	s1, a3
   15f58: 2e 84        	mv	s0, a1
   15f5a: a1 26        	addiw	a3, a3, 8
   15f5c: a5 45        	addi	a1, zero, 9
   15f5e: 06 fc        	sd	ra, 56(sp)
   15f60: 56 e4        	sd	s5, 8(sp)
   15f62: 5a e0        	sd	s6, 0(sp)
   15f64: a5 47        	addi	a5, zero, 9
   15f66: bb c6 b6 02  	divw	a3, a3, a1
   15f6a: 2a 89        	mv	s2, a0
   15f6c: 32 8a        	mv	s4, a2
   15f6e: ba 89        	mv	s3, a4
   15f70: 63 d1 97 0a  	bge	a5, s1, 0x16012 <__s2b+0xc8>
   15f74: 85 47        	addi	a5, zero, 1
   15f76: 81 45        	mv	a1, zero
   15f78: 9b 97 17 00  	slliw	a5, a5, 1
   15f7c: 85 25        	addiw	a1, a1, 1
   15f7e: e3 cd d7 fe  	blt	a5, a3, 0x15f78 <__s2b+0x2e>
   15f82: 4a 85        	mv	a0, s2
   15f84: ef f0 df e6  	jal	0x15df0 <_Balloc>
   15f88: aa 85        	mv	a1, a0
   15f8a: 51 c5        	beqz	a0, 0x16016 <__s2b+0xcc>
   15f8c: 85 47        	addi	a5, zero, 1
   15f8e: 5c c9        	sw	a5, 20(a0)
   15f90: 23 2c 35 01  	sw	s3, 24(a0)
   15f94: a5 47        	addi	a5, zero, 9
   15f96: 63 db 47 07  	bge	a5, s4, 0x1600c <__s2b+0xc2>
   15f9a: 93 17 0a 02  	slli	a5, s4, 32
   15f9e: 13 0b 94 00  	addi	s6, s0, 9
   15fa2: 81 93        	srli	a5, a5, 32
   15fa4: 9b 0a 0a 00  	sext.w	s5, s4
   15fa8: 3e 94        	add	s0, s0, a5
   15faa: da 89        	mv	s3, s6
   15fac: 83 c6 09 00  	lbu	a3, 0(s3)
   15fb0: 29 46        	addi	a2, zero, 10
   15fb2: 4a 85        	mv	a0, s2
   15fb4: 9b 86 06 fd  	addiw	a3, a3, -48
   15fb8: ef f0 df ea  	jal	0x15e64 <__multadd>
   15fbc: 85 09        	addi	s3, s3, 1
   15fbe: aa 85        	mv	a1, a0
   15fc0: e3 96 89 fe  	bne	s3, s0, 0x15fac <__s2b+0x62>
   15fc4: e1 3a        	addiw	s5, s5, -8
   15fc6: 82 1a        	slli	s5, s5, 32
   15fc8: 93 da 0a 02  	srli	s5, s5, 32
   15fcc: 33 04 5b 01  	add	s0, s6, s5
   15fd0: 63 53 9a 02  	bge	s4, s1, 0x15ff6 <__s2b+0xac>
   15fd4: bb 84 44 41  	subw	s1, s1, s4
   15fd8: 82 14        	slli	s1, s1, 32
   15fda: 81 90        	srli	s1, s1, 32
   15fdc: a2 94        	add	s1, s1, s0
   15fde: 83 46 04 00  	lbu	a3, 0(s0)
   15fe2: 29 46        	addi	a2, zero, 10
   15fe4: 4a 85        	mv	a0, s2
   15fe6: 9b 86 06 fd  	addiw	a3, a3, -48
   15fea: ef f0 bf e7  	jal	0x15e64 <__multadd>
   15fee: 05 04        	addi	s0, s0, 1
   15ff0: aa 85        	mv	a1, a0
   15ff2: e3 16 94 fe  	bne	s0, s1, 0x15fde <__s2b+0x94>
   15ff6: e2 70        	ld	ra, 56(sp)
   15ff8: 42 74        	ld	s0, 48(sp)
   15ffa: a2 74        	ld	s1, 40(sp)
   15ffc: 02 79        	ld	s2, 32(sp)
   15ffe: e2 69        	ld	s3, 24(sp)
   16000: 42 6a        	ld	s4, 16(sp)
   16002: a2 6a        	ld	s5, 8(sp)
   16004: 02 6b        	ld	s6, 0(sp)
   16006: 2e 85        	mv	a0, a1
   16008: 21 61        	addi	sp, sp, 64
   1600a: 82 80        	ret
   1600c: 29 04        	addi	s0, s0, 10
   1600e: 25 4a        	addi	s4, zero, 9
   16010: c1 b7        	j	0x15fd0 <__s2b+0x86>
   16012: 81 45        	mv	a1, zero
   16014: bd b7        	j	0x15f82 <__s2b+0x38>
   16016: f9 66        	lui	a3, 30
   16018: 79 65        	lui	a0, 30
   1601a: 93 86 06 a2  	addi	a3, a3, -1504
   1601e: 01 46        	mv	a2, zero
   16020: 93 05 e0 0c  	addi	a1, zero, 206
   16024: 13 05 05 d2  	addi	a0, a0, -736
   16028: ef 30 f0 13  	jal	0x19966 <__assert_func>

000000000001602c <__hi0bits>:
   1602c: aa 87        	mv	a5, a0
   1602e: 41 67        	lui	a4, 16
   16030: 01 45        	mv	a0, zero
   16032: 63 f5 e7 00  	bgeu	a5, a4, 0x1603c <__hi0bits+0x10>
   16036: 9b 97 07 01  	slliw	a5, a5, 16
   1603a: 41 45        	addi	a0, zero, 16
   1603c: 37 07 00 01  	lui	a4, 4096
   16040: 63 f5 e7 00  	bgeu	a5, a4, 0x1604a <__hi0bits+0x1e>
   16044: 9b 97 87 00  	slliw	a5, a5, 8
   16048: 21 25        	addiw	a0, a0, 8
   1604a: 37 07 00 10  	lui	a4, 65536
   1604e: 63 f5 e7 00  	bgeu	a5, a4, 0x16058 <__hi0bits+0x2c>
   16052: 9b 97 47 00  	slliw	a5, a5, 4
   16056: 11 25        	addiw	a0, a0, 4
   16058: 37 07 00 40  	lui	a4, 262144
   1605c: 63 e6 e7 00  	bltu	a5, a4, 0x16068 <__hi0bits+0x3c>
   16060: 63 c0 07 02  	bltz	a5, 0x16080 <__hi0bits+0x54>
   16064: 05 25        	addiw	a0, a0, 1
   16066: 82 80        	ret
   16068: 9b 97 27 00  	slliw	a5, a5, 2
   1606c: 1b 87 07 00  	sext.w	a4, a5
   16070: 63 49 07 00  	bltz	a4, 0x16082 <__hi0bits+0x56>
   16074: 37 07 00 40  	lui	a4, 262144
   16078: f9 8f        	and	a5, a5, a4
   1607a: 91 e7        	bnez	a5, 0x16086 <__hi0bits+0x5a>
   1607c: 13 05 00 02  	addi	a0, zero, 32
   16080: 82 80        	ret
   16082: 09 25        	addiw	a0, a0, 2
   16084: 82 80        	ret
   16086: 0d 25        	addiw	a0, a0, 3
   16088: 82 80        	ret

000000000001608a <__lo0bits>:
   1608a: 1c 41        	lw	a5, 0(a0)
   1608c: 2a 87        	mv	a4, a0
   1608e: 93 f6 77 00  	andi	a3, a5, 7
   16092: 3e 86        	mv	a2, a5
   16094: 81 ce        	beqz	a3, 0x160ac <__lo0bits+0x22>
   16096: 93 f6 17 00  	andi	a3, a5, 1
   1609a: 01 45        	mv	a0, zero
   1609c: 9d ee        	bnez	a3, 0x160da <__lo0bits+0x50>
   1609e: 09 8a        	andi	a2, a2, 2
   160a0: 25 ce        	beqz	a2, 0x16118 <__lo0bits+0x8e>
   160a2: 9b d7 17 00  	srliw	a5, a5, 1
   160a6: 1c c3        	sw	a5, 0(a4)
   160a8: 05 45        	addi	a0, zero, 1
   160aa: 82 80        	ret
   160ac: 93 96 07 03  	slli	a3, a5, 48
   160b0: c1 92        	srli	a3, a3, 48
   160b2: 01 45        	mv	a0, zero
   160b4: b9 ca        	beqz	a3, 0x1610a <__lo0bits+0x80>
   160b6: 93 f6 f7 0f  	andi	a3, a5, 255
   160ba: a9 c2        	beqz	a3, 0x160fc <__lo0bits+0x72>
   160bc: 93 f6 f7 00  	andi	a3, a5, 15
   160c0: 9d c6        	beqz	a3, 0x160ee <__lo0bits+0x64>
   160c2: 93 f6 37 00  	andi	a3, a5, 3
   160c6: 89 ce        	beqz	a3, 0x160e0 <__lo0bits+0x56>
   160c8: 93 f6 17 00  	andi	a3, a5, 1
   160cc: 81 ea        	bnez	a3, 0x160dc <__lo0bits+0x52>
   160ce: 9b d7 17 00  	srliw	a5, a5, 1
   160d2: 05 25        	addiw	a0, a0, 1
   160d4: 81 e7        	bnez	a5, 0x160dc <__lo0bits+0x52>
   160d6: 13 05 00 02  	addi	a0, zero, 32
   160da: 82 80        	ret
   160dc: 1c c3        	sw	a5, 0(a4)
   160de: 82 80        	ret
   160e0: 9b d7 27 00  	srliw	a5, a5, 2
   160e4: 93 f6 17 00  	andi	a3, a5, 1
   160e8: 09 25        	addiw	a0, a0, 2
   160ea: ed fa        	bnez	a3, 0x160dc <__lo0bits+0x52>
   160ec: cd b7        	j	0x160ce <__lo0bits+0x44>
   160ee: 9b d7 47 00  	srliw	a5, a5, 4
   160f2: 93 f6 37 00  	andi	a3, a5, 3
   160f6: 11 25        	addiw	a0, a0, 4
   160f8: e1 fa        	bnez	a3, 0x160c8 <__lo0bits+0x3e>
   160fa: dd b7        	j	0x160e0 <__lo0bits+0x56>
   160fc: 9b d7 87 00  	srliw	a5, a5, 8
   16100: 93 f6 f7 00  	andi	a3, a5, 15
   16104: 21 25        	addiw	a0, a0, 8
   16106: d5 fe        	bnez	a3, 0x160c2 <__lo0bits+0x38>
   16108: dd b7        	j	0x160ee <__lo0bits+0x64>
   1610a: 9b d7 07 01  	srliw	a5, a5, 16
   1610e: 93 f6 f7 0f  	andi	a3, a5, 255
   16112: 41 45        	addi	a0, zero, 16
   16114: c5 f6        	bnez	a3, 0x160bc <__lo0bits+0x32>
   16116: dd b7        	j	0x160fc <__lo0bits+0x72>
   16118: 9b d7 27 00  	srliw	a5, a5, 2
   1611c: 1c c3        	sw	a5, 0(a4)
   1611e: 09 45        	addi	a0, zero, 2
   16120: 82 80        	ret

0000000000016122 <__i2b>:
   16122: 3c 7d        	ld	a5, 120(a0)
   16124: 01 11        	addi	sp, sp, -32
   16126: 22 e8        	sd	s0, 16(sp)
   16128: 26 e4        	sd	s1, 8(sp)
   1612a: 06 ec        	sd	ra, 24(sp)
   1612c: 2a 84        	mv	s0, a0
   1612e: ae 84        	mv	s1, a1
   16130: 91 cf        	beqz	a5, 0x1614c <__i2b+0x2a>
   16132: 88 67        	ld	a0, 8(a5)
   16134: 1d cd        	beqz	a0, 0x16172 <__i2b+0x50>
   16136: 18 61        	ld	a4, 0(a0)
   16138: 98 e7        	sd	a4, 8(a5)
   1613a: e2 60        	ld	ra, 24(sp)
   1613c: 42 64        	ld	s0, 16(sp)
   1613e: 85 47        	addi	a5, zero, 1
   16140: 82 17        	slli	a5, a5, 32
   16142: 04 cd        	sw	s1, 24(a0)
   16144: 1c e9        	sd	a5, 16(a0)
   16146: a2 64        	ld	s1, 8(sp)
   16148: 05 61        	addi	sp, sp, 32
   1614a: 82 80        	ret
   1614c: 13 06 10 04  	addi	a2, zero, 65
   16150: a1 45        	addi	a1, zero, 8
   16152: ef 30 90 06  	jal	0x199ba <_calloc_r>
   16156: 28 fc        	sd	a0, 120(s0)
   16158: aa 87        	mv	a5, a0
   1615a: 61 fd        	bnez	a0, 0x16132 <__i2b+0x10>
   1615c: f9 66        	lui	a3, 30
   1615e: 79 65        	lui	a0, 30
   16160: 93 86 06 a2  	addi	a3, a3, -1504
   16164: 01 46        	mv	a2, zero
   16166: 93 05 00 14  	addi	a1, zero, 320
   1616a: 13 05 05 d2  	addi	a0, a0, -736
   1616e: ef 30 80 7f  	jal	0x19966 <__assert_func>
   16172: 13 06 40 02  	addi	a2, zero, 36
   16176: 85 45        	addi	a1, zero, 1
   16178: 22 85        	mv	a0, s0
   1617a: ef 30 10 04  	jal	0x199ba <_calloc_r>
   1617e: 79 dd        	beqz	a0, 0x1615c <__i2b+0x3a>
   16180: 85 47        	addi	a5, zero, 1
   16182: 86 17        	slli	a5, a5, 33
   16184: 85 07        	addi	a5, a5, 1
   16186: 1c e5        	sd	a5, 8(a0)
   16188: 4d bf        	j	0x1613a <__i2b+0x18>

000000000001618a <__multiply>:
   1618a: 79 71        	addi	sp, sp, -48
   1618c: 4a e8        	sd	s2, 16(sp)
   1618e: 4e e4        	sd	s3, 8(sp)
   16190: 03 29 46 01  	lw	s2, 20(a2)
   16194: 83 a9 45 01  	lw	s3, 20(a1)
   16198: 26 ec        	sd	s1, 24(sp)
   1619a: 52 e0        	sd	s4, 0(sp)
   1619c: 06 f4        	sd	ra, 40(sp)
   1619e: 22 f0        	sd	s0, 32(sp)
   161a0: 2e 8a        	mv	s4, a1
   161a2: b2 84        	mv	s1, a2
   161a4: 63 c7 29 01  	blt	s3, s2, 0x161b2 <__multiply+0x28>
   161a8: ca 87        	mv	a5, s2
   161aa: ae 84        	mv	s1, a1
   161ac: 4e 89        	mv	s2, s3
   161ae: 32 8a        	mv	s4, a2
   161b0: be 89        	mv	s3, a5
   161b2: dc 44        	lw	a5, 12(s1)
   161b4: 3b 04 39 01  	addw	s0, s2, s3
   161b8: 8c 44        	lw	a1, 8(s1)
   161ba: 63 d3 87 00  	bge	a5, s0, 0x161c0 <__multiply+0x36>
   161be: 85 25        	addiw	a1, a1, 1
   161c0: ef f0 1f c3  	jal	0x15df0 <_Balloc>
   161c4: 63 09 05 16  	beqz	a0, 0x16336 <__multiply+0x1ac>
   161c8: 13 03 85 01  	addi	t1, a0, 24
   161cc: 93 18 24 00  	slli	a7, s0, 2
   161d0: 9a 98        	add	a7, a7, t1
   161d2: 63 75 13 03  	bgeu	t1, a7, 0x161fc <__multiply+0x72>
   161d6: b3 87 a8 40  	sub	a5, a7, a0
   161da: 9d 17        	addi	a5, a5, -25
   161dc: 89 83        	srli	a5, a5, 2
   161de: 85 07        	addi	a5, a5, 1
   161e0: 1a 87        	mv	a4, t1
   161e2: 73 26 20 c2  	csrr	a2, vlenb
   161e6: d7 76 00 05  	<unknown>
   161ea: 57 3c 00 5e  	<unknown>
   161ee: d7 f6 07 05  	<unknown>
   161f2: 27 6c 07 02  	<unknown>
   161f6: 95 8f        	sub	a5, a5, a3
   161f8: 32 97        	add	a4, a4, a2
   161fa: f5 fb        	bnez	a5, 0x161ee <__multiply+0x64>
   161fc: 13 08 8a 01  	addi	a6, s4, 24
   16200: 13 9e 29 00  	slli	t3, s3, 2
   16204: 93 8e 84 01  	addi	t4, s1, 24
   16208: 93 15 29 00  	slli	a1, s2, 2
   1620c: 42 9e        	add	t3, t3, a6
   1620e: f6 95        	add	a1, a1, t4
   16210: 63 7a c8 0f  	bgeu	a6, t3, 0x16304 <__multiply+0x17a>
   16214: 93 87 94 01  	addi	a5, s1, 25
   16218: 11 4f        	addi	t5, zero, 4
   1621a: 63 f7 f5 10  	bgeu	a1, a5, 0x16328 <__multiply+0x19e>
   1621e: c1 66        	lui	a3, 16
   16220: fd 16        	addi	a3, a3, -1
   16222: 09 a8        	j	0x16234 <__multiply+0xaa>
   16224: 9b df 0f 01  	srliw	t6, t6, 16
   16228: 63 9d 0f 06  	bnez	t6, 0x162a2 <__multiply+0x118>
   1622c: 11 08        	addi	a6, a6, 4
   1622e: 11 03        	addi	t1, t1, 4
   16230: 63 7a c8 0d  	bgeu	a6, t3, 0x16304 <__multiply+0x17a>
   16234: 83 2f 08 00  	lw	t6, 0(a6)
   16238: b3 f3 df 00  	and	t2, t6, a3
   1623c: e3 84 03 fe  	beqz	t2, 0x16224 <__multiply+0x9a>
   16240: 9a 82        	mv	t0, t1
   16242: f6 8f        	mv	t6, t4
   16244: 81 44        	mv	s1, zero
   16246: 83 a7 0f 00  	lw	a5, 0(t6)
   1624a: 03 a6 02 00  	lw	a2, 0(t0)
   1624e: 91 02        	addi	t0, t0, 4
   16250: 33 f7 d7 00  	and	a4, a5, a3
   16254: 3b 07 77 02  	mulw	a4, a4, t2
   16258: 9b d7 07 01  	srliw	a5, a5, 16
   1625c: 33 79 d6 00  	and	s2, a2, a3
   16260: 1b 56 06 01  	srliw	a2, a2, 16
   16264: 91 0f        	addi	t6, t6, 4
   16266: bb 87 77 02  	mulw	a5, a5, t2
   1626a: 3b 07 27 01  	addw	a4, a4, s2
   1626e: 25 9f        	addw	a4, a4, s1
   16270: 9b 54 07 01  	srliw	s1, a4, 16
   16274: 75 8f        	and	a4, a4, a3
   16276: b1 9f        	addw	a5, a5, a2
   16278: a5 9f        	addw	a5, a5, s1
   1627a: 1b 96 07 01  	slliw	a2, a5, 16
   1627e: 51 8f        	or	a4, a4, a2
   16280: 23 ae e2 fe  	sw	a4, -4(t0)
   16284: 9b d4 07 01  	srliw	s1, a5, 16
   16288: 1b d7 07 01  	srliw	a4, a5, 16
   1628c: e3 ed bf fa  	bltu	t6, a1, 0x16246 <__multiply+0xbc>
   16290: b3 07 e3 01  	add	a5, t1, t5
   16294: 98 c3        	sw	a4, 0(a5)
   16296: 83 2f 08 00  	lw	t6, 0(a6)
   1629a: 9b df 0f 01  	srliw	t6, t6, 16
   1629e: e3 87 0f f8  	beqz	t6, 0x1622c <__multiply+0xa2>
   162a2: 03 27 03 00  	lw	a4, 0(t1)
   162a6: 9a 82        	mv	t0, t1
   162a8: 76 86        	mv	a2, t4
   162aa: ba 83        	mv	t2, a4
   162ac: 81 44        	mv	s1, zero
   162ae: 1c 42        	lw	a5, 0(a2)
   162b0: 9b d9 03 01  	srliw	s3, t2, 16
   162b4: 75 8f        	and	a4, a4, a3
   162b6: f5 8f        	and	a5, a5, a3
   162b8: bb 87 f7 03  	mulw	a5, a5, t6
   162bc: 83 a3 42 00  	lw	t2, 4(t0)
   162c0: 91 02        	addi	t0, t0, 4
   162c2: 11 06        	addi	a2, a2, 4
   162c4: 33 f9 d3 00  	and	s2, t2, a3
   162c8: a5 9f        	addw	a5, a5, s1
   162ca: bb 87 37 01  	addw	a5, a5, s3
   162ce: 9b 94 07 01  	slliw	s1, a5, 16
   162d2: 45 8f        	or	a4, a4, s1
   162d4: 23 ae e2 fe  	sw	a4, -4(t0)
   162d8: 03 57 e6 ff  	lhu	a4, -2(a2)
   162dc: 9b d7 07 01  	srliw	a5, a5, 16
   162e0: 3b 07 f7 03  	mulw	a4, a4, t6
   162e4: 3b 07 27 01  	addw	a4, a4, s2
   162e8: b9 9f        	addw	a5, a5, a4
   162ea: 1b 87 07 00  	sext.w	a4, a5
   162ee: 9b d4 07 01  	srliw	s1, a5, 16
   162f2: e3 6e b6 fa  	bltu	a2, a1, 0x162ae <__multiply+0x124>
   162f6: 33 07 e3 01  	add	a4, t1, t5
   162fa: 1c c3        	sw	a5, 0(a4)
   162fc: 11 08        	addi	a6, a6, 4
   162fe: 11 03        	addi	t1, t1, 4
   16300: e3 6a c8 f3  	bltu	a6, t3, 0x16234 <__multiply+0xaa>
   16304: 63 45 80 00  	bgtz	s0, 0x1630e <__multiply+0x184>
   16308: 39 a0        	j	0x16316 <__multiply+0x18c>
   1630a: 7d 34        	addiw	s0, s0, -1
   1630c: 09 c4        	beqz	s0, 0x16316 <__multiply+0x18c>
   1630e: 83 a7 c8 ff  	lw	a5, -4(a7)
   16312: f1 18        	addi	a7, a7, -4
   16314: fd db        	beqz	a5, 0x1630a <__multiply+0x180>
   16316: a2 70        	ld	ra, 40(sp)
   16318: 40 c9        	sw	s0, 20(a0)
   1631a: 02 74        	ld	s0, 32(sp)
   1631c: e2 64        	ld	s1, 24(sp)
   1631e: 42 69        	ld	s2, 16(sp)
   16320: a2 69        	ld	s3, 8(sp)
   16322: 02 6a        	ld	s4, 0(sp)
   16324: 45 61        	addi	sp, sp, 48
   16326: 82 80        	ret
   16328: b3 87 95 40  	sub	a5, a1, s1
   1632c: 9d 17        	addi	a5, a5, -25
   1632e: f1 9b        	andi	a5, a5, -4
   16330: 13 8f 47 00  	addi	t5, a5, 4
   16334: ed b5        	j	0x1621e <__multiply+0x94>
   16336: f9 66        	lui	a3, 30
   16338: 79 65        	lui	a0, 30
   1633a: 93 86 06 a2  	addi	a3, a3, -1504
   1633e: 01 46        	mv	a2, zero
   16340: 93 05 d0 15  	addi	a1, zero, 349
   16344: 13 05 05 d2  	addi	a0, a0, -736
   16348: ef 30 e0 61  	jal	0x19966 <__assert_func>

000000000001634c <__pow5mult>:
   1634c: 79 71        	addi	sp, sp, -48
   1634e: 22 f0        	sd	s0, 32(sp)
   16350: 4a e8        	sd	s2, 16(sp)
   16352: 4e e4        	sd	s3, 8(sp)
   16354: 06 f4        	sd	ra, 40(sp)
   16356: 26 ec        	sd	s1, 24(sp)
   16358: 93 77 36 00  	andi	a5, a2, 3
   1635c: 32 84        	mv	s0, a2
   1635e: aa 89        	mv	s3, a0
   16360: 2e 89        	mv	s2, a1
   16362: a5 eb        	bnez	a5, 0x163d2 <__pow5mult+0x86>
   16364: 09 84        	srai	s0, s0, 2
   16366: 21 c4        	beqz	s0, 0x163ae <__pow5mult+0x62>
   16368: 83 b4 09 07  	ld	s1, 112(s3)
   1636c: bd cc        	beqz	s1, 0x163ea <__pow5mult+0x9e>
   1636e: 93 77 14 00  	andi	a5, s0, 1
   16372: 89 eb        	bnez	a5, 0x16384 <__pow5mult+0x38>
   16374: 05 84        	srai	s0, s0, 1
   16376: 05 cc        	beqz	s0, 0x163ae <__pow5mult+0x62>
   16378: 88 60        	ld	a0, 0(s1)
   1637a: 31 c1        	beqz	a0, 0x163be <__pow5mult+0x72>
   1637c: aa 84        	mv	s1, a0
   1637e: 93 77 14 00  	andi	a5, s0, 1
   16382: ed db        	beqz	a5, 0x16374 <__pow5mult+0x28>
   16384: 26 86        	mv	a2, s1
   16386: ca 85        	mv	a1, s2
   16388: 4e 85        	mv	a0, s3
   1638a: ef f0 1f e0  	jal	0x1618a <__multiply>
   1638e: 63 0d 09 00  	beqz	s2, 0x163a8 <__pow5mult+0x5c>
   16392: 03 27 89 00  	lw	a4, 8(s2)
   16396: 83 b7 89 07  	ld	a5, 120(s3)
   1639a: 0e 07        	slli	a4, a4, 3
   1639c: ba 97        	add	a5, a5, a4
   1639e: 98 63        	ld	a4, 0(a5)
   163a0: 23 30 e9 00  	sd	a4, 0(s2)
   163a4: 23 b0 27 01  	sd	s2, 0(a5)
   163a8: 05 84        	srai	s0, s0, 1
   163aa: 2a 89        	mv	s2, a0
   163ac: 71 f4        	bnez	s0, 0x16378 <__pow5mult+0x2c>
   163ae: a2 70        	ld	ra, 40(sp)
   163b0: 02 74        	ld	s0, 32(sp)
   163b2: e2 64        	ld	s1, 24(sp)
   163b4: a2 69        	ld	s3, 8(sp)
   163b6: 4a 85        	mv	a0, s2
   163b8: 42 69        	ld	s2, 16(sp)
   163ba: 45 61        	addi	sp, sp, 48
   163bc: 82 80        	ret
   163be: 26 86        	mv	a2, s1
   163c0: a6 85        	mv	a1, s1
   163c2: 4e 85        	mv	a0, s3
   163c4: ef f0 7f dc  	jal	0x1618a <__multiply>
   163c8: 88 e0        	sd	a0, 0(s1)
   163ca: 23 30 05 00  	sd	zero, 0(a0)
   163ce: aa 84        	mv	s1, a0
   163d0: 7d b7        	j	0x1637e <__pow5mult+0x32>
   163d2: fd 37        	addiw	a5, a5, -1
   163d4: 79 67        	lui	a4, 30
   163d6: 13 07 07 d8  	addi	a4, a4, -640
   163da: 8a 07        	slli	a5, a5, 2
   163dc: ba 97        	add	a5, a5, a4
   163de: 90 43        	lw	a2, 0(a5)
   163e0: 81 46        	mv	a3, zero
   163e2: ef f0 3f a8  	jal	0x15e64 <__multadd>
   163e6: 2a 89        	mv	s2, a0
   163e8: b5 bf        	j	0x16364 <__pow5mult+0x18>
   163ea: 85 45        	addi	a1, zero, 1
   163ec: 4e 85        	mv	a0, s3
   163ee: ef f0 3f a0  	jal	0x15df0 <_Balloc>
   163f2: aa 84        	mv	s1, a0
   163f4: 19 c9        	beqz	a0, 0x1640a <__pow5mult+0xbe>
   163f6: 93 07 10 27  	addi	a5, zero, 625
   163fa: 1c cd        	sw	a5, 24(a0)
   163fc: 85 47        	addi	a5, zero, 1
   163fe: 5c c9        	sw	a5, 20(a0)
   16400: 23 b8 a9 06  	sd	a0, 112(s3)
   16404: 23 30 05 00  	sd	zero, 0(a0)
   16408: 9d b7        	j	0x1636e <__pow5mult+0x22>
   1640a: f9 66        	lui	a3, 30
   1640c: 79 65        	lui	a0, 30
   1640e: 93 86 06 a2  	addi	a3, a3, -1504
   16412: 01 46        	mv	a2, zero
   16414: 93 05 00 14  	addi	a1, zero, 320
   16418: 13 05 05 d2  	addi	a0, a0, -736
   1641c: ef 30 a0 54  	jal	0x19966 <__assert_func>

0000000000016420 <__lshift>:
   16420: 39 71        	addi	sp, sp, -64
   16422: 26 f4        	sd	s1, 40(sp)
   16424: c4 49        	lw	s1, 20(a1)
   16426: dc 45        	lw	a5, 12(a1)
   16428: 5a e0        	sd	s6, 0(sp)
   1642a: 1b 5b 56 40  	sraiw	s6, a2, 5
   1642e: bb 04 9b 00  	addw	s1, s6, s1
   16432: 22 f8        	sd	s0, 48(sp)
   16434: 4a f0        	sd	s2, 32(sp)
   16436: 4e ec        	sd	s3, 24(sp)
   16438: 52 e8        	sd	s4, 16(sp)
   1643a: 56 e4        	sd	s5, 8(sp)
   1643c: 06 fc        	sd	ra, 56(sp)
   1643e: 9b 89 14 00  	addiw	s3, s1, 1
   16442: 2e 89        	mv	s2, a1
   16444: b2 8a        	mv	s5, a2
   16446: 8c 45        	lw	a1, 8(a1)
   16448: 2a 8a        	mv	s4, a0
   1644a: 13 54 56 40  	srai	s0, a2, 5
   1644e: 63 d7 37 01  	bge	a5, s3, 0x1645c <__lshift+0x3c>
   16452: 9b 97 17 00  	slliw	a5, a5, 1
   16456: 85 25        	addiw	a1, a1, 1
   16458: e3 cd 37 ff  	blt	a5, s3, 0x16452 <__lshift+0x32>
   1645c: 52 85        	mv	a0, s4
   1645e: ef f0 3f 99  	jal	0x15df0 <_Balloc>
   16462: 79 cd        	beqz	a0, 0x16540 <__lshift+0x120>
   16464: 93 06 85 01  	addi	a3, a0, 24
   16468: 63 56 80 02  	blez	s0, 0x16494 <__lshift+0x74>
   1646c: a2 87        	mv	a5, s0
   1646e: 36 87        	mv	a4, a3
   16470: f3 25 20 c2  	csrr	a1, vlenb
   16474: 57 76 00 05  	<unknown>
   16478: 57 3c 00 5e  	<unknown>
   1647c: 57 f6 07 05  	<unknown>
   16480: 27 6c 07 02  	<unknown>
   16484: 91 8f        	sub	a5, a5, a2
   16486: 2e 97        	add	a4, a4, a1
   16488: f5 fb        	bnez	a5, 0x1647c <__lshift+0x5c>
   1648a: 93 17 0b 02  	slli	a5, s6, 32
   1648e: 13 db e7 01  	srli	s6, a5, 30
   16492: da 96        	add	a3, a3, s6
   16494: 83 28 49 01  	lw	a7, 20(s2)
   16498: 93 07 89 01  	addi	a5, s2, 24
   1649c: 13 f6 fa 01  	andi	a2, s5, 31
   164a0: 8a 08        	slli	a7, a7, 2
   164a2: be 98        	add	a7, a7, a5
   164a4: 3d ce        	beqz	a2, 0x16522 <__lshift+0x102>
   164a6: 13 03 00 02  	addi	t1, zero, 32
   164aa: 3b 03 c3 40  	subw	t1, t1, a2
   164ae: b6 85        	mv	a1, a3
   164b0: 01 47        	mv	a4, zero
   164b2: 03 a8 07 00  	lw	a6, 0(a5)
   164b6: 91 05        	addi	a1, a1, 4
   164b8: 91 07        	addi	a5, a5, 4
   164ba: 3b 18 c8 00  	sllw	a6, a6, a2
   164be: 33 67 07 01  	or	a4, a4, a6
   164c2: 23 ae e5 fe  	sw	a4, -4(a1)
   164c6: 03 a8 c7 ff  	lw	a6, -4(a5)
   164ca: 3b 57 68 00  	srlw	a4, a6, t1
   164ce: 3a 88        	mv	a6, a4
   164d0: e3 e1 17 ff  	bltu	a5, a7, 0x164b2 <__lshift+0x92>
   164d4: 93 07 99 01  	addi	a5, s2, 25
   164d8: 11 46        	addi	a2, zero, 4
   164da: 63 e8 f8 00  	bltu	a7, a5, 0x164ea <__lshift+0xca>
   164de: b3 87 28 41  	sub	a5, a7, s2
   164e2: 9d 17        	addi	a5, a5, -25
   164e4: f1 9b        	andi	a5, a5, -4
   164e6: 13 86 47 00  	addi	a2, a5, 4
   164ea: b2 96        	add	a3, a3, a2
   164ec: 23 a0 06 01  	sw	a6, 0(a3)
   164f0: 11 e3        	bnez	a4, 0x164f4 <__lshift+0xd4>
   164f2: a6 89        	mv	s3, s1
   164f4: 03 27 89 00  	lw	a4, 8(s2)
   164f8: 83 37 8a 07  	ld	a5, 120(s4)
   164fc: e2 70        	ld	ra, 56(sp)
   164fe: 0e 07        	slli	a4, a4, 3
   16500: ba 97        	add	a5, a5, a4
   16502: 98 63        	ld	a4, 0(a5)
   16504: 23 2a 35 01  	sw	s3, 20(a0)
   16508: 42 74        	ld	s0, 48(sp)
   1650a: 23 30 e9 00  	sd	a4, 0(s2)
   1650e: 23 b0 27 01  	sd	s2, 0(a5)
   16512: a2 74        	ld	s1, 40(sp)
   16514: 02 79        	ld	s2, 32(sp)
   16516: e2 69        	ld	s3, 24(sp)
   16518: 42 6a        	ld	s4, 16(sp)
   1651a: a2 6a        	ld	s5, 8(sp)
   1651c: 02 6b        	ld	s6, 0(sp)
   1651e: 21 61        	addi	sp, sp, 64
   16520: 82 80        	ret
   16522: 98 43        	lw	a4, 0(a5)
   16524: 91 07        	addi	a5, a5, 4
   16526: 91 06        	addi	a3, a3, 4
   16528: 23 ae e6 fe  	sw	a4, -4(a3)
   1652c: e3 f3 17 fd  	bgeu	a5, a7, 0x164f2 <__lshift+0xd2>
   16530: 98 43        	lw	a4, 0(a5)
   16532: 91 07        	addi	a5, a5, 4
   16534: 91 06        	addi	a3, a3, 4
   16536: 23 ae e6 fe  	sw	a4, -4(a3)
   1653a: e3 e4 17 ff  	bltu	a5, a7, 0x16522 <__lshift+0x102>
   1653e: 55 bf        	j	0x164f2 <__lshift+0xd2>
   16540: f9 66        	lui	a3, 30
   16542: 79 65        	lui	a0, 30
   16544: 93 86 06 a2  	addi	a3, a3, -1504
   16548: 01 46        	mv	a2, zero
   1654a: 93 05 90 1d  	addi	a1, zero, 473
   1654e: 13 05 05 d2  	addi	a0, a0, -736
   16552: ef 30 40 41  	jal	0x19966 <__assert_func>

0000000000016556 <__mcmp>:
   16556: dc 49        	lw	a5, 20(a1)
   16558: 2a 88        	mv	a6, a0
   1655a: 48 49        	lw	a0, 20(a0)
   1655c: 1d 9d        	subw	a0, a0, a5
   1655e: 1d e5        	bnez	a0, 0x1658c <__mcmp+0x36>
   16560: 8a 07        	slli	a5, a5, 2
   16562: 61 08        	addi	a6, a6, 24
   16564: e1 05        	addi	a1, a1, 24
   16566: 33 07 f8 00  	add	a4, a6, a5
   1656a: ae 97        	add	a5, a5, a1
   1656c: 19 a0        	j	0x16572 <__mcmp+0x1c>
   1656e: 63 7f e8 00  	bgeu	a6, a4, 0x1658c <__mcmp+0x36>
   16572: 03 26 c7 ff  	lw	a2, -4(a4)
   16576: 83 a6 c7 ff  	lw	a3, -4(a5)
   1657a: 71 17        	addi	a4, a4, -4
   1657c: f1 17        	addi	a5, a5, -4
   1657e: e3 08 d6 fe  	beq	a2, a3, 0x1656e <__mcmp+0x18>
   16582: 05 45        	addi	a0, zero, 1
   16584: 63 74 d6 00  	bgeu	a2, a3, 0x1658c <__mcmp+0x36>
   16588: 7d 55        	addi	a0, zero, -1
   1658a: 82 80        	ret
   1658c: 82 80        	ret

000000000001658e <__mdiff>:
   1658e: 01 11        	addi	sp, sp, -32
   16590: 5c 4a        	lw	a5, 20(a2)
   16592: 4a e0        	sd	s2, 0(sp)
   16594: 03 a9 45 01  	lw	s2, 20(a1)
   16598: 22 e8        	sd	s0, 16(sp)
   1659a: 26 e4        	sd	s1, 8(sp)
   1659c: 06 ec        	sd	ra, 24(sp)
   1659e: 3b 09 f9 40  	subw	s2, s2, a5
   165a2: ae 84        	mv	s1, a1
   165a4: 32 84        	mv	s0, a2
   165a6: 63 1c 09 02  	bnez	s2, 0x165de <__mdiff+0x50>
   165aa: 93 96 27 00  	slli	a3, a5, 2
   165ae: e1 05        	addi	a1, a1, 24
   165b0: 13 07 86 01  	addi	a4, a2, 24
   165b4: b3 87 d5 00  	add	a5, a1, a3
   165b8: 36 97        	add	a4, a4, a3
   165ba: 19 a0        	j	0x165c0 <__mdiff+0x32>
   165bc: 63 f6 f5 12  	bgeu	a1, a5, 0x166e8 <__mdiff+0x15a>
   165c0: 03 a6 c7 ff  	lw	a2, -4(a5)
   165c4: 83 26 c7 ff  	lw	a3, -4(a4)
   165c8: f1 17        	addi	a5, a5, -4
   165ca: 71 17        	addi	a4, a4, -4
   165cc: e3 08 d6 fe  	beq	a2, a3, 0x165bc <__mdiff+0x2e>
   165d0: 63 7a d6 00  	bgeu	a2, a3, 0x165e4 <__mdiff+0x56>
   165d4: a6 87        	mv	a5, s1
   165d6: 05 49        	addi	s2, zero, 1
   165d8: a2 84        	mv	s1, s0
   165da: 3e 84        	mv	s0, a5
   165dc: 21 a0        	j	0x165e4 <__mdiff+0x56>
   165de: e3 4b 09 fe  	bltz	s2, 0x165d4 <__mdiff+0x46>
   165e2: 01 49        	mv	s2, zero
   165e4: 8c 44        	lw	a1, 8(s1)
   165e6: ef f0 bf 80  	jal	0x15df0 <_Balloc>
   165ea: 63 01 05 14  	beqz	a0, 0x1672c <__mdiff+0x19e>
   165ee: 03 a3 44 01  	lw	t1, 20(s1)
   165f2: 03 2f 44 01  	lw	t5, 20(s0)
   165f6: 93 82 84 01  	addi	t0, s1, 24
   165fa: 13 1e 23 00  	slli	t3, t1, 2
   165fe: 13 08 84 01  	addi	a6, s0, 24
   16602: 0a 0f        	slli	t5, t5, 2
   16604: 93 0f 85 01  	addi	t6, a0, 24
   16608: c1 68        	lui	a7, 16
   1660a: 23 28 25 01  	sw	s2, 16(a0)
   1660e: 16 9e        	add	t3, t3, t0
   16610: 42 9f        	add	t5, t5, a6
   16612: 7e 86        	mv	a2, t6
   16614: 96 8e        	mv	t4, t0
   16616: 81 45        	mv	a1, zero
   16618: fd 18        	addi	a7, a7, -1
   1661a: 03 a7 0e 00  	lw	a4, 0(t4)
   1661e: 83 26 08 00  	lw	a3, 0(a6)
   16622: 11 06        	addi	a2, a2, 4
   16624: b3 77 17 01  	and	a5, a4, a7
   16628: b3 f3 16 01  	and	t2, a3, a7
   1662c: bb 87 77 40  	subw	a5, a5, t2
   16630: ad 9f        	addw	a5, a5, a1
   16632: 9b d6 06 01  	srliw	a3, a3, 16
   16636: 1b 57 07 01  	srliw	a4, a4, 16
   1663a: 15 9f        	subw	a4, a4, a3
   1663c: 9b d6 07 41  	sraiw	a3, a5, 16
   16640: 35 9f        	addw	a4, a4, a3
   16642: b3 f7 f8 00  	and	a5, a7, a5
   16646: 9b 16 07 01  	slliw	a3, a4, 16
   1664a: d5 8f        	or	a5, a5, a3
   1664c: 81 27        	sext.w	a5, a5
   1664e: 11 08        	addi	a6, a6, 4
   16650: 23 2e f6 fe  	sw	a5, -4(a2)
   16654: 91 0e        	addi	t4, t4, 4
   16656: 9b 55 07 41  	sraiw	a1, a4, 16
   1665a: e3 60 e8 fd  	bltu	a6, t5, 0x1661a <__mdiff+0x8c>
   1665e: 33 07 8f 40  	sub	a4, t5, s0
   16662: 1d 17        	addi	a4, a4, -25
   16664: 65 04        	addi	s0, s0, 25
   16666: 93 5e 27 00  	srli	t4, a4, 2
   1666a: 11 47        	addi	a4, zero, 4
   1666c: 63 65 8f 00  	bltu	t5, s0, 0x16676 <__mdiff+0xe8>
   16670: 13 87 1e 00  	addi	a4, t4, 1
   16674: 0a 07        	slli	a4, a4, 2
   16676: ba 92        	add	t0, t0, a4
   16678: 33 88 ef 00  	add	a6, t6, a4
   1667c: c1 68        	lui	a7, 16
   1667e: 42 86        	mv	a2, a6
   16680: 96 86        	mv	a3, t0
   16682: fd 18        	addi	a7, a7, -1
   16684: 63 f0 c2 09  	bgeu	t0, t3, 0x16704 <__mdiff+0x176>
   16688: 9c 42        	lw	a5, 0(a3)
   1668a: 9b 8e 05 00  	sext.w	t4, a1
   1668e: 11 06        	addi	a2, a2, 4
   16690: 33 f7 17 01  	and	a4, a5, a7
   16694: 2d 9f        	addw	a4, a4, a1
   16696: 1b 57 07 41  	sraiw	a4, a4, 16
   1669a: 9b d5 07 01  	srliw	a1, a5, 16
   1669e: 2d 9f        	addw	a4, a4, a1
   166a0: bb 87 d7 01  	addw	a5, a5, t4
   166a4: 9b 15 07 01  	slliw	a1, a4, 16
   166a8: b3 f7 17 01  	and	a5, a5, a7
   166ac: cd 8f        	or	a5, a5, a1
   166ae: 81 27        	sext.w	a5, a5
   166b0: 91 06        	addi	a3, a3, 4
   166b2: 23 2e f6 fe  	sw	a5, -4(a2)
   166b6: 9b 55 07 41  	sraiw	a1, a4, 16
   166ba: e3 e7 c6 fd  	bltu	a3, t3, 0x16688 <__mdiff+0xfa>
   166be: 7d 1e        	addi	t3, t3, -1
   166c0: 33 0e 5e 40  	sub	t3, t3, t0
   166c4: 13 7e ce ff  	andi	t3, t3, -4
   166c8: 33 07 c8 01  	add	a4, a6, t3
   166cc: 91 e7        	bnez	a5, 0x166d8 <__mdiff+0x14a>
   166ce: 83 27 c7 ff  	lw	a5, -4(a4)
   166d2: 7d 33        	addiw	t1, t1, -1
   166d4: 71 17        	addi	a4, a4, -4
   166d6: e5 df        	beqz	a5, 0x166ce <__mdiff+0x140>
   166d8: e2 60        	ld	ra, 24(sp)
   166da: 42 64        	ld	s0, 16(sp)
   166dc: 23 2a 65 00  	sw	t1, 20(a0)
   166e0: a2 64        	ld	s1, 8(sp)
   166e2: 02 69        	ld	s2, 0(sp)
   166e4: 05 61        	addi	sp, sp, 32
   166e6: 82 80        	ret
   166e8: 81 45        	mv	a1, zero
   166ea: ef f0 6f f0  	jal	0x15df0 <_Balloc>
   166ee: 05 c5        	beqz	a0, 0x16716 <__mdiff+0x188>
   166f0: e2 60        	ld	ra, 24(sp)
   166f2: 42 64        	ld	s0, 16(sp)
   166f4: 85 47        	addi	a5, zero, 1
   166f6: 5c c9        	sw	a5, 20(a0)
   166f8: 23 2c 05 00  	sw	zero, 24(a0)
   166fc: a2 64        	ld	s1, 8(sp)
   166fe: 02 69        	ld	s2, 0(sp)
   16700: 05 61        	addi	sp, sp, 32
   16702: 82 80        	ret
   16704: 01 47        	mv	a4, zero
   16706: 63 74 8f 00  	bgeu	t5, s0, 0x1670e <__mdiff+0x180>
   1670a: 7e 97        	add	a4, a4, t6
   1670c: c1 b7        	j	0x166cc <__mdiff+0x13e>
   1670e: 13 97 2e 00  	slli	a4, t4, 2
   16712: 7e 97        	add	a4, a4, t6
   16714: 65 bf        	j	0x166cc <__mdiff+0x13e>
   16716: f9 66        	lui	a3, 30
   16718: 79 65        	lui	a0, 30
   1671a: 93 86 06 a2  	addi	a3, a3, -1504
   1671e: 01 46        	mv	a2, zero
   16720: 93 05 20 23  	addi	a1, zero, 562
   16724: 13 05 05 d2  	addi	a0, a0, -736
   16728: ef 30 e0 23  	jal	0x19966 <__assert_func>
   1672c: f9 66        	lui	a3, 30
   1672e: 79 65        	lui	a0, 30
   16730: 93 86 06 a2  	addi	a3, a3, -1504
   16734: 01 46        	mv	a2, zero
   16736: 93 05 00 24  	addi	a1, zero, 576
   1673a: 13 05 05 d2  	addi	a0, a0, -736
   1673e: ef 30 80 22  	jal	0x19966 <__assert_func>

0000000000016742 <__ulp>:
   16742: d3 07 05 e2  	fmv.x.d	a5, fa0
   16746: 37 07 f0 7f  	lui	a4, 524032
   1674a: 81 46        	mv	a3, zero
   1674c: 81 93        	srli	a5, a5, 32
   1674e: f9 8f        	and	a5, a5, a4
   16750: 37 07 c0 fc  	lui	a4, 1035264
   16754: b9 9f        	addw	a5, a5, a4
   16756: 63 59 f0 00  	blez	a5, 0x16768 <__ulp+0x26>
   1675a: 82 16        	slli	a3, a3, 32
   1675c: 81 92        	srli	a3, a3, 32
   1675e: 82 17        	slli	a5, a5, 32
   16760: d5 8f        	or	a5, a5, a3
   16762: 53 85 07 f2  	fmv.d.x	fa0, a5
   16766: 82 80        	ret
   16768: 3b 07 f0 40  	negw	a4, a5
   1676c: 9b 56 47 41  	sraiw	a3, a4, 20
   16770: cd 47        	addi	a5, zero, 19
   16772: 63 ce d7 00  	blt	a5, a3, 0x1678e <__ulp+0x4c>
   16776: b7 07 08 00  	lui	a5, 128
   1677a: bb d7 d7 40  	sraw	a5, a5, a3
   1677e: 81 46        	mv	a3, zero
   16780: 82 16        	slli	a3, a3, 32
   16782: 81 92        	srli	a3, a3, 32
   16784: 82 17        	slli	a5, a5, 32
   16786: d5 8f        	or	a5, a5, a3
   16788: 53 85 07 f2  	fmv.d.x	fa0, a5
   1678c: 82 80        	ret
   1678e: 1b 87 c6 fe  	addiw	a4, a3, -20
   16792: f9 47        	addi	a5, zero, 30
   16794: 85 46        	addi	a3, zero, 1
   16796: 63 c6 e7 00  	blt	a5, a4, 0x167a2 <__ulp+0x60>
   1679a: b7 06 00 80  	lui	a3, 524288
   1679e: bb d6 e6 00  	srlw	a3, a3, a4
   167a2: 81 47        	mv	a5, zero
   167a4: 82 16        	slli	a3, a3, 32
   167a6: 81 92        	srli	a3, a3, 32
   167a8: 82 17        	slli	a5, a5, 32
   167aa: d5 8f        	or	a5, a5, a3
   167ac: 53 85 07 f2  	fmv.d.x	fa0, a5
   167b0: 82 80        	ret

00000000000167b2 <__b2d>:
   167b2: 79 71        	addi	sp, sp, -48
   167b4: 26 ec        	sd	s1, 24(sp)
   167b6: 44 49        	lw	s1, 20(a0)
   167b8: 22 f0        	sd	s0, 32(sp)
   167ba: 13 04 85 01  	addi	s0, a0, 24
   167be: 8a 04        	slli	s1, s1, 2
   167c0: a2 94        	add	s1, s1, s0
   167c2: 4a e8        	sd	s2, 16(sp)
   167c4: 03 a9 c4 ff  	lw	s2, -4(s1)
   167c8: 4e e4        	sd	s3, 8(sp)
   167ca: 52 e0        	sd	s4, 0(sp)
   167cc: 4a 85        	mv	a0, s2
   167ce: ae 89        	mv	s3, a1
   167d0: 06 f4        	sd	ra, 40(sp)
   167d2: ef f0 bf 85  	jal	0x1602c <__hi0bits>
   167d6: 93 06 00 02  	addi	a3, zero, 32
   167da: bb 87 a6 40  	subw	a5, a3, a0
   167de: 23 a0 f9 00  	sw	a5, 0(s3)
   167e2: a9 47        	addi	a5, zero, 10
   167e4: 13 8a c4 ff  	addi	s4, s1, -4
   167e8: 63 d5 a7 08  	bge	a5, a0, 0x16872 <__b2d+0xc0>
   167ec: 9b 07 55 ff  	addiw	a5, a0, -11
   167f0: 63 71 44 05  	bgeu	s0, s4, 0x16832 <__b2d+0x80>
   167f4: 03 a7 84 ff  	lw	a4, -8(s1)
   167f8: a9 cb        	beqz	a5, 0x1684a <__b2d+0x98>
   167fa: 9d 9e        	subw	a3, a3, a5
   167fc: 3b 56 d7 00  	srlw	a2, a4, a3
   16800: 3b 19 f9 00  	sllw	s2, s2, a5
   16804: 33 69 c9 00  	or	s2, s2, a2
   16808: 37 06 f0 3f  	lui	a2, 261888
   1680c: 33 69 c9 00  	or	s2, s2, a2
   16810: bb 15 f7 00  	sllw	a1, a4, a5
   16814: 13 86 84 ff  	addi	a2, s1, -8
   16818: 93 17 09 02  	slli	a5, s2, 32
   1681c: 1b 87 05 00  	sext.w	a4, a1
   16820: 63 7a c4 02  	bgeu	s0, a2, 0x16854 <__b2d+0xa2>
   16824: 03 a7 44 ff  	lw	a4, -12(s1)
   16828: 3b 57 d7 00  	srlw	a4, a4, a3
   1682c: 4d 8f        	or	a4, a4, a1
   1682e: 01 27        	sext.w	a4, a4
   16830: 15 a0        	j	0x16854 <__b2d+0xa2>
   16832: 2d 47        	addi	a4, zero, 11
   16834: 63 0a e5 00  	beq	a0, a4, 0x16848 <__b2d+0x96>
   16838: 37 07 f0 3f  	lui	a4, 261888
   1683c: bb 17 f9 00  	sllw	a5, s2, a5
   16840: d9 8f        	or	a5, a5, a4
   16842: 82 17        	slli	a5, a5, 32
   16844: 01 47        	mv	a4, zero
   16846: 39 a0        	j	0x16854 <__b2d+0xa2>
   16848: 01 47        	mv	a4, zero
   1684a: b7 07 f0 3f  	lui	a5, 261888
   1684e: b3 67 f9 00  	or	a5, s2, a5
   16852: 82 17        	slli	a5, a5, 32
   16854: a2 70        	ld	ra, 40(sp)
   16856: 02 74        	ld	s0, 32(sp)
   16858: 81 93        	srli	a5, a5, 32
   1685a: 02 17        	slli	a4, a4, 32
   1685c: 01 93        	srli	a4, a4, 32
   1685e: 82 17        	slli	a5, a5, 32
   16860: d9 8f        	or	a5, a5, a4
   16862: e2 64        	ld	s1, 24(sp)
   16864: 42 69        	ld	s2, 16(sp)
   16866: a2 69        	ld	s3, 8(sp)
   16868: 02 6a        	ld	s4, 0(sp)
   1686a: 53 85 07 f2  	fmv.d.x	fa0, a5
   1686e: 45 61        	addi	sp, sp, 48
   16870: 82 80        	ret
   16872: ad 46        	addi	a3, zero, 11
   16874: 89 9e        	subw	a3, a3, a0
   16876: 37 07 f0 3f  	lui	a4, 261888
   1687a: bb 57 d9 00  	srlw	a5, s2, a3
   1687e: d9 8f        	or	a5, a5, a4
   16880: 82 17        	slli	a5, a5, 32
   16882: 01 47        	mv	a4, zero
   16884: 63 76 44 01  	bgeu	s0, s4, 0x16890 <__b2d+0xde>
   16888: 03 a7 84 ff  	lw	a4, -8(s1)
   1688c: 3b 57 d7 00  	srlw	a4, a4, a3
   16890: 55 25        	addiw	a0, a0, 21
   16892: 3b 15 a9 00  	sllw	a0, s2, a0
   16896: 49 8f        	or	a4, a4, a0
   16898: 01 27        	sext.w	a4, a4
   1689a: 6d bf        	j	0x16854 <__b2d+0xa2>

000000000001689c <__d2b>:
   1689c: 39 71        	addi	sp, sp, -64
   1689e: 52 e8        	sd	s4, 16(sp)
   168a0: 2e 8a        	mv	s4, a1
   168a2: 85 45        	addi	a1, zero, 1
   168a4: 22 f8        	sd	s0, 48(sp)
   168a6: 4a f0        	sd	s2, 32(sp)
   168a8: 06 fc        	sd	ra, 56(sp)
   168aa: 26 f4        	sd	s1, 40(sp)
   168ac: 4e ec        	sd	s3, 24(sp)
   168ae: 53 04 05 e2  	fmv.x.d	s0, fa0
   168b2: 32 89        	mv	s2, a2
   168b4: ef f0 cf d3  	jal	0x15df0 <_Balloc>
   168b8: 55 c9        	beqz	a0, 0x1696c <__d2b+0xd0>
   168ba: 93 57 04 42  	srai	a5, s0, 32
   168be: 9b d9 47 01  	srliw	s3, a5, 20
   168c2: 93 f9 f9 7f  	andi	s3, s3, 2047
   168c6: b2 17        	slli	a5, a5, 44
   168c8: aa 84        	mv	s1, a0
   168ca: b1 93        	srli	a5, a5, 44
   168cc: 63 85 09 00  	beqz	s3, 0x168d6 <__d2b+0x3a>
   168d0: 37 07 10 00  	lui	a4, 256
   168d4: d9 8f        	or	a5, a5, a4
   168d6: 3e c6        	sw	a5, 12(sp)
   168d8: 01 24        	sext.w	s0, s0
   168da: 29 e0        	bnez	s0, 0x1691c <__d2b+0x80>
   168dc: 68 00        	addi	a0, sp, 12
   168de: ef f0 cf fa  	jal	0x1608a <__lo0bits>
   168e2: b2 47        	lw	a5, 12(sp)
   168e4: 05 44        	addi	s0, zero, 1
   168e6: c0 c8        	sw	s0, 20(s1)
   168e8: 9c cc        	sw	a5, 24(s1)
   168ea: 1b 05 05 02  	addiw	a0, a0, 32
   168ee: 63 80 09 06  	beqz	s3, 0x1694e <__d2b+0xb2>
   168f2: 9b 89 d9 bc  	addiw	s3, s3, -1075
   168f6: bb 89 a9 00  	addw	s3, s3, a0
   168fa: 13 04 50 03  	addi	s0, zero, 53
   168fe: 23 20 3a 01  	sw	s3, 0(s4)
   16902: 3b 05 a4 40  	subw	a0, s0, a0
   16906: e2 70        	ld	ra, 56(sp)
   16908: 42 74        	ld	s0, 48(sp)
   1690a: 23 20 a9 00  	sw	a0, 0(s2)
   1690e: e2 69        	ld	s3, 24(sp)
   16910: 02 79        	ld	s2, 32(sp)
   16912: 42 6a        	ld	s4, 16(sp)
   16914: 26 85        	mv	a0, s1
   16916: a2 74        	ld	s1, 40(sp)
   16918: 21 61        	addi	sp, sp, 64
   1691a: 82 80        	ret
   1691c: 28 00        	addi	a0, sp, 8
   1691e: 22 c4        	sw	s0, 8(sp)
   16920: ef f0 af f6  	jal	0x1608a <__lo0bits>
   16924: b2 47        	lw	a5, 12(sp)
   16926: 22 47        	lw	a4, 8(sp)
   16928: 19 c9        	beqz	a0, 0x1693e <__d2b+0xa2>
   1692a: 93 06 00 02  	addi	a3, zero, 32
   1692e: 89 9e        	subw	a3, a3, a0
   16930: bb 96 d7 00  	sllw	a3, a5, a3
   16934: 55 8f        	or	a4, a4, a3
   16936: bb d7 a7 00  	srlw	a5, a5, a0
   1693a: 01 27        	sext.w	a4, a4
   1693c: 3e c6        	sw	a5, 12(sp)
   1693e: 33 34 f0 00  	snez	s0, a5
   16942: 05 04        	addi	s0, s0, 1
   16944: 98 cc        	sw	a4, 24(s1)
   16946: dc cc        	sw	a5, 28(s1)
   16948: c0 c8        	sw	s0, 20(s1)
   1694a: e3 94 09 fa  	bnez	s3, 0x168f2 <__d2b+0x56>
   1694e: 1b 05 e5 bc  	addiw	a0, a0, -1074
   16952: 93 17 24 00  	slli	a5, s0, 2
   16956: 23 20 aa 00  	sw	a0, 0(s4)
   1695a: a6 97        	add	a5, a5, s1
   1695c: c8 4b        	lw	a0, 20(a5)
   1695e: 1b 14 54 00  	slliw	s0, s0, 5
   16962: ef f0 af ec  	jal	0x1602c <__hi0bits>
   16966: 3b 05 a4 40  	subw	a0, s0, a0
   1696a: 71 bf        	j	0x16906 <__d2b+0x6a>
   1696c: f9 66        	lui	a3, 30
   1696e: 79 65        	lui	a0, 30
   16970: 93 86 06 a2  	addi	a3, a3, -1504
   16974: 01 46        	mv	a2, zero
   16976: 93 05 a0 30  	addi	a1, zero, 778
   1697a: 13 05 05 d2  	addi	a0, a0, -736
   1697e: ef 20 90 7e  	jal	0x19966 <__assert_func>

0000000000016982 <__ratio>:
   16982: 79 71        	addi	sp, sp, -48
   16984: 26 ec        	sd	s1, 24(sp)
   16986: ae 84        	mv	s1, a1
   16988: 2c 00        	addi	a1, sp, 8
   1698a: 06 f4        	sd	ra, 40(sp)
   1698c: 22 f0        	sd	s0, 32(sp)
   1698e: 4a e8        	sd	s2, 16(sp)
   16990: 2a 89        	mv	s2, a0
   16992: ef f0 1f e2  	jal	0x167b2 <__b2d>
   16996: 6c 00        	addi	a1, sp, 12
   16998: 26 85        	mv	a0, s1
   1699a: 53 04 05 e2  	fmv.x.d	s0, fa0
   1699e: ef f0 5f e1  	jal	0x167b2 <__b2d>
   169a2: d8 48        	lw	a4, 20(s1)
   169a4: 83 27 49 01  	lw	a5, 20(s2)
   169a8: a2 46        	lw	a3, 8(sp)
   169aa: 32 46        	lw	a2, 12(sp)
   169ac: 99 9f        	subw	a5, a5, a4
   169ae: 9b 97 57 00  	slliw	a5, a5, 5
   169b2: 91 9e        	subw	a3, a3, a2
   169b4: bb 85 d7 00  	addw	a1, a5, a3
   169b8: 53 07 05 e2  	fmv.x.d	a4, fa0
   169bc: 63 59 b0 02  	blez	a1, 0x169ee <__ratio+0x6c>
   169c0: 13 56 04 02  	srli	a2, s0, 32
   169c4: 9b 97 45 01  	slliw	a5, a1, 20
   169c8: 93 16 04 02  	slli	a3, s0, 32
   169cc: b1 9f        	addw	a5, a5, a2
   169ce: 82 17        	slli	a5, a5, 32
   169d0: 81 92        	srli	a3, a3, 32
   169d2: 33 e4 f6 00  	or	s0, a3, a5
   169d6: d3 07 04 f2  	fmv.d.x	fa5, s0
   169da: a2 70        	ld	ra, 40(sp)
   169dc: 02 74        	ld	s0, 32(sp)
   169de: 53 07 07 f2  	fmv.d.x	fa4, a4
   169e2: e2 64        	ld	s1, 24(sp)
   169e4: 42 69        	ld	s2, 16(sp)
   169e6: 53 f5 e7 1a  	fdiv.d	fa0, fa5, fa4
   169ea: 45 61        	addi	sp, sp, 48
   169ec: 82 80        	ret
   169ee: 3a 86        	mv	a2, a4
   169f0: 9b 97 45 01  	slliw	a5, a1, 20
   169f4: 01 93        	srli	a4, a4, 32
   169f6: 1d 9f        	subw	a4, a4, a5
   169f8: 02 16        	slli	a2, a2, 32
   169fa: 02 17        	slli	a4, a4, 32
   169fc: 01 92        	srli	a2, a2, 32
   169fe: 51 8f        	or	a4, a4, a2
   16a00: d9 bf        	j	0x169d6 <__ratio+0x54>

0000000000016a02 <_mprec_log10>:
   16a02: dd 47        	addi	a5, zero, 23
   16a04: 63 db a7 00  	bge	a5, a0, 0x16a1a <_mprec_log10+0x18>
   16a08: 07 b5 01 81  	fld	fa0, -2032(gp)
   16a0c: 87 b7 81 81  	fld	fa5, -2024(gp)
   16a10: 7d 35        	addiw	a0, a0, -1
   16a12: 53 75 f5 12  	fmul.d	fa0, fa0, fa5
   16a16: 6d fd        	bnez	a0, 0x16a10 <_mprec_log10+0xe>
   16a18: 82 80        	ret
   16a1a: f9 67        	lui	a5, 30
   16a1c: 0e 05        	slli	a0, a0, 3
   16a1e: 93 87 07 de  	addi	a5, a5, -544
   16a22: aa 97        	add	a5, a5, a0
   16a24: 88 23        	fld	fa0, 0(a5)
   16a26: 82 80        	ret

0000000000016a28 <__copybits>:
   16a28: 54 4a        	lw	a3, 20(a2)
   16a2a: 9b 87 f5 ff  	addiw	a5, a1, -1
   16a2e: 9b d7 57 40  	sraiw	a5, a5, 5
   16a32: 85 27        	addiw	a5, a5, 1
   16a34: 13 07 86 01  	addi	a4, a2, 24
   16a38: 93 95 26 00  	slli	a1, a3, 2
   16a3c: 8a 07        	slli	a5, a5, 2
   16a3e: ba 95        	add	a1, a1, a4
   16a40: aa 97        	add	a5, a5, a0
   16a42: 63 71 b7 02  	bgeu	a4, a1, 0x16a64 <__copybits+0x3c>
   16a46: aa 86        	mv	a3, a0
   16a48: 03 28 07 00  	lw	a6, 0(a4)
   16a4c: 11 07        	addi	a4, a4, 4
   16a4e: 91 06        	addi	a3, a3, 4
   16a50: 23 ae 06 ff  	sw	a6, -4(a3)
   16a54: e3 6a b7 fe  	bltu	a4, a1, 0x16a48 <__copybits+0x20>
   16a58: 33 87 c5 40  	sub	a4, a1, a2
   16a5c: 1d 17        	addi	a4, a4, -25
   16a5e: 71 9b        	andi	a4, a4, -4
   16a60: 11 07        	addi	a4, a4, 4
   16a62: 3a 95        	add	a0, a0, a4
   16a64: 63 73 f5 02  	bgeu	a0, a5, 0x16a8a <__copybits+0x62>
   16a68: fd 17        	addi	a5, a5, -1
   16a6a: 89 8f        	sub	a5, a5, a0
   16a6c: 89 83        	srli	a5, a5, 2
   16a6e: 85 07        	addi	a5, a5, 1
   16a70: f3 26 20 c2  	csrr	a3, vlenb
   16a74: 57 77 00 05  	<unknown>
   16a78: 57 3c 00 5e  	<unknown>
   16a7c: 57 f7 07 05  	<unknown>
   16a80: 27 6c 05 02  	<unknown>
   16a84: 99 8f        	sub	a5, a5, a4
   16a86: 36 95        	add	a0, a0, a3
   16a88: f5 fb        	bnez	a5, 0x16a7c <__copybits+0x54>
   16a8a: 82 80        	ret

0000000000016a8c <__any_on>:
   16a8c: 58 49        	lw	a4, 20(a0)
   16a8e: 13 d6 55 40  	srai	a2, a1, 5
   16a92: 93 06 85 01  	addi	a3, a0, 24
   16a96: 63 5d c7 00  	bge	a4, a2, 0x16ab0 <__any_on+0x24>
   16a9a: 0a 07        	slli	a4, a4, 2
   16a9c: b3 87 e6 00  	add	a5, a3, a4
   16aa0: 63 f8 f6 02  	bgeu	a3, a5, 0x16ad0 <__any_on+0x44>
   16aa4: 03 a7 c7 ff  	lw	a4, -4(a5)
   16aa8: f1 17        	addi	a5, a5, -4
   16aaa: 7d db        	beqz	a4, 0x16aa0 <__any_on+0x14>
   16aac: 05 45        	addi	a0, zero, 1
   16aae: 82 80        	ret
   16ab0: 93 17 26 00  	slli	a5, a2, 2
   16ab4: b6 97        	add	a5, a5, a3
   16ab6: e3 55 e6 fe  	bge	a2, a4, 0x16aa0 <__any_on+0x14>
   16aba: fd 89        	andi	a1, a1, 31
   16abc: f5 d1        	beqz	a1, 0x16aa0 <__any_on+0x14>
   16abe: 90 43        	lw	a2, 0(a5)
   16ac0: 05 45        	addi	a0, zero, 1
   16ac2: 3b 57 b6 00  	srlw	a4, a2, a1
   16ac6: 3b 17 b7 00  	sllw	a4, a4, a1
   16aca: e3 0b c7 fc  	beq	a4, a2, 0x16aa0 <__any_on+0x14>
   16ace: 82 80        	ret
   16ad0: 01 45        	mv	a0, zero
   16ad2: 82 80        	ret

0000000000016ad4 <_sbrk_r>:
   16ad4: 01 11        	addi	sp, sp, -32
   16ad6: 22 e8        	sd	s0, 16(sp)
   16ad8: 26 e4        	sd	s1, 8(sp)
   16ada: 2a 84        	mv	s0, a0
   16adc: 2e 85        	mv	a0, a1
   16ade: 06 ec        	sd	ra, 24(sp)
   16ae0: 23 ac 01 88  	sw	zero, -1896(gp)
   16ae4: ef 40 f0 7a  	jal	0x1ba92 <_sbrk>
   16ae8: fd 57        	addi	a5, zero, -1
   16aea: 63 07 f5 00  	beq	a0, a5, 0x16af8 <_sbrk_r+0x24>
   16aee: e2 60        	ld	ra, 24(sp)
   16af0: 42 64        	ld	s0, 16(sp)
   16af2: a2 64        	ld	s1, 8(sp)
   16af4: 05 61        	addi	sp, sp, 32
   16af6: 82 80        	ret
   16af8: 83 a7 81 89  	lw	a5, -1896(gp)
   16afc: ed db        	beqz	a5, 0x16aee <_sbrk_r+0x1a>
   16afe: e2 60        	ld	ra, 24(sp)
   16b00: 1c c0        	sw	a5, 0(s0)
   16b02: 42 64        	ld	s0, 16(sp)
   16b04: a2 64        	ld	s1, 8(sp)
   16b06: 05 61        	addi	sp, sp, 32
   16b08: 82 80        	ret

0000000000016b0a <frexp>:
   16b0a: d3 07 05 e2  	fmv.x.d	a5, fa0
   16b0e: b7 05 00 80  	lui	a1, 524288
   16b12: 93 c5 f5 ff  	not	a1, a1
   16b16: 93 d6 07 42  	srai	a3, a5, 32
   16b1a: 33 f7 b6 00  	and	a4, a3, a1
   16b1e: 23 20 05 00  	sw	zero, 0(a0)
   16b22: 37 08 f0 7f  	lui	a6, 524032
   16b26: 63 7f 07 03  	bgeu	a4, a6, 0x16b64 <frexp+0x5a>
   16b2a: 33 e6 e7 00  	or	a2, a5, a4
   16b2e: 01 26        	sext.w	a2, a2
   16b30: 15 ca        	beqz	a2, 0x16b64 <frexp+0x5a>
   16b32: 37 06 f0 7f  	lui	a2, 524032
   16b36: 75 8e        	and	a2, a2, a3
   16b38: 0d ca        	beqz	a2, 0x16b6a <frexp+0x60>
   16b3a: 81 26        	sext.w	a3, a3
   16b3c: 01 27        	sext.w	a4, a4
   16b3e: 01 48        	mv	a6, zero
   16b40: 37 06 10 80  	lui	a2, 524544
   16b44: 7d 16        	addi	a2, a2, -1
   16b46: f1 8e        	and	a3, a3, a2
   16b48: 1b 57 47 41  	sraiw	a4, a4, 20
   16b4c: 37 06 e0 3f  	lui	a2, 261632
   16b50: 1b 07 27 c0  	addiw	a4, a4, -1022
   16b54: d1 8e        	or	a3, a3, a2
   16b56: 82 17        	slli	a5, a5, 32
   16b58: 3b 07 07 01  	addw	a4, a4, a6
   16b5c: 82 16        	slli	a3, a3, 32
   16b5e: 81 93        	srli	a5, a5, 32
   16b60: 18 c1        	sw	a4, 0(a0)
   16b62: d5 8f        	or	a5, a5, a3
   16b64: 53 85 07 f2  	fmv.d.x	fa0, a5
   16b68: 82 80        	ret
   16b6a: 87 b7 01 82  	fld	fa5, -2016(gp)
   16b6e: 13 08 a0 fc  	addi	a6, zero, -54
   16b72: d3 77 f5 12  	fmul.d	fa5, fa0, fa5
   16b76: d3 87 07 e2  	fmv.x.d	a5, fa5
   16b7a: 13 d7 07 42  	srai	a4, a5, 32
   16b7e: ba 86        	mv	a3, a4
   16b80: 6d 8f        	and	a4, a4, a1
   16b82: 7d bf        	j	0x16b40 <frexp+0x36>

0000000000016b84 <_sprintf_r>:
   16b84: 11 71        	addi	sp, sp, -256
   16b86: 93 0e 81 0d  	addi	t4, sp, 216
   16b8a: be f5        	sd	a5, 232(sp)
   16b8c: 37 03 00 80  	lui	t1, 524288
   16b90: c1 77        	lui	a5, 1048560
   16b92: 2e 8e        	mv	t3, a1
   16b94: 13 43 f3 ff  	not	t1, t1
   16b98: b6 ed        	sd	a3, 216(sp)
   16b9a: 93 87 87 20  	addi	a5, a5, 520
   16b9e: 0c 08        	addi	a1, sp, 16
   16ba0: f6 86        	mv	a3, t4
   16ba2: 86 e5        	sd	ra, 200(sp)
   16ba4: 3e d0        	sw	a5, 32(sp)
   16ba6: ba f1        	sd	a4, 224(sp)
   16ba8: c2 f9        	sd	a6, 240(sp)
   16baa: c6 fd        	sd	a7, 248(sp)
   16bac: 72 e8        	sd	t3, 16(sp)
   16bae: 72 f4        	sd	t3, 40(sp)
   16bb0: 1a d8        	sw	t1, 48(sp)
   16bb2: 1a ce        	sw	t1, 28(sp)
   16bb4: 76 e4        	sd	t4, 8(sp)
   16bb6: ef 00 80 3e  	jal	0x16f9e <_svfprintf_r>
   16bba: c2 67        	ld	a5, 16(sp)
   16bbc: 23 80 07 00  	sb	zero, 0(a5)
   16bc0: ae 60        	ld	ra, 200(sp)
   16bc2: 11 61        	addi	sp, sp, 256
   16bc4: 82 80        	ret

0000000000016bc6 <sprintf>:
   16bc6: 2a 8e        	mv	t3, a0
   16bc8: 11 71        	addi	sp, sp, -256
   16bca: 03 b5 81 85  	ld	a0, -1960(gp)
   16bce: 93 0e 01 0d  	addi	t4, sp, 208
   16bd2: be f5        	sd	a5, 232(sp)
   16bd4: 37 03 00 80  	lui	t1, 524288
   16bd8: c1 77        	lui	a5, 1048560
   16bda: 13 43 f3 ff  	not	t1, t1
   16bde: b2 e9        	sd	a2, 208(sp)
   16be0: b6 ed        	sd	a3, 216(sp)
   16be2: 93 87 87 20  	addi	a5, a5, 520
   16be6: 2e 86        	mv	a2, a1
   16be8: f6 86        	mv	a3, t4
   16bea: 0c 08        	addi	a1, sp, 16
   16bec: 86 e5        	sd	ra, 200(sp)
   16bee: 3e d0        	sw	a5, 32(sp)
   16bf0: ba f1        	sd	a4, 224(sp)
   16bf2: c2 f9        	sd	a6, 240(sp)
   16bf4: c6 fd        	sd	a7, 248(sp)
   16bf6: 72 e8        	sd	t3, 16(sp)
   16bf8: 72 f4        	sd	t3, 40(sp)
   16bfa: 1a d8        	sw	t1, 48(sp)
   16bfc: 1a ce        	sw	t1, 28(sp)
   16bfe: 76 e4        	sd	t4, 8(sp)
   16c00: ef 00 e0 39  	jal	0x16f9e <_svfprintf_r>
   16c04: c2 67        	ld	a5, 16(sp)
   16c06: 23 80 07 00  	sb	zero, 0(a5)
   16c0a: ae 60        	ld	ra, 200(sp)
   16c0c: 11 61        	addi	sp, sp, 256
   16c0e: 82 80        	ret

0000000000016c10 <__sread>:
   16c10: 41 11        	addi	sp, sp, -16
   16c12: 22 e0        	sd	s0, 0(sp)
   16c14: 2e 84        	mv	s0, a1
   16c16: 83 95 25 01  	lh	a1, 18(a1)
   16c1a: 06 e4        	sd	ra, 8(sp)
   16c1c: ef 30 60 55  	jal	0x1a172 <_read_r>
   16c20: 63 49 05 00  	bltz	a0, 0x16c32 <__sread+0x22>
   16c24: 5c 68        	ld	a5, 144(s0)
   16c26: a2 60        	ld	ra, 8(sp)
   16c28: aa 97        	add	a5, a5, a0
   16c2a: 5c e8        	sd	a5, 144(s0)
   16c2c: 02 64        	ld	s0, 0(sp)
   16c2e: 41 01        	addi	sp, sp, 16
   16c30: 82 80        	ret
   16c32: 83 57 04 01  	lhu	a5, 16(s0)
   16c36: 7d 77        	lui	a4, 1048575
   16c38: 7d 17        	addi	a4, a4, -1
   16c3a: f9 8f        	and	a5, a5, a4
   16c3c: a2 60        	ld	ra, 8(sp)
   16c3e: 23 18 f4 00  	sh	a5, 16(s0)
   16c42: 02 64        	ld	s0, 0(sp)
   16c44: 41 01        	addi	sp, sp, 16
   16c46: 82 80        	ret

0000000000016c48 <__seofread>:
   16c48: 01 45        	mv	a0, zero
   16c4a: 82 80        	ret

0000000000016c4c <__swrite>:
   16c4c: 83 97 05 01  	lh	a5, 16(a1)
   16c50: 79 71        	addi	sp, sp, -48
   16c52: 22 f0        	sd	s0, 32(sp)
   16c54: 26 ec        	sd	s1, 24(sp)
   16c56: 4a e8        	sd	s2, 16(sp)
   16c58: 4e e4        	sd	s3, 8(sp)
   16c5a: 06 f4        	sd	ra, 40(sp)
   16c5c: 13 f7 07 10  	andi	a4, a5, 256
   16c60: 2e 84        	mv	s0, a1
   16c62: aa 84        	mv	s1, a0
   16c64: 32 89        	mv	s2, a2
   16c66: b6 89        	mv	s3, a3
   16c68: 1d e3        	bnez	a4, 0x16c8e <__swrite+0x42>
   16c6a: 7d 77        	lui	a4, 1048575
   16c6c: 7d 17        	addi	a4, a4, -1
   16c6e: f9 8f        	and	a5, a5, a4
   16c70: 83 15 24 01  	lh	a1, 18(s0)
   16c74: 23 18 f4 00  	sh	a5, 16(s0)
   16c78: 02 74        	ld	s0, 32(sp)
   16c7a: a2 70        	ld	ra, 40(sp)
   16c7c: ce 86        	mv	a3, s3
   16c7e: 4a 86        	mv	a2, s2
   16c80: a2 69        	ld	s3, 8(sp)
   16c82: 42 69        	ld	s2, 16(sp)
   16c84: 26 85        	mv	a0, s1
   16c86: e2 64        	ld	s1, 24(sp)
   16c88: 45 61        	addi	sp, sp, 48
   16c8a: 6f 20 10 4a  	j	0x1992a <_write_r>
   16c8e: 83 95 25 01  	lh	a1, 18(a1)
   16c92: 89 46        	addi	a3, zero, 2
   16c94: 01 46        	mv	a2, zero
   16c96: ef 30 80 3c  	jal	0x1a05e <_lseek_r>
   16c9a: 83 17 04 01  	lh	a5, 16(s0)
   16c9e: f1 b7        	j	0x16c6a <__swrite+0x1e>

0000000000016ca0 <__sseek>:
   16ca0: 41 11        	addi	sp, sp, -16
   16ca2: 22 e0        	sd	s0, 0(sp)
   16ca4: 2e 84        	mv	s0, a1
   16ca6: 83 95 25 01  	lh	a1, 18(a1)
   16caa: 06 e4        	sd	ra, 8(sp)
   16cac: ef 30 20 3b  	jal	0x1a05e <_lseek_r>
   16cb0: fd 57        	addi	a5, zero, -1
   16cb2: 63 01 f5 02  	beq	a0, a5, 0x16cd4 <__sseek+0x34>
   16cb6: 83 57 04 01  	lhu	a5, 16(s0)
   16cba: 05 67        	lui	a4, 1
   16cbc: a2 60        	ld	ra, 8(sp)
   16cbe: d9 8f        	or	a5, a5, a4
   16cc0: 9b 97 07 01  	slliw	a5, a5, 16
   16cc4: 9b d7 07 41  	sraiw	a5, a5, 16
   16cc8: 48 e8        	sd	a0, 144(s0)
   16cca: 23 18 f4 00  	sh	a5, 16(s0)
   16cce: 02 64        	ld	s0, 0(sp)
   16cd0: 41 01        	addi	sp, sp, 16
   16cd2: 82 80        	ret
   16cd4: 83 57 04 01  	lhu	a5, 16(s0)
   16cd8: 7d 77        	lui	a4, 1048575
   16cda: 7d 17        	addi	a4, a4, -1
   16cdc: f9 8f        	and	a5, a5, a4
   16cde: 9b 97 07 01  	slliw	a5, a5, 16
   16ce2: 9b d7 07 41  	sraiw	a5, a5, 16
   16ce6: a2 60        	ld	ra, 8(sp)
   16ce8: 23 18 f4 00  	sh	a5, 16(s0)
   16cec: 02 64        	ld	s0, 0(sp)
   16cee: 41 01        	addi	sp, sp, 16
   16cf0: 82 80        	ret

0000000000016cf2 <__sclose>:
   16cf2: 83 95 25 01  	lh	a1, 18(a1)
   16cf6: 6f 20 10 57  	j	0x19a66 <_close_r>

0000000000016cfa <strcmp>:
   16cfa: 33 67 b5 00  	or	a4, a0, a1
   16cfe: fd 53        	addi	t2, zero, -1
   16d00: 1d 8b        	andi	a4, a4, 7
   16d02: 4d eb        	bnez	a4, 0x16db4 <strcmp+0xba>
   16d04: 97 a7 00 00  	auipc	a5, 10
   16d08: 83 b7 47 a7  	ld	a5, -1420(a5)
   16d0c: 10 61        	ld	a2, 0(a0)
   16d0e: 94 61        	ld	a3, 0(a1)
   16d10: b3 72 f6 00  	and	t0, a2, a5
   16d14: 33 63 f6 00  	or	t1, a2, a5
   16d18: be 92        	add	t0, t0, a5
   16d1a: b3 e2 62 00  	or	t0, t0, t1
   16d1e: 63 99 72 0a  	bne	t0, t2, 0x16dd0 <strcmp+0xd6>
   16d22: 63 1e d6 02  	bne	a2, a3, 0x16d5e <strcmp+0x64>
   16d26: 10 65        	ld	a2, 8(a0)
   16d28: 94 65        	ld	a3, 8(a1)
   16d2a: b3 72 f6 00  	and	t0, a2, a5
   16d2e: 33 63 f6 00  	or	t1, a2, a5
   16d32: be 92        	add	t0, t0, a5
   16d34: b3 e2 62 00  	or	t0, t0, t1
   16d38: 63 9a 72 08  	bne	t0, t2, 0x16dcc <strcmp+0xd2>
   16d3c: 63 11 d6 02  	bne	a2, a3, 0x16d5e <strcmp+0x64>
   16d40: 10 69        	ld	a2, 16(a0)
   16d42: 94 69        	ld	a3, 16(a1)
   16d44: b3 72 f6 00  	and	t0, a2, a5
   16d48: 33 63 f6 00  	or	t1, a2, a5
   16d4c: be 92        	add	t0, t0, a5
   16d4e: b3 e2 62 00  	or	t0, t0, t1
   16d52: 63 93 72 08  	bne	t0, t2, 0x16dd8 <strcmp+0xde>
   16d56: 61 05        	addi	a0, a0, 24
   16d58: e1 05        	addi	a1, a1, 24
   16d5a: e3 09 d6 fa  	beq	a2, a3, 0x16d0c <strcmp+0x12>
   16d5e: 13 17 06 03  	slli	a4, a2, 48
   16d62: 93 97 06 03  	slli	a5, a3, 48
   16d66: 63 18 f7 02  	bne	a4, a5, 0x16d96 <strcmp+0x9c>
   16d6a: 13 17 06 02  	slli	a4, a2, 32
   16d6e: 93 97 06 02  	slli	a5, a3, 32
   16d72: 63 12 f7 02  	bne	a4, a5, 0x16d96 <strcmp+0x9c>
   16d76: 13 17 06 01  	slli	a4, a2, 16
   16d7a: 93 97 06 01  	slli	a5, a3, 16
   16d7e: 63 1c f7 00  	bne	a4, a5, 0x16d96 <strcmp+0x9c>
   16d82: 13 57 06 03  	srli	a4, a2, 48
   16d86: 93 d7 06 03  	srli	a5, a3, 48
   16d8a: 33 05 f7 40  	sub	a0, a4, a5
   16d8e: 93 75 f5 0f  	andi	a1, a0, 255
   16d92: 91 e9        	bnez	a1, 0x16da6 <strcmp+0xac>
   16d94: 82 80        	ret
   16d96: 41 93        	srli	a4, a4, 48
   16d98: c1 93        	srli	a5, a5, 48
   16d9a: 33 05 f7 40  	sub	a0, a4, a5
   16d9e: 93 75 f5 0f  	andi	a1, a0, 255
   16da2: 91 e1        	bnez	a1, 0x16da6 <strcmp+0xac>
   16da4: 82 80        	ret
   16da6: 13 77 f7 0f  	andi	a4, a4, 255
   16daa: 93 f7 f7 0f  	andi	a5, a5, 255
   16dae: 33 05 f7 40  	sub	a0, a4, a5
   16db2: 82 80        	ret
   16db4: 03 46 05 00  	lbu	a2, 0(a0)
   16db8: 83 c6 05 00  	lbu	a3, 0(a1)
   16dbc: 05 05        	addi	a0, a0, 1
   16dbe: 85 05        	addi	a1, a1, 1
   16dc0: 63 13 d6 00  	bne	a2, a3, 0x16dc6 <strcmp+0xcc>
   16dc4: 65 fa        	bnez	a2, 0x16db4 <strcmp+0xba>
   16dc6: 33 05 d6 40  	sub	a0, a2, a3
   16dca: 82 80        	ret
   16dcc: 21 05        	addi	a0, a0, 8
   16dce: a1 05        	addi	a1, a1, 8
   16dd0: e3 12 d6 fe  	bne	a2, a3, 0x16db4 <strcmp+0xba>
   16dd4: 01 45        	mv	a0, zero
   16dd6: 82 80        	ret
   16dd8: 41 05        	addi	a0, a0, 16
   16dda: c1 05        	addi	a1, a1, 16
   16ddc: e3 1c d6 fc  	bne	a2, a3, 0x16db4 <strcmp+0xba>
   16de0: 01 45        	mv	a0, zero
   16de2: 82 80        	ret

0000000000016de4 <strcpy>:
   16de4: b3 67 b5 00  	or	a5, a0, a1
   16de8: 9d 8b        	andi	a5, a5, 7
   16dea: ad ef        	bnez	a5, 0x16e64 <strcpy+0x80>
   16dec: 83 b6 81 82  	ld	a3, -2008(gp)
   16df0: 98 61        	ld	a4, 0(a1)
   16df2: 7d 56        	addi	a2, zero, -1
   16df4: b3 77 d7 00  	and	a5, a4, a3
   16df8: b6 97        	add	a5, a5, a3
   16dfa: d9 8f        	or	a5, a5, a4
   16dfc: d5 8f        	or	a5, a5, a3
   16dfe: 63 9c c7 06  	bne	a5, a2, 0x16e76 <strcpy+0x92>
   16e02: 2a 86        	mv	a2, a0
   16e04: 7d 58        	addi	a6, zero, -1
   16e06: 18 e2        	sd	a4, 0(a2)
   16e08: 98 65        	ld	a4, 8(a1)
   16e0a: a1 05        	addi	a1, a1, 8
   16e0c: 21 06        	addi	a2, a2, 8
   16e0e: b3 77 d7 00  	and	a5, a4, a3
   16e12: b6 97        	add	a5, a5, a3
   16e14: d9 8f        	or	a5, a5, a4
   16e16: d5 8f        	or	a5, a5, a3
   16e18: e3 87 07 ff  	beq	a5, a6, 0x16e06 <strcpy+0x22>
   16e1c: 83 c7 05 00  	lbu	a5, 0(a1)
   16e20: 03 c7 15 00  	lbu	a4, 1(a1)
   16e24: 83 c6 25 00  	lbu	a3, 2(a1)
   16e28: 23 00 f6 00  	sb	a5, 0(a2)
   16e2c: 9d cb        	beqz	a5, 0x16e62 <strcpy+0x7e>
   16e2e: a3 00 e6 00  	sb	a4, 1(a2)
   16e32: 05 cb        	beqz	a4, 0x16e62 <strcpy+0x7e>
   16e34: 83 c7 35 00  	lbu	a5, 3(a1)
   16e38: 23 01 d6 00  	sb	a3, 2(a2)
   16e3c: 9d c2        	beqz	a3, 0x16e62 <strcpy+0x7e>
   16e3e: 03 c7 45 00  	lbu	a4, 4(a1)
   16e42: a3 01 f6 00  	sb	a5, 3(a2)
   16e46: 91 cf        	beqz	a5, 0x16e62 <strcpy+0x7e>
   16e48: 83 c7 55 00  	lbu	a5, 5(a1)
   16e4c: 23 02 e6 00  	sb	a4, 4(a2)
   16e50: 09 cb        	beqz	a4, 0x16e62 <strcpy+0x7e>
   16e52: 03 c7 65 00  	lbu	a4, 6(a1)
   16e56: a3 02 f6 00  	sb	a5, 5(a2)
   16e5a: 81 c7        	beqz	a5, 0x16e62 <strcpy+0x7e>
   16e5c: 23 03 e6 00  	sb	a4, 6(a2)
   16e60: 09 ef        	bnez	a4, 0x16e7a <strcpy+0x96>
   16e62: 82 80        	ret
   16e64: aa 87        	mv	a5, a0
   16e66: 03 c7 05 00  	lbu	a4, 0(a1)
   16e6a: 85 07        	addi	a5, a5, 1
   16e6c: 85 05        	addi	a1, a1, 1
   16e6e: a3 8f e7 fe  	sb	a4, -1(a5)
   16e72: 75 fb        	bnez	a4, 0x16e66 <strcpy+0x82>
   16e74: 82 80        	ret
   16e76: 2a 86        	mv	a2, a0
   16e78: 55 b7        	j	0x16e1c <strcpy+0x38>
   16e7a: a3 03 06 00  	sb	zero, 7(a2)
   16e7e: 82 80        	ret

0000000000016e80 <strlen>:
   16e80: 93 77 75 00  	andi	a5, a0, 7
   16e84: 2a 87        	mv	a4, a0
   16e86: a9 eb        	bnez	a5, 0x16ed8 <strlen+0x58>
   16e88: 83 b6 81 82  	ld	a3, -2008(gp)
   16e8c: fd 55        	addi	a1, zero, -1
   16e8e: 10 63        	ld	a2, 0(a4)
   16e90: 21 07        	addi	a4, a4, 8
   16e92: b3 77 d6 00  	and	a5, a2, a3
   16e96: b6 97        	add	a5, a5, a3
   16e98: d1 8f        	or	a5, a5, a2
   16e9a: d5 8f        	or	a5, a5, a3
   16e9c: e3 89 b7 fe  	beq	a5, a1, 0x16e8e <strlen+0xe>
   16ea0: 83 46 87 ff  	lbu	a3, -8(a4)
   16ea4: b3 07 a7 40  	sub	a5, a4, a0
   16ea8: a9 c6        	beqz	a3, 0x16ef2 <strlen+0x72>
   16eaa: 83 46 97 ff  	lbu	a3, -7(a4)
   16eae: 9d ce        	beqz	a3, 0x16eec <strlen+0x6c>
   16eb0: 83 46 a7 ff  	lbu	a3, -6(a4)
   16eb4: a9 c6        	beqz	a3, 0x16efe <strlen+0x7e>
   16eb6: 83 46 b7 ff  	lbu	a3, -5(a4)
   16eba: 9d ce        	beqz	a3, 0x16ef8 <strlen+0x78>
   16ebc: 83 46 c7 ff  	lbu	a3, -4(a4)
   16ec0: b1 c2        	beqz	a3, 0x16f04 <strlen+0x84>
   16ec2: 83 46 d7 ff  	lbu	a3, -3(a4)
   16ec6: b1 c2        	beqz	a3, 0x16f0a <strlen+0x8a>
   16ec8: 03 45 e7 ff  	lbu	a0, -2(a4)
   16ecc: 33 35 a0 00  	snez	a0, a0
   16ed0: 3e 95        	add	a0, a0, a5
   16ed2: 79 15        	addi	a0, a0, -2
   16ed4: 82 80        	ret
   16ed6: cd da        	beqz	a3, 0x16e88 <strlen+0x8>
   16ed8: 83 47 07 00  	lbu	a5, 0(a4)
   16edc: 05 07        	addi	a4, a4, 1
   16ede: 93 76 77 00  	andi	a3, a4, 7
   16ee2: f5 fb        	bnez	a5, 0x16ed6 <strlen+0x56>
   16ee4: 09 8f        	sub	a4, a4, a0
   16ee6: 13 05 f7 ff  	addi	a0, a4, -1
   16eea: 82 80        	ret
   16eec: 13 85 97 ff  	addi	a0, a5, -7
   16ef0: 82 80        	ret
   16ef2: 13 85 87 ff  	addi	a0, a5, -8
   16ef6: 82 80        	ret
   16ef8: 13 85 b7 ff  	addi	a0, a5, -5
   16efc: 82 80        	ret
   16efe: 13 85 a7 ff  	addi	a0, a5, -6
   16f02: 82 80        	ret
   16f04: 13 85 c7 ff  	addi	a0, a5, -4
   16f08: 82 80        	ret
   16f0a: 13 85 d7 ff  	addi	a0, a5, -3
   16f0e: 82 80        	ret

0000000000016f10 <strncpy>:
   16f10: b3 67 b5 00  	or	a5, a0, a1
   16f14: 9d 8b        	andi	a5, a5, 7
   16f16: b9 e3        	bnez	a5, 0x16f5c <strncpy+0x4c>
   16f18: 9d 47        	addi	a5, zero, 7
   16f1a: 63 f1 c7 04  	bgeu	a5, a2, 0x16f5c <strncpy+0x4c>
   16f1e: b7 07 02 00  	lui	a5, 32
   16f22: 03 b3 07 75  	ld	t1, 1872(a5)
   16f26: b7 07 02 00  	lui	a5, 32
   16f2a: 83 b8 87 75  	ld	a7, 1880(a5)
   16f2e: 1d 4e        	addi	t3, zero, 7
   16f30: aa 87        	mv	a5, a0
   16f32: 39 a0        	j	0x16f40 <strncpy+0x30>
   16f34: 94 e3        	sd	a3, 0(a5)
   16f36: 61 16        	addi	a2, a2, -8
   16f38: a1 07        	addi	a5, a5, 8
   16f3a: a1 05        	addi	a1, a1, 8
   16f3c: 63 78 ce 02  	bgeu	t3, a2, 0x16f6c <strncpy+0x5c>
   16f40: 94 61        	ld	a3, 0(a1)
   16f42: 33 87 66 00  	add	a4, a3, t1
   16f46: 13 c8 f6 ff  	not	a6, a3
   16f4a: 33 77 07 01  	and	a4, a4, a6
   16f4e: 33 77 17 01  	and	a4, a4, a7
   16f52: 6d d3        	beqz	a4, 0x16f34 <strncpy+0x24>
   16f54: 85 05        	addi	a1, a1, 1
   16f56: 85 07        	addi	a5, a5, 1
   16f58: 01 e6        	bnez	a2, 0x16f60 <strncpy+0x50>
   16f5a: 82 80        	ret
   16f5c: aa 87        	mv	a5, a0
   16f5e: 39 a0        	j	0x16f6c <strncpy+0x5c>
   16f60: 03 c7 f5 ff  	lbu	a4, -1(a1)
   16f64: 7d 16        	addi	a2, a2, -1
   16f66: a3 8f e7 fe  	sb	a4, -1(a5)
   16f6a: 11 cb        	beqz	a4, 0x16f7e <strncpy+0x6e>
   16f6c: 85 05        	addi	a1, a1, 1
   16f6e: 85 07        	addi	a5, a5, 1
   16f70: 6d d6        	beqz	a2, 0x16f5a <strncpy+0x4a>
   16f72: 03 c7 f5 ff  	lbu	a4, -1(a1)
   16f76: 7d 16        	addi	a2, a2, -1
   16f78: a3 8f e7 fe  	sb	a4, -1(a5)
   16f7c: 65 fb        	bnez	a4, 0x16f6c <strncpy+0x5c>
   16f7e: 57 77 00 04  	<unknown>
   16f82: f3 26 20 c2  	csrr	a3, vlenb
   16f86: 57 3c 00 5e  	<unknown>
   16f8a: 09 ca        	beqz	a2, 0x16f9c <strncpy+0x8c>
   16f8c: 57 77 06 04  	<unknown>
   16f90: 27 8c 07 02  	<unknown>
   16f94: 19 8e        	sub	a2, a2, a4
   16f96: b6 97        	add	a5, a5, a3
   16f98: 75 fa        	bnez	a2, 0x16f8c <strncpy+0x7c>
   16f9a: 82 80        	ret
   16f9c: 82 80        	ret

0000000000016f9e <_svfprintf_r>:
   16f9e: 13 01 01 d9  	addi	sp, sp, -624
   16fa2: 23 34 11 26  	sd	ra, 616(sp)
   16fa6: 23 30 81 26  	sd	s0, 608(sp)
   16faa: 23 38 21 25  	sd	s2, 592(sp)
   16fae: 2e 84        	mv	s0, a1
   16fb0: 2e e0        	sd	a1, 0(sp)
   16fb2: 32 89        	mv	s2, a2
   16fb4: 36 f0        	sd	a3, 32(sp)
   16fb6: 23 3c 91 24  	sd	s1, 600(sp)
   16fba: 23 34 31 25  	sd	s3, 584(sp)
   16fbe: 23 30 41 25  	sd	s4, 576(sp)
   16fc2: 23 3c 51 23  	sd	s5, 568(sp)
   16fc6: 23 38 61 23  	sd	s6, 560(sp)
   16fca: 23 34 71 23  	sd	s7, 552(sp)
   16fce: 23 30 81 23  	sd	s8, 544(sp)
   16fd2: 23 3c 91 21  	sd	s9, 536(sp)
   16fd6: 23 38 a1 21  	sd	s10, 528(sp)
   16fda: 23 34 b1 21  	sd	s11, 520(sp)
   16fde: aa e8        	sd	a0, 80(sp)
   16fe0: ef e0 8f ce  	jal	0x154c8 <_localeconv_r>
   16fe4: 1c 61        	ld	a5, 0(a0)
   16fe6: 3e 85        	mv	a0, a5
   16fe8: be f0        	sd	a5, 96(sp)
   16fea: ef f0 7f e9  	jal	0x16e80 <strlen>
   16fee: 03 57 04 01  	lhu	a4, 16(s0)
   16ff2: 82 f9        	sd	zero, 240(sp)
   16ff4: 82 fd        	sd	zero, 248(sp)
   16ff6: 13 77 07 08  	andi	a4, a4, 128
   16ffa: aa ec        	sd	a0, 88(sp)
   16ffc: 01 c7        	beqz	a4, 0x17004 <_svfprintf_r+0x66>
   16ffe: 18 6c        	ld	a4, 24(s0)
   17000: e3 07 07 68  	beqz	a4, 0x17e8e <_svfprintf_r+0xef0>
   17004: 1c 03        	addi	a5, sp, 384
   17006: 79 67        	lui	a4, 30
   17008: 3e 83        	mv	t1, a5
   1700a: 3e e2        	sd	a5, 256(sp)
   1700c: 4a 8b        	mv	s6, s2
   1700e: 93 07 87 ea  	addi	a5, a4, -344
   17012: 79 67        	lui	a4, 30
   17014: 3e f4        	sd	a5, 40(sp)
   17016: 93 07 07 0b  	addi	a5, a4, 176
   1701a: 03 47 0b 00  	lbu	a4, 0(s6)
   1701e: 02 ea        	sd	zero, 272(sp)
   17020: 23 24 01 10  	sw	zero, 264(sp)
   17024: 02 fc        	sd	zero, 56(sp)
   17026: 82 f4        	sd	zero, 104(sp)
   17028: 82 f8        	sd	zero, 112(sp)
   1702a: 82 fc        	sd	zero, 120(sp)
   1702c: 02 e1        	sd	zero, 128(sp)
   1702e: 02 e5        	sd	zero, 136(sp)
   17030: 02 e8        	sd	zero, 16(sp)
   17032: 3e e4        	sd	a5, 8(sp)
   17034: c6 69        	ld	s3, 80(sp)
   17036: 1a 84        	mv	s0, t1
   17038: 63 01 07 1c  	beqz	a4, 0x171fa <_svfprintf_r+0x25c>
   1703c: da 84        	mv	s1, s6
   1703e: 93 06 50 02  	addi	a3, zero, 37
   17042: 63 0c d7 20  	beq	a4, a3, 0x1725a <_svfprintf_r+0x2bc>
   17046: 03 c7 14 00  	lbu	a4, 1(s1)
   1704a: 85 04        	addi	s1, s1, 1
   1704c: 7d fb        	bnez	a4, 0x17042 <_svfprintf_r+0xa4>
   1704e: 3b 89 64 41  	subw	s2, s1, s6
   17052: 63 04 09 1a  	beqz	s2, 0x171fa <_svfprintf_r+0x25c>
   17056: d2 6b        	ld	s7, 272(sp)
   17058: 83 26 81 10  	lw	a3, 264(sp)
   1705c: 23 30 64 01  	sd	s6, 0(s0)
   17060: 33 87 2b 01  	add	a4, s7, s2
   17064: 1b 86 16 00  	addiw	a2, a3, 1
   17068: 23 34 24 01  	sd	s2, 8(s0)
   1706c: 3a ea        	sd	a4, 272(sp)
   1706e: 23 24 c1 10  	sw	a2, 264(sp)
   17072: 1d 47        	addi	a4, zero, 7
   17074: 41 04        	addi	s0, s0, 16
   17076: 63 49 c7 1e  	blt	a4, a2, 0x17268 <_svfprintf_r+0x2ca>
   1707a: c2 67        	ld	a5, 16(sp)
   1707c: 03 c7 04 00  	lbu	a4, 0(s1)
   17080: bb 87 27 01  	addw	a5, a5, s2
   17084: 3e e8        	sd	a5, 16(sp)
   17086: 63 0a 07 16  	beqz	a4, 0x171fa <_svfprintf_r+0x25c>
   1708a: 7d 53        	addi	t1, zero, -1
   1708c: 03 ce 14 00  	lbu	t3, 1(s1)
   17090: a3 07 01 0c  	sb	zero, 207(sp)
   17094: 85 04        	addi	s1, s1, 1
   17096: 01 4a        	mv	s4, zero
   17098: 01 49        	mv	s2, zero
   1709a: 93 0b a0 05  	addi	s7, zero, 90
   1709e: 9a 8a        	mv	s5, t1
   170a0: 85 04        	addi	s1, s1, 1
   170a2: 1b 0d 0e 00  	sext.w	s10, t3
   170a6: 9b 07 0d fe  	addiw	a5, s10, -32
   170aa: 1b 87 07 00  	sext.w	a4, a5
   170ae: 63 e5 eb 04  	bltu	s7, a4, 0x170f8 <_svfprintf_r+0x15a>
   170b2: 13 97 07 02  	slli	a4, a5, 32
   170b6: 93 57 e7 01  	srli	a5, a4, 30
   170ba: 22 77        	ld	a4, 40(sp)
   170bc: ba 97        	add	a5, a5, a4
   170be: 9c 43        	lw	a5, 0(a5)
   170c0: 82 87        	jr	a5
   170c2: 01 4a        	mv	s4, zero
   170c4: a5 46        	addi	a3, zero, 9
   170c6: 03 c7 04 00  	lbu	a4, 0(s1)
   170ca: 9b 17 2a 00  	slliw	a5, s4, 2
   170ce: 3b 8a 47 01  	addw	s4, a5, s4
   170d2: 1b 0d 0d fd  	addiw	s10, s10, -48
   170d6: 1b 1a 1a 00  	slliw	s4, s4, 1
   170da: 9b 07 07 fd  	addiw	a5, a4, -48
   170de: 3b 0a 4d 01  	addw	s4, s10, s4
   170e2: 85 04        	addi	s1, s1, 1
   170e4: 1b 0d 07 00  	sext.w	s10, a4
   170e8: e3 ff f6 fc  	bgeu	a3, a5, 0x170c6 <_svfprintf_r+0x128>
   170ec: 9b 07 0d fe  	addiw	a5, s10, -32
   170f0: 1b 87 07 00  	sext.w	a4, a5
   170f4: e3 ff eb fa  	bgeu	s7, a4, 0x170b2 <_svfprintf_r+0x114>
   170f8: 63 01 0d 10  	beqz	s10, 0x171fa <_svfprintf_r+0x25c>
   170fc: 23 0c a1 11  	sb	s10, 280(sp)
   17100: a3 07 01 0c  	sb	zero, 207(sp)
   17104: 05 4c        	addi	s8, zero, 1
   17106: 85 4d        	addi	s11, zero, 1
   17108: 13 0b 81 11  	addi	s6, sp, 280
   1710c: 02 ec        	sd	zero, 24(sp)
   1710e: 01 43        	mv	t1, zero
   17110: 82 e4        	sd	zero, 72(sp)
   17112: 82 e0        	sd	zero, 64(sp)
   17114: 02 f8        	sd	zero, 48(sp)
   17116: 93 7c 29 00  	andi	s9, s2, 2
   1711a: 63 83 0c 00  	beqz	s9, 0x17120 <_svfprintf_r+0x182>
   1711e: 09 2c        	addiw	s8, s8, 2
   17120: 93 7a 49 08  	andi	s5, s2, 132
   17124: 63 96 0a 00  	bnez	s5, 0x17130 <_svfprintf_r+0x192>
   17128: 3b 07 8a 41  	subw	a4, s4, s8
   1712c: e3 46 e0 00  	bgtz	a4, 0x17938 <_svfprintf_r+0x99a>
   17130: 03 47 f1 0c  	lbu	a4, 207(sp)
   17134: d2 6b        	ld	s7, 272(sp)
   17136: 1d c3        	beqz	a4, 0x1715c <_svfprintf_r+0x1be>
   17138: 03 26 81 10  	lw	a2, 264(sp)
   1713c: 93 05 f1 0c  	addi	a1, sp, 207
   17140: 0c e0        	sd	a1, 0(s0)
   17142: 05 26        	addiw	a2, a2, 1
   17144: 85 45        	addi	a1, zero, 1
   17146: 85 0b        	addi	s7, s7, 1
   17148: 0c e4        	sd	a1, 8(s0)
   1714a: 23 24 c1 10  	sw	a2, 264(sp)
   1714e: 9b 05 06 00  	sext.w	a1, a2
   17152: 5e ea        	sd	s7, 272(sp)
   17154: 1d 46        	addi	a2, zero, 7
   17156: 41 04        	addi	s0, s0, 16
   17158: 63 4c b6 1c  	blt	a2, a1, 0x17330 <_svfprintf_r+0x392>
   1715c: 63 83 0c 02  	beqz	s9, 0x17182 <_svfprintf_r+0x1e4>
   17160: 03 26 81 10  	lw	a2, 264(sp)
   17164: 89 45        	addi	a1, zero, 2
   17166: 89 0b        	addi	s7, s7, 2
   17168: 05 26        	addiw	a2, a2, 1
   1716a: 9c 09        	addi	a5, sp, 208
   1716c: 0c e4        	sd	a1, 8(s0)
   1716e: 1c e0        	sd	a5, 0(s0)
   17170: 9b 05 06 00  	sext.w	a1, a2
   17174: 23 24 c1 10  	sw	a2, 264(sp)
   17178: 5e ea        	sd	s7, 272(sp)
   1717a: 1d 46        	addi	a2, zero, 7
   1717c: 41 04        	addi	s0, s0, 16
   1717e: e3 49 b6 06  	blt	a2, a1, 0x179f0 <_svfprintf_r+0xa52>
   17182: 13 06 00 08  	addi	a2, zero, 128
   17186: 63 85 ca 66  	beq	s5, a2, 0x177f0 <_svfprintf_r+0x852>
   1718a: bb 0c b3 41  	subw	s9, t1, s11
   1718e: 63 4f 90 6f  	bgtz	s9, 0x1788c <_svfprintf_r+0x8ee>
   17192: 13 76 09 10  	andi	a2, s2, 256
   17196: 63 1a 06 50  	bnez	a2, 0x176aa <_svfprintf_r+0x70c>
   1719a: 03 26 81 10  	lw	a2, 264(sp)
   1719e: 33 87 7d 01  	add	a4, s11, s7
   171a2: 23 30 64 01  	sd	s6, 0(s0)
   171a6: 9b 06 16 00  	addiw	a3, a2, 1
   171aa: 23 34 b4 01  	sd	s11, 8(s0)
   171ae: 3a ea        	sd	a4, 272(sp)
   171b0: 23 24 d1 10  	sw	a3, 264(sp)
   171b4: 9d 47        	addi	a5, zero, 7
   171b6: 63 ce d7 22  	blt	a5, a3, 0x173f2 <_svfprintf_r+0x454>
   171ba: 41 04        	addi	s0, s0, 16
   171bc: 13 79 49 00  	andi	s2, s2, 4
   171c0: 63 06 09 00  	beqz	s2, 0x171cc <_svfprintf_r+0x22e>
   171c4: 3b 09 8a 41  	subw	s2, s4, s8
   171c8: e3 40 20 05  	bgtz	s2, 0x17a08 <_svfprintf_r+0xa6a>
   171cc: d2 87        	mv	a5, s4
   171ce: 63 53 8a 01  	bge	s4, s8, 0x171d4 <_svfprintf_r+0x236>
   171d2: e2 87        	mv	a5, s8
   171d4: c2 66        	ld	a3, 16(sp)
   171d6: b5 9f        	addw	a5, a5, a3
   171d8: 3e e8        	sd	a5, 16(sp)
   171da: 63 11 07 74  	bnez	a4, 0x1791c <_svfprintf_r+0x97e>
   171de: e2 67        	ld	a5, 24(sp)
   171e0: 23 24 01 10  	sw	zero, 264(sp)
   171e4: 89 c7        	beqz	a5, 0x171ee <_svfprintf_r+0x250>
   171e6: e2 65        	ld	a1, 24(sp)
   171e8: 4e 85        	mv	a0, s3
   171ea: ef b0 bf 8a  	jal	0x12a94 <_free_r>
   171ee: 00 03        	addi	s0, sp, 384
   171f0: 26 8b        	mv	s6, s1
   171f2: 03 47 0b 00  	lbu	a4, 0(s6)
   171f6: e3 13 07 e4  	bnez	a4, 0x1703c <_svfprintf_r+0x9e>
   171fa: d2 67        	ld	a5, 272(sp)
   171fc: 99 c3        	beqz	a5, 0x17202 <_svfprintf_r+0x264>
   171fe: 6f 10 80 30  	j	0x18506 <_svfprintf_r+0x1568>
   17202: 82 67        	ld	a5, 0(sp)
   17204: 83 d7 07 01  	lhu	a5, 16(a5)
   17208: 93 f7 07 04  	andi	a5, a5, 64
   1720c: 99 c3        	beqz	a5, 0x17212 <_svfprintf_r+0x274>
   1720e: 6f 10 60 7b  	j	0x189c4 <_svfprintf_r+0x1a26>
   17212: 83 30 81 26  	ld	ra, 616(sp)
   17216: 03 34 01 26  	ld	s0, 608(sp)
   1721a: 42 65        	ld	a0, 16(sp)
   1721c: 83 34 81 25  	ld	s1, 600(sp)
   17220: 03 39 01 25  	ld	s2, 592(sp)
   17224: 83 39 81 24  	ld	s3, 584(sp)
   17228: 03 3a 01 24  	ld	s4, 576(sp)
   1722c: 83 3a 81 23  	ld	s5, 568(sp)
   17230: 03 3b 01 23  	ld	s6, 560(sp)
   17234: 83 3b 81 22  	ld	s7, 552(sp)
   17238: 03 3c 01 22  	ld	s8, 544(sp)
   1723c: 83 3c 81 21  	ld	s9, 536(sp)
   17240: 03 3d 01 21  	ld	s10, 528(sp)
   17244: 83 3d 81 20  	ld	s11, 520(sp)
   17248: 13 01 01 27  	addi	sp, sp, 624
   1724c: 82 80        	ret
   1724e: 13 69 09 01  	ori	s2, s2, 16
   17252: 03 ce 04 00  	lbu	t3, 0(s1)
   17256: 01 29        	sext.w	s2, s2
   17258: a1 b5        	j	0x170a0 <_svfprintf_r+0x102>
   1725a: 3b 89 64 41  	subw	s2, s1, s6
   1725e: e3 1c 09 de  	bnez	s2, 0x17056 <_svfprintf_r+0xb8>
   17262: 03 c7 04 00  	lbu	a4, 0(s1)
   17266: 05 b5        	j	0x17086 <_svfprintf_r+0xe8>
   17268: 82 65        	ld	a1, 0(sp)
   1726a: 10 02        	addi	a2, sp, 256
   1726c: 4e 85        	mv	a0, s3
   1726e: ef 30 00 44  	jal	0x1a6ae <__ssprint_r>
   17272: 41 f9        	bnez	a0, 0x17202 <_svfprintf_r+0x264>
   17274: 00 03        	addi	s0, sp, 384
   17276: 11 b5        	j	0x1707a <_svfprintf_r+0xdc>
   17278: 13 77 89 00  	andi	a4, s2, 8
   1727c: 56 83        	mv	t1, s5
   1727e: e3 0d 07 2e  	beqz	a4, 0x17d78 <_svfprintf_r+0xdda>
   17282: 82 77        	ld	a5, 32(sp)
   17284: 93 8a f7 00  	addi	s5, a5, 15
   17288: 93 fa 0a ff  	andi	s5, s5, -16
   1728c: 03 b7 0a 00  	ld	a4, 0(s5)
   17290: 83 b5 8a 00  	ld	a1, 8(s5)
   17294: 93 87 0a 01  	addi	a5, s5, 16
   17298: 3e f0        	sd	a5, 32(sp)
   1729a: 88 19        	addi	a0, sp, 240
   1729c: 1a ec        	sd	t1, 24(sp)
   1729e: ba f9        	sd	a4, 240(sp)
   172a0: ae fd        	sd	a1, 248(sp)
   172a2: ef e0 ef 9d  	jal	0x15480 <_ldcheck>
   172a6: aa cd        	sw	a0, 216(sp)
   172a8: 09 47        	addi	a4, zero, 2
   172aa: 62 63        	ld	t1, 24(sp)
   172ac: e3 0c e5 6c  	beq	a0, a4, 0x18184 <_svfprintf_r+0x11e6>
   172b0: 05 47        	addi	a4, zero, 1
   172b2: 63 14 e5 00  	bne	a0, a4, 0x172ba <_svfprintf_r+0x31c>
   172b6: 6f 00 d0 7f  	j	0x182b2 <_svfprintf_r+0x1314>
   172ba: 13 07 10 06  	addi	a4, zero, 97
   172be: 63 14 ed 00  	bne	s10, a4, 0x172c6 <_svfprintf_r+0x328>
   172c2: 6f 10 60 4c  	j	0x18788 <_svfprintf_r+0x17ea>
   172c6: 13 07 10 04  	addi	a4, zero, 65
   172ca: 63 14 ed 00  	bne	s10, a4, 0x172d2 <_svfprintf_r+0x334>
   172ce: 6f 10 a0 5e  	j	0x188b8 <_svfprintf_r+0x191a>
   172d2: 7d 57        	addi	a4, zero, -1
   172d4: 63 14 e3 00  	bne	t1, a4, 0x172dc <_svfprintf_r+0x33e>
   172d8: 6f 10 a0 25  	j	0x18532 <_svfprintf_r+0x1594>
   172dc: 13 77 fd fd  	andi	a4, s10, -33
   172e0: 93 06 70 04  	addi	a3, zero, 71
   172e4: 63 14 d7 00  	bne	a4, a3, 0x172ec <_svfprintf_r+0x34e>
   172e8: 6f 10 80 6c  	j	0x189b0 <_svfprintf_r+0x1a12>
   172ec: 13 67 09 10  	ori	a4, s2, 256
   172f0: ee 7c        	ld	s9, 248(sp)
   172f2: 9b 07 07 00  	sext.w	a5, a4
   172f6: 3e e9        	sd	a5, 144(sp)
   172f8: ca e0        	sd	s2, 64(sp)
   172fa: 4e 7c        	ld	s8, 240(sp)
   172fc: 63 d4 0c 00  	bgez	s9, 0x17304 <_svfprintf_r+0x366>
   17300: 6f 10 c0 50  	j	0x1880c <_svfprintf_r+0x186e>
   17304: 02 ed        	sd	zero, 152(sp)
   17306: 02 ec        	sd	zero, 24(sp)
   17308: 1b 07 fd fb  	addiw	a4, s10, -65
   1730c: 1b 06 07 00  	sext.w	a2, a4
   17310: 93 06 50 02  	addi	a3, zero, 37
   17314: 63 f4 c6 00  	bgeu	a3, a2, 0x1731c <_svfprintf_r+0x37e>
   17318: 6f 10 a0 04  	j	0x18362 <_svfprintf_r+0x13c4>
   1731c: 93 16 07 02  	slli	a3, a4, 32
   17320: 13 d7 e6 01  	srli	a4, a3, 30
   17324: f9 66        	lui	a3, 30
   17326: 93 86 46 01  	addi	a3, a3, 20
   1732a: 36 97        	add	a4, a4, a3
   1732c: 18 43        	lw	a4, 0(a4)
   1732e: 02 87        	jr	a4
   17330: 82 65        	ld	a1, 0(sp)
   17332: 10 02        	addi	a2, sp, 256
   17334: 4e 85        	mv	a0, s3
   17336: 1a e9        	sd	t1, 144(sp)
   17338: ef 30 60 37  	jal	0x1a6ae <__ssprint_r>
   1733c: 63 17 05 5e  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17340: d2 6b        	ld	s7, 272(sp)
   17342: 4a 63        	ld	t1, 144(sp)
   17344: 00 03        	addi	s0, sp, 384
   17346: 19 bd        	j	0x1715c <_svfprintf_r+0x1be>
   17348: e2 77        	ld	a5, 56(sp)
   1734a: 85 46        	addi	a3, zero, 1
   1734c: 93 0c 04 01  	addi	s9, s0, 16
   17350: e3 d1 f6 06  	bge	a3, a5, 0x17bb2 <_svfprintf_r+0xc14>
   17354: 03 26 81 10  	lw	a2, 264(sp)
   17358: 85 46        	addi	a3, zero, 1
   1735a: 13 87 1b 00  	addi	a4, s7, 1
   1735e: 05 26        	addiw	a2, a2, 1
   17360: 14 e4        	sd	a3, 8(s0)
   17362: 23 24 c1 10  	sw	a2, 264(sp)
   17366: 9b 06 06 00  	sext.w	a3, a2
   1736a: 23 30 64 01  	sd	s6, 0(s0)
   1736e: 3a ea        	sd	a4, 272(sp)
   17370: 1d 46        	addi	a2, zero, 7
   17372: e3 44 d6 16  	blt	a2, a3, 0x17cda <_svfprintf_r+0xd3c>
   17376: e6 67        	ld	a5, 88(sp)
   17378: 1b 84 16 00  	addiw	s0, a3, 1
   1737c: 23 24 81 10  	sw	s0, 264(sp)
   17380: 33 8d e7 00  	add	s10, a5, a4
   17384: 06 77        	ld	a4, 96(sp)
   17386: 23 b4 fc 00  	sd	a5, 8(s9)
   1738a: 6a ea        	sd	s10, 272(sp)
   1738c: 23 b0 ec 00  	sd	a4, 0(s9)
   17390: 1d 47        	addi	a4, zero, 7
   17392: c1 0c        	addi	s9, s9, 16
   17394: e3 41 87 16  	blt	a4, s0, 0x17cf6 <_svfprintf_r+0xd58>
   17398: 4e 75        	ld	a0, 240(sp)
   1739a: ee 75        	ld	a1, 248(sp)
   1739c: 01 46        	mv	a2, zero
   1739e: 81 46        	mv	a3, zero
   173a0: ef 40 c0 7c  	jal	0x1bb6c <__netf2>
   173a4: e2 77        	ld	a5, 56(sp)
   173a6: 9b 8d f7 ff  	addiw	s11, a5, -1
   173aa: e3 07 05 04  	beqz	a0, 0x17bf8 <_svfprintf_r+0xc5a>
   173ae: 93 07 1b 00  	addi	a5, s6, 1
   173b2: 05 24        	addiw	s0, s0, 1
   173b4: 6e 9d        	add	s10, s10, s11
   173b6: 23 b0 fc 00  	sd	a5, 0(s9)
   173ba: 23 b4 bc 01  	sd	s11, 8(s9)
   173be: 6a ea        	sd	s10, 272(sp)
   173c0: 23 24 81 10  	sw	s0, 264(sp)
   173c4: 9d 47        	addi	a5, zero, 7
   173c6: e3 cb 87 00  	blt	a5, s0, 0x17bdc <_svfprintf_r+0xc3e>
   173ca: c1 0c        	addi	s9, s9, 16
   173cc: c6 76        	ld	a3, 112(sp)
   173ce: 05 24        	addiw	s0, s0, 1
   173d0: 9c 11        	addi	a5, sp, 224
   173d2: 33 87 a6 01  	add	a4, a3, s10
   173d6: 23 b0 fc 00  	sd	a5, 0(s9)
   173da: 23 b4 dc 00  	sd	a3, 8(s9)
   173de: 23 24 81 10  	sw	s0, 264(sp)
   173e2: 9b 06 04 00  	sext.w	a3, s0
   173e6: 3a ea        	sd	a4, 272(sp)
   173e8: 9d 47        	addi	a5, zero, 7
   173ea: 13 84 0c 01  	addi	s0, s9, 16
   173ee: e3 d7 d7 dc  	bge	a5, a3, 0x171bc <_svfprintf_r+0x21e>
   173f2: 82 65        	ld	a1, 0(sp)
   173f4: 10 02        	addi	a2, sp, 256
   173f6: 4e 85        	mv	a0, s3
   173f8: ef 30 60 2b  	jal	0x1a6ae <__ssprint_r>
   173fc: 63 17 05 52  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17400: 52 67        	ld	a4, 272(sp)
   17402: 00 03        	addi	s0, sp, 384
   17404: 65 bb        	j	0x171bc <_svfprintf_r+0x21e>
   17406: 02 77        	ld	a4, 32(sp)
   17408: a3 07 01 0c  	sb	zero, 207(sp)
   1740c: 05 4c        	addi	s8, zero, 1
   1740e: 1c 43        	lw	a5, 0(a4)
   17410: 21 07        	addi	a4, a4, 8
   17412: 3a f0        	sd	a4, 32(sp)
   17414: 23 0c f1 10  	sb	a5, 280(sp)
   17418: 85 4d        	addi	s11, zero, 1
   1741a: 13 0b 81 11  	addi	s6, sp, 280
   1741e: fd b1        	j	0x1710c <_svfprintf_r+0x16e>
   17420: 82 77        	ld	a5, 32(sp)
   17422: a3 07 01 0c  	sb	zero, 207(sp)
   17426: 56 83        	mv	t1, s5
   17428: 03 bb 07 00  	ld	s6, 0(a5)
   1742c: 93 8a 87 00  	addi	s5, a5, 8
   17430: e3 09 0b 58  	beqz	s6, 0x181c2 <_svfprintf_r+0x1224>
   17434: 7d 57        	addi	a4, zero, -1
   17436: e3 05 e3 2c  	beq	t1, a4, 0x17f00 <_svfprintf_r+0xf62>
   1743a: 1a 86        	mv	a2, t1
   1743c: 81 45        	mv	a1, zero
   1743e: 5a 85        	mv	a0, s6
   17440: 1a f0        	sd	t1, 32(sp)
   17442: ef e0 5f 91  	jal	0x15d56 <memchr>
   17446: 2a ec        	sd	a0, 24(sp)
   17448: 02 73        	ld	t1, 32(sp)
   1744a: 19 e1        	bnez	a0, 0x17450 <_svfprintf_r+0x4b2>
   1744c: 6f 10 80 3f  	j	0x18844 <_svfprintf_r+0x18a6>
   17450: e2 67        	ld	a5, 24(sp)
   17452: bb 8d 67 41  	subw	s11, a5, s6
   17456: 03 47 f1 0c  	lbu	a4, 207(sp)
   1745a: 13 c8 fd ff  	not	a6, s11
   1745e: 13 58 f8 43  	srai	a6, a6, 63
   17462: 56 f0        	sd	s5, 32(sp)
   17464: 02 ec        	sd	zero, 24(sp)
   17466: 82 e4        	sd	zero, 72(sp)
   17468: 82 e0        	sd	zero, 64(sp)
   1746a: 02 f8        	sd	zero, 48(sp)
   1746c: 33 7c b8 01  	and	s8, a6, s11
   17470: 01 43        	mv	t1, zero
   17472: e3 02 07 ca  	beqz	a4, 0x17116 <_svfprintf_r+0x178>
   17476: 05 2c        	addiw	s8, s8, 1
   17478: 79 b9        	j	0x17116 <_svfprintf_r+0x178>
   1747a: 82 77        	ld	a5, 32(sp)
   1747c: 03 aa 07 00  	lw	s4, 0(a5)
   17480: a1 07        	addi	a5, a5, 8
   17482: e3 50 0a 0a  	bgez	s4, 0x17d22 <_svfprintf_r+0xd84>
   17486: 3b 0a 40 41  	negw	s4, s4
   1748a: 3e f0        	sd	a5, 32(sp)
   1748c: 13 69 49 00  	ori	s2, s2, 4
   17490: 03 ce 04 00  	lbu	t3, 0(s1)
   17494: 01 29        	sext.w	s2, s2
   17496: 29 b1        	j	0x170a0 <_svfprintf_r+0x102>
   17498: 93 07 b0 02  	addi	a5, zero, 43
   1749c: 03 ce 04 00  	lbu	t3, 0(s1)
   174a0: a3 07 f1 0c  	sb	a5, 207(sp)
   174a4: f5 be        	j	0x170a0 <_svfprintf_r+0x102>
   174a6: 13 69 09 08  	ori	s2, s2, 128
   174aa: 03 ce 04 00  	lbu	t3, 0(s1)
   174ae: 01 29        	sext.w	s2, s2
   174b0: c5 be        	j	0x170a0 <_svfprintf_r+0x102>
   174b2: 03 cd 04 00  	lbu	s10, 0(s1)
   174b6: 93 06 a0 02  	addi	a3, zero, 42
   174ba: 93 87 14 00  	addi	a5, s1, 1
   174be: 63 14 dd 00  	bne	s10, a3, 0x174c6 <_svfprintf_r+0x528>
   174c2: 6f 10 e0 5a  	j	0x18a70 <_svfprintf_r+0x1ad2>
   174c6: 1b 07 0d fd  	addiw	a4, s10, -48
   174ca: 25 46        	addi	a2, zero, 9
   174cc: be 84        	mv	s1, a5
   174ce: 81 4a        	mv	s5, zero
   174d0: a5 46        	addi	a3, zero, 9
   174d2: e3 6a e6 bc  	bltu	a2, a4, 0x170a6 <_svfprintf_r+0x108>
   174d6: 03 cd 04 00  	lbu	s10, 0(s1)
   174da: 9b 97 2a 00  	slliw	a5, s5, 2
   174de: 3b 83 57 01  	addw	t1, a5, s5
   174e2: 1b 13 13 00  	slliw	t1, t1, 1
   174e6: bb 0a e3 00  	addw	s5, t1, a4
   174ea: 1b 07 0d fd  	addiw	a4, s10, -48
   174ee: 85 04        	addi	s1, s1, 1
   174f0: e3 f3 e6 fe  	bgeu	a3, a4, 0x174d6 <_svfprintf_r+0x538>
   174f4: 4d be        	j	0x170a6 <_svfprintf_r+0x108>
   174f6: 4e 85        	mv	a0, s3
   174f8: ef d0 1f fd  	jal	0x154c8 <_localeconv_r>
   174fc: 1c 65        	ld	a5, 8(a0)
   174fe: 3e 85        	mv	a0, a5
   17500: 3e e5        	sd	a5, 136(sp)
   17502: ef f0 ff 97  	jal	0x16e80 <strlen>
   17506: aa 87        	mv	a5, a0
   17508: 4e 85        	mv	a0, s3
   1750a: 3e 8c        	mv	s8, a5
   1750c: 3e e1        	sd	a5, 128(sp)
   1750e: ef d0 bf fb  	jal	0x154c8 <_localeconv_r>
   17512: 1c 69        	ld	a5, 16(a0)
   17514: 03 ce 04 00  	lbu	t3, 0(s1)
   17518: be fc        	sd	a5, 120(sp)
   1751a: e3 03 0c b8  	beqz	s8, 0x170a0 <_svfprintf_r+0x102>
   1751e: e3 81 07 b8  	beqz	a5, 0x170a0 <_svfprintf_r+0x102>
   17522: 83 c7 07 00  	lbu	a5, 0(a5)
   17526: e3 8d 07 b6  	beqz	a5, 0x170a0 <_svfprintf_r+0x102>
   1752a: 13 69 09 40  	ori	s2, s2, 1024
   1752e: 01 29        	sext.w	s2, s2
   17530: 85 be        	j	0x170a0 <_svfprintf_r+0x102>
   17532: 13 69 19 00  	ori	s2, s2, 1
   17536: 03 ce 04 00  	lbu	t3, 0(s1)
   1753a: 01 29        	sext.w	s2, s2
   1753c: 95 b6        	j	0x170a0 <_svfprintf_r+0x102>
   1753e: 83 47 f1 0c  	lbu	a5, 207(sp)
   17542: 03 ce 04 00  	lbu	t3, 0(s1)
   17546: e3 9d 07 b4  	bnez	a5, 0x170a0 <_svfprintf_r+0x102>
   1754a: 93 07 00 02  	addi	a5, zero, 32
   1754e: a3 07 f1 0c  	sb	a5, 207(sp)
   17552: b9 b6        	j	0x170a0 <_svfprintf_r+0x102>
   17554: 93 6c 09 01  	ori	s9, s2, 16
   17558: 56 83        	mv	t1, s5
   1755a: 81 2c        	sext.w	s9, s9
   1755c: 82 76        	ld	a3, 32(sp)
   1755e: 93 f7 0c 02  	andi	a5, s9, 32
   17562: 13 87 86 00  	addi	a4, a3, 8
   17566: 63 8d 07 70  	beqz	a5, 0x17c80 <_svfprintf_r+0xce2>
   1756a: 83 bb 06 00  	ld	s7, 0(a3)
   1756e: 85 47        	addi	a5, zero, 1
   17570: 3a f0        	sd	a4, 32(sp)
   17572: 01 47        	mv	a4, zero
   17574: a3 07 e1 0c  	sb	a4, 207(sp)
   17578: 7d 57        	addi	a4, zero, -1
   1757a: 63 0a e3 1e  	beq	t1, a4, 0x1776e <_svfprintf_r+0x7d0>
   1757e: 13 f9 fc f7  	andi	s2, s9, -129
   17582: 01 29        	sext.w	s2, s2
   17584: 63 94 0b 1e  	bnez	s7, 0x1776c <_svfprintf_r+0x7ce>
   17588: 63 17 03 22  	bnez	t1, 0x177b6 <_svfprintf_r+0x818>
   1758c: 63 93 07 74  	bnez	a5, 0x17cd2 <_svfprintf_r+0xd34>
   17590: 93 fd 1c 00  	andi	s11, s9, 1
   17594: 13 0b c1 17  	addi	s6, sp, 380
   17598: 63 9a 0d 78  	bnez	s11, 0x17d2c <_svfprintf_r+0xd8e>
   1759c: 03 47 f1 0c  	lbu	a4, 207(sp)
   175a0: 1a 88        	mv	a6, t1
   175a2: 63 53 b3 01  	bge	t1, s11, 0x175a8 <_svfprintf_r+0x60a>
   175a6: 6e 88        	mv	a6, s11
   175a8: 02 ec        	sd	zero, 24(sp)
   175aa: 82 e4        	sd	zero, 72(sp)
   175ac: 82 e0        	sd	zero, 64(sp)
   175ae: 02 f8        	sd	zero, 48(sp)
   175b0: 1b 0c 08 00  	sext.w	s8, a6
   175b4: e3 11 07 ec  	bnez	a4, 0x17476 <_svfprintf_r+0x4d8>
   175b8: b9 be        	j	0x17116 <_svfprintf_r+0x178>
   175ba: 13 69 89 00  	ori	s2, s2, 8
   175be: 03 ce 04 00  	lbu	t3, 0(s1)
   175c2: 01 29        	sext.w	s2, s2
   175c4: f1 bc        	j	0x170a0 <_svfprintf_r+0x102>
   175c6: 13 69 09 01  	ori	s2, s2, 16
   175ca: 56 83        	mv	t1, s5
   175cc: 01 29        	sext.w	s2, s2
   175ce: 82 76        	ld	a3, 32(sp)
   175d0: 93 77 09 02  	andi	a5, s2, 32
   175d4: 13 87 86 00  	addi	a4, a3, 8
   175d8: 63 84 07 6c  	beqz	a5, 0x17ca0 <_svfprintf_r+0xd02>
   175dc: 9c 62        	ld	a5, 0(a3)
   175de: be 8b        	mv	s7, a5
   175e0: 63 c1 07 6e  	bltz	a5, 0x17cc2 <_svfprintf_r+0xd24>
   175e4: fd 57        	addi	a5, zero, -1
   175e6: 63 07 f3 3e  	beq	t1, a5, 0x179d4 <_svfprintf_r+0xa36>
   175ea: 93 7c f9 f7  	andi	s9, s2, -129
   175ee: 81 2c        	sext.w	s9, s9
   175f0: 63 96 0b 7a  	bnez	s7, 0x17d9c <_svfprintf_r+0xdfe>
   175f4: 63 04 03 00  	beqz	t1, 0x175fc <_svfprintf_r+0x65e>
   175f8: 6f 10 40 3d  	j	0x189cc <_svfprintf_r+0x1a2e>
   175fc: 81 4d        	mv	s11, zero
   175fe: 3a f0        	sd	a4, 32(sp)
   17600: 66 89        	mv	s2, s9
   17602: 13 0b c1 17  	addi	s6, sp, 380
   17606: 59 bf        	j	0x1759c <_svfprintf_r+0x5fe>
   17608: 13 69 09 01  	ori	s2, s2, 16
   1760c: 56 83        	mv	t1, s5
   1760e: 01 29        	sext.w	s2, s2
   17610: 82 76        	ld	a3, 32(sp)
   17612: 93 77 09 02  	andi	a5, s2, 32
   17616: 13 87 86 00  	addi	a4, a3, 8
   1761a: 63 85 07 64  	beqz	a5, 0x17c64 <_svfprintf_r+0xcc6>
   1761e: 83 bb 06 00  	ld	s7, 0(a3)
   17622: 93 7c f9 bf  	andi	s9, s2, -1025
   17626: 81 2c        	sext.w	s9, s9
   17628: 3a f0        	sd	a4, 32(sp)
   1762a: 81 47        	mv	a5, zero
   1762c: 99 b7        	j	0x17572 <_svfprintf_r+0x5d4>
   1762e: 13 69 09 02  	ori	s2, s2, 32
   17632: 03 ce 04 00  	lbu	t3, 0(s1)
   17636: 01 29        	sext.w	s2, s2
   17638: a5 b4        	j	0x170a0 <_svfprintf_r+0x102>
   1763a: 02 77        	ld	a4, 32(sp)
   1763c: 93 77 09 02  	andi	a5, s2, 32
   17640: 18 63        	ld	a4, 0(a4)
   17642: 63 8b 07 70  	beqz	a5, 0x17d58 <_svfprintf_r+0xdba>
   17646: c2 67        	ld	a5, 16(sp)
   17648: 1c e3        	sd	a5, 0(a4)
   1764a: 82 77        	ld	a5, 32(sp)
   1764c: 26 8b        	mv	s6, s1
   1764e: a1 07        	addi	a5, a5, 8
   17650: 3e f0        	sd	a5, 32(sp)
   17652: 45 b6        	j	0x171f2 <_svfprintf_r+0x254>
   17654: 03 ce 04 00  	lbu	t3, 0(s1)
   17658: 93 07 c0 06  	addi	a5, zero, 108
   1765c: e3 01 fe 02  	beq	t3, a5, 0x17e7e <_svfprintf_r+0xee0>
   17660: 13 69 09 01  	ori	s2, s2, 16
   17664: 01 29        	sext.w	s2, s2
   17666: 2d bc        	j	0x170a0 <_svfprintf_r+0x102>
   17668: 03 ce 04 00  	lbu	t3, 0(s1)
   1766c: 93 07 80 06  	addi	a5, zero, 104
   17670: 63 0f fe 7e  	beq	t3, a5, 0x17e6e <_svfprintf_r+0xed0>
   17674: 13 69 09 04  	ori	s2, s2, 64
   17678: 01 29        	sext.w	s2, s2
   1767a: 1d b4        	j	0x170a0 <_svfprintf_r+0x102>
   1767c: 02 77        	ld	a4, 32(sp)
   1767e: e1 77        	lui	a5, 1048568
   17680: 93 c7 07 83  	xori	a5, a5, -2000
   17684: 23 18 f1 0c  	sh	a5, 208(sp)
   17688: 93 07 87 00  	addi	a5, a4, 8
   1768c: 3e f0        	sd	a5, 32(sp)
   1768e: f5 67        	lui	a5, 29
   17690: 93 87 87 77  	addi	a5, a5, 1912
   17694: 93 6c 29 00  	ori	s9, s2, 2
   17698: be f4        	sd	a5, 104(sp)
   1769a: 83 3b 07 00  	ld	s7, 0(a4)
   1769e: 56 83        	mv	t1, s5
   176a0: 81 2c        	sext.w	s9, s9
   176a2: 89 47        	addi	a5, zero, 2
   176a4: 13 0d 80 07  	addi	s10, zero, 120
   176a8: e9 b5        	j	0x17572 <_svfprintf_r+0x5d4>
   176aa: 93 06 50 06  	addi	a3, zero, 101
   176ae: e3 dd a6 c9  	bge	a3, s10, 0x17348 <_svfprintf_r+0x3aa>
   176b2: 4e 75        	ld	a0, 240(sp)
   176b4: ee 75        	ld	a1, 248(sp)
   176b6: 01 46        	mv	a2, zero
   176b8: 81 46        	mv	a3, zero
   176ba: ef 40 20 4b  	jal	0x1bb6c <__netf2>
   176be: 63 15 05 3c  	bnez	a0, 0x17a88 <_svfprintf_r+0xaea>
   176c2: 83 27 81 10  	lw	a5, 264(sp)
   176c6: f5 66        	lui	a3, 29
   176c8: 93 86 86 79  	addi	a3, a3, 1944
   176cc: 85 27        	addiw	a5, a5, 1
   176ce: 14 e0        	sd	a3, 0(s0)
   176d0: 85 0b        	addi	s7, s7, 1
   176d2: 85 46        	addi	a3, zero, 1
   176d4: 14 e4        	sd	a3, 8(s0)
   176d6: 23 24 f1 10  	sw	a5, 264(sp)
   176da: 9b 86 07 00  	sext.w	a3, a5
   176de: 5e ea        	sd	s7, 272(sp)
   176e0: 9d 47        	addi	a5, zero, 7
   176e2: 41 04        	addi	s0, s0, 16
   176e4: e3 c7 d7 22  	blt	a5, a3, 0x18112 <_svfprintf_r+0x1174>
   176e8: ee 47        	lw	a5, 216(sp)
   176ea: 62 77        	ld	a4, 56(sp)
   176ec: 63 d3 e7 62  	bge	a5, a4, 0x17d12 <_svfprintf_r+0xd74>
   176f0: e6 66        	ld	a3, 88(sp)
   176f2: 83 27 81 10  	lw	a5, 264(sp)
   176f6: 06 76        	ld	a2, 96(sp)
   176f8: 33 87 76 01  	add	a4, a3, s7
   176fc: 85 27        	addiw	a5, a5, 1
   176fe: 14 e4        	sd	a3, 8(s0)
   17700: 10 e0        	sd	a2, 0(s0)
   17702: 9b 86 07 00  	sext.w	a3, a5
   17706: 23 24 f1 10  	sw	a5, 264(sp)
   1770a: 3a ea        	sd	a4, 272(sp)
   1770c: 9d 47        	addi	a5, zero, 7
   1770e: 41 04        	addi	s0, s0, 16
   17710: 63 cc d7 72  	blt	a5, a3, 0x17e48 <_svfprintf_r+0xeaa>
   17714: e2 77        	ld	a5, 56(sp)
   17716: 9b 8c f7 ff  	addiw	s9, a5, -1
   1771a: e3 51 90 ab  	blez	s9, 0x171bc <_svfprintf_r+0x21e>
   1771e: 79 6f        	lui	t5, 30
   17720: c1 46        	addi	a3, zero, 16
   17722: 83 27 81 10  	lw	a5, 264(sp)
   17726: 93 0b 0f 0b  	addi	s7, t5, 176
   1772a: e3 df 96 1f  	bge	a3, s9, 0x18128 <_svfprintf_r+0x118a>
   1772e: 82 6a        	ld	s5, 0(sp)
   17730: 41 4d        	addi	s10, zero, 16
   17732: 9d 4d        	addi	s11, zero, 7
   17734: 21 a0        	j	0x1773c <_svfprintf_r+0x79e>
   17736: c1 3c        	addiw	s9, s9, -16
   17738: e3 58 9d 1f  	bge	s10, s9, 0x18128 <_svfprintf_r+0x118a>
   1773c: 85 27        	addiw	a5, a5, 1
   1773e: 41 07        	addi	a4, a4, 16
   17740: 23 30 74 01  	sd	s7, 0(s0)
   17744: 23 34 a4 01  	sd	s10, 8(s0)
   17748: 3a ea        	sd	a4, 272(sp)
   1774a: 23 24 f1 10  	sw	a5, 264(sp)
   1774e: 41 04        	addi	s0, s0, 16
   17750: e3 d3 fd fe  	bge	s11, a5, 0x17736 <_svfprintf_r+0x798>
   17754: 10 02        	addi	a2, sp, 256
   17756: d6 85        	mv	a1, s5
   17758: 4e 85        	mv	a0, s3
   1775a: ef 20 50 75  	jal	0x1a6ae <__ssprint_r>
   1775e: 63 16 05 1c  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17762: 52 67        	ld	a4, 272(sp)
   17764: 83 27 81 10  	lw	a5, 264(sp)
   17768: 00 03        	addi	s0, sp, 384
   1776a: f1 b7        	j	0x17736 <_svfprintf_r+0x798>
   1776c: ca 8c        	mv	s9, s2
   1776e: 05 47        	addi	a4, zero, 1
   17770: 63 84 e7 26  	beq	a5, a4, 0x179d8 <_svfprintf_r+0xa3a>
   17774: 09 47        	addi	a4, zero, 2
   17776: 63 89 e7 04  	beq	a5, a4, 0x177c8 <_svfprintf_r+0x82a>
   1777a: 93 0a c1 17  	addi	s5, sp, 380
   1777e: 56 8b        	mv	s6, s5
   17780: 13 f7 7b 00  	andi	a4, s7, 7
   17784: 13 07 07 03  	addi	a4, a4, 48
   17788: a3 0f eb fe  	sb	a4, -1(s6)
   1778c: 93 db 3b 00  	srli	s7, s7, 3
   17790: 5a 86        	mv	a2, s6
   17792: 7d 1b        	addi	s6, s6, -1
   17794: e3 96 0b fe  	bnez	s7, 0x17780 <_svfprintf_r+0x7e2>
   17798: 93 f5 1c 00  	andi	a1, s9, 1
   1779c: b1 c5        	beqz	a1, 0x177e8 <_svfprintf_r+0x84a>
   1779e: 93 05 00 03  	addi	a1, zero, 48
   177a2: 63 03 b7 04  	beq	a4, a1, 0x177e8 <_svfprintf_r+0x84a>
   177a6: 79 16        	addi	a2, a2, -2
   177a8: a3 0f bb fe  	sb	a1, -1(s6)
   177ac: bb 8d ca 40  	subw	s11, s5, a2
   177b0: 66 89        	mv	s2, s9
   177b2: 32 8b        	mv	s6, a2
   177b4: e5 b3        	j	0x1759c <_svfprintf_r+0x5fe>
   177b6: 05 47        	addi	a4, zero, 1
   177b8: 63 94 e7 00  	bne	a5, a4, 0x177c0 <_svfprintf_r+0x822>
   177bc: 6f 10 20 10  	j	0x188be <_svfprintf_r+0x1920>
   177c0: 09 47        	addi	a4, zero, 2
   177c2: ca 8c        	mv	s9, s2
   177c4: e3 9b e7 fa  	bne	a5, a4, 0x1777a <_svfprintf_r+0x7dc>
   177c8: 26 77        	ld	a4, 104(sp)
   177ca: 93 0a c1 17  	addi	s5, sp, 380
   177ce: 56 8b        	mv	s6, s5
   177d0: 93 f7 fb 00  	andi	a5, s7, 15
   177d4: ba 97        	add	a5, a5, a4
   177d6: 83 c7 07 00  	lbu	a5, 0(a5)
   177da: 93 db 4b 00  	srli	s7, s7, 4
   177de: 7d 1b        	addi	s6, s6, -1
   177e0: 23 00 fb 00  	sb	a5, 0(s6)
   177e4: e3 96 0b fe  	bnez	s7, 0x177d0 <_svfprintf_r+0x832>
   177e8: bb 8d 6a 41  	subw	s11, s5, s6
   177ec: 66 89        	mv	s2, s9
   177ee: 7d b3        	j	0x1759c <_svfprintf_r+0x5fe>
   177f0: bb 0a 8a 41  	subw	s5, s4, s8
   177f4: e3 5b 50 99  	blez	s5, 0x1718a <_svfprintf_r+0x1ec>
   177f8: 79 6f        	lui	t5, 30
   177fa: c1 45        	addi	a1, zero, 16
   177fc: 5e 87        	mv	a4, s7
   177fe: 03 26 81 10  	lw	a2, 264(sp)
   17802: 93 0b 0f 0b  	addi	s7, t5, 176
   17806: 63 d0 55 07  	bge	a1, s5, 0x17866 <_svfprintf_r+0x8c8>
   1780a: 26 e9        	sd	s1, 144(sp)
   1780c: c1 4f        	addi	t6, zero, 16
   1780e: d6 84        	mv	s1, s5
   17810: 9d 4c        	addi	s9, zero, 7
   17812: d2 8a        	mv	s5, s4
   17814: 4a 8a        	mv	s4, s2
   17816: 5e 89        	mv	s2, s7
   17818: da 8b        	mv	s7, s6
   1781a: 1a 8b        	mv	s6, t1
   1781c: 21 a0        	j	0x17824 <_svfprintf_r+0x886>
   1781e: c1 34        	addiw	s1, s1, -16
   17820: 63 dc 9f 02  	bge	t6, s1, 0x17858 <_svfprintf_r+0x8ba>
   17824: 05 26        	addiw	a2, a2, 1
   17826: 41 07        	addi	a4, a4, 16
   17828: 23 30 24 01  	sd	s2, 0(s0)
   1782c: 23 34 f4 01  	sd	t6, 8(s0)
   17830: 3a ea        	sd	a4, 272(sp)
   17832: 23 24 c1 10  	sw	a2, 264(sp)
   17836: 41 04        	addi	s0, s0, 16
   17838: e3 d3 cc fe  	bge	s9, a2, 0x1781e <_svfprintf_r+0x880>
   1783c: 82 65        	ld	a1, 0(sp)
   1783e: 10 02        	addi	a2, sp, 256
   17840: 4e 85        	mv	a0, s3
   17842: ef 20 d0 66  	jal	0x1a6ae <__ssprint_r>
   17846: 75 e1        	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17848: c1 4f        	addi	t6, zero, 16
   1784a: c1 34        	addiw	s1, s1, -16
   1784c: 52 67        	ld	a4, 272(sp)
   1784e: 03 26 81 10  	lw	a2, 264(sp)
   17852: 00 03        	addi	s0, sp, 384
   17854: e3 c8 9f fc  	blt	t6, s1, 0x17824 <_svfprintf_r+0x886>
   17858: 5a 83        	mv	t1, s6
   1785a: 5e 8b        	mv	s6, s7
   1785c: ca 8b        	mv	s7, s2
   1785e: 52 89        	mv	s2, s4
   17860: 56 8a        	mv	s4, s5
   17862: a6 8a        	mv	s5, s1
   17864: ca 64        	ld	s1, 144(sp)
   17866: 9b 05 16 00  	addiw	a1, a2, 1
   1786a: 56 97        	add	a4, a4, s5
   1786c: 23 30 74 01  	sd	s7, 0(s0)
   17870: 23 34 54 01  	sd	s5, 8(s0)
   17874: 3a ea        	sd	a4, 272(sp)
   17876: 23 24 b1 10  	sw	a1, 264(sp)
   1787a: 1d 46        	addi	a2, zero, 7
   1787c: e3 47 b6 0e  	blt	a2, a1, 0x1816a <_svfprintf_r+0x11cc>
   17880: bb 0c b3 41  	subw	s9, t1, s11
   17884: 41 04        	addi	s0, s0, 16
   17886: ba 8b        	mv	s7, a4
   17888: e3 55 90 91  	blez	s9, 0x17192 <_svfprintf_r+0x1f4>
   1788c: 79 6f        	lui	t5, 30
   1788e: c1 45        	addi	a1, zero, 16
   17890: 5e 87        	mv	a4, s7
   17892: 03 26 81 10  	lw	a2, 264(sp)
   17896: 93 0b 0f 0b  	addi	s7, t5, 176
   1789a: 63 d1 95 07  	bge	a1, s9, 0x178fc <_svfprintf_r+0x95e>
   1789e: a2 6b        	ld	s7, 8(sp)
   178a0: c1 48        	addi	a7, zero, 16
   178a2: 9d 4a        	addi	s5, zero, 7
   178a4: de 87        	mv	a5, s7
   178a6: da 8b        	mv	s7, s6
   178a8: 52 8b        	mv	s6, s4
   178aa: 4a 8a        	mv	s4, s2
   178ac: 26 89        	mv	s2, s1
   178ae: e6 84        	mv	s1, s9
   178b0: be 8c        	mv	s9, a5
   178b2: 21 a0        	j	0x178ba <_svfprintf_r+0x91c>
   178b4: c1 34        	addiw	s1, s1, -16
   178b6: 63 dc 98 02  	bge	a7, s1, 0x178ee <_svfprintf_r+0x950>
   178ba: a2 66        	ld	a3, 8(sp)
   178bc: 05 26        	addiw	a2, a2, 1
   178be: 41 07        	addi	a4, a4, 16
   178c0: 14 e0        	sd	a3, 0(s0)
   178c2: 23 34 14 01  	sd	a7, 8(s0)
   178c6: 3a ea        	sd	a4, 272(sp)
   178c8: 23 24 c1 10  	sw	a2, 264(sp)
   178cc: 41 04        	addi	s0, s0, 16
   178ce: e3 d3 ca fe  	bge	s5, a2, 0x178b4 <_svfprintf_r+0x916>
   178d2: 82 65        	ld	a1, 0(sp)
   178d4: 10 02        	addi	a2, sp, 256
   178d6: 4e 85        	mv	a0, s3
   178d8: ef 20 70 5d  	jal	0x1a6ae <__ssprint_r>
   178dc: 39 e5        	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   178de: c1 48        	addi	a7, zero, 16
   178e0: c1 34        	addiw	s1, s1, -16
   178e2: 52 67        	ld	a4, 272(sp)
   178e4: 03 26 81 10  	lw	a2, 264(sp)
   178e8: 00 03        	addi	s0, sp, 384
   178ea: e3 c8 98 fc  	blt	a7, s1, 0x178ba <_svfprintf_r+0x91c>
   178ee: e6 87        	mv	a5, s9
   178f0: a6 8c        	mv	s9, s1
   178f2: ca 84        	mv	s1, s2
   178f4: 52 89        	mv	s2, s4
   178f6: 5a 8a        	mv	s4, s6
   178f8: 5e 8b        	mv	s6, s7
   178fa: be 8b        	mv	s7, a5
   178fc: 9b 05 16 00  	addiw	a1, a2, 1
   17900: 66 97        	add	a4, a4, s9
   17902: 23 30 74 01  	sd	s7, 0(s0)
   17906: 23 34 94 01  	sd	s9, 8(s0)
   1790a: 3a ea        	sd	a4, 272(sp)
   1790c: 23 24 b1 10  	sw	a1, 264(sp)
   17910: 1d 46        	addi	a2, zero, 7
   17912: 63 4e b6 32  	blt	a2, a1, 0x17c4e <_svfprintf_r+0xcb0>
   17916: 41 04        	addi	s0, s0, 16
   17918: ba 8b        	mv	s7, a4
   1791a: a5 b8        	j	0x17192 <_svfprintf_r+0x1f4>
   1791c: 82 65        	ld	a1, 0(sp)
   1791e: 10 02        	addi	a2, sp, 256
   17920: 4e 85        	mv	a0, s3
   17922: ef 20 d0 58  	jal	0x1a6ae <__ssprint_r>
   17926: e3 0c 05 8a  	beqz	a0, 0x171de <_svfprintf_r+0x240>
   1792a: e2 65        	ld	a1, 24(sp)
   1792c: e3 8b 05 8c  	beqz	a1, 0x17202 <_svfprintf_r+0x264>
   17930: 46 65        	ld	a0, 80(sp)
   17932: ef b0 2f 96  	jal	0x12a94 <_free_r>
   17936: f1 b0        	j	0x17202 <_svfprintf_r+0x264>
   17938: f9 68        	lui	a7, 30
   1793a: 41 45        	addi	a0, zero, 16
   1793c: d2 65        	ld	a1, 272(sp)
   1793e: 03 26 81 10  	lw	a2, 264(sp)
   17942: 93 8b 08 0c  	addi	s7, a7, 192
   17946: 63 5e e5 04  	bge	a0, a4, 0x179a2 <_svfprintf_r+0xa04>
   1794a: 26 e9        	sd	s1, 144(sp)
   1794c: a2 87        	mv	a5, s0
   1794e: de 84        	mv	s1, s7
   17950: 41 4f        	addi	t5, zero, 16
   17952: da 8b        	mv	s7, s6
   17954: 9d 43        	addi	t2, zero, 7
   17956: 3a 84        	mv	s0, a4
   17958: 1a 8b        	mv	s6, t1
   1795a: 21 a0        	j	0x17962 <_svfprintf_r+0x9c4>
   1795c: 41 34        	addiw	s0, s0, -16
   1795e: 63 5c 8f 02  	bge	t5, s0, 0x17996 <_svfprintf_r+0x9f8>
   17962: 05 26        	addiw	a2, a2, 1
   17964: c1 05        	addi	a1, a1, 16
   17966: 84 e3        	sd	s1, 0(a5)
   17968: 23 b4 e7 01  	sd	t5, 8(a5)
   1796c: 2e ea        	sd	a1, 272(sp)
   1796e: 23 24 c1 10  	sw	a2, 264(sp)
   17972: c1 07        	addi	a5, a5, 16
   17974: e3 d4 c3 fe  	bge	t2, a2, 0x1795c <_svfprintf_r+0x9be>
   17978: 82 65        	ld	a1, 0(sp)
   1797a: 10 02        	addi	a2, sp, 256
   1797c: 4e 85        	mv	a0, s3
   1797e: ef 20 10 53  	jal	0x1a6ae <__ssprint_r>
   17982: 45 f5        	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17984: 41 4f        	addi	t5, zero, 16
   17986: 41 34        	addiw	s0, s0, -16
   17988: d2 65        	ld	a1, 272(sp)
   1798a: 03 26 81 10  	lw	a2, 264(sp)
   1798e: 1c 03        	addi	a5, sp, 384
   17990: 9d 43        	addi	t2, zero, 7
   17992: e3 48 8f fc  	blt	t5, s0, 0x17962 <_svfprintf_r+0x9c4>
   17996: 5a 83        	mv	t1, s6
   17998: 5e 8b        	mv	s6, s7
   1799a: a6 8b        	mv	s7, s1
   1799c: ca 64        	ld	s1, 144(sp)
   1799e: 22 87        	mv	a4, s0
   179a0: 3e 84        	mv	s0, a5
   179a2: ba 95        	add	a1, a1, a4
   179a4: 05 26        	addiw	a2, a2, 1
   179a6: 18 e4        	sd	a4, 8(s0)
   179a8: 23 30 74 01  	sd	s7, 0(s0)
   179ac: 2e ea        	sd	a1, 272(sp)
   179ae: 23 24 c1 10  	sw	a2, 264(sp)
   179b2: 9b 05 06 00  	sext.w	a1, a2
   179b6: 1d 47        	addi	a4, zero, 7
   179b8: 41 04        	addi	s0, s0, 16
   179ba: 63 5b b7 f6  	bge	a4, a1, 0x17130 <_svfprintf_r+0x192>
   179be: 82 65        	ld	a1, 0(sp)
   179c0: 10 02        	addi	a2, sp, 256
   179c2: 4e 85        	mv	a0, s3
   179c4: 1a e9        	sd	t1, 144(sp)
   179c6: ef 20 90 4e  	jal	0x1a6ae <__ssprint_r>
   179ca: 25 f1        	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   179cc: 4a 63        	ld	t1, 144(sp)
   179ce: 00 03        	addi	s0, sp, 384
   179d0: 6f f0 0f f6  	j	0x17130 <_svfprintf_r+0x192>
   179d4: ca 8c        	mv	s9, s2
   179d6: 3a f0        	sd	a4, 32(sp)
   179d8: a5 47        	addi	a5, zero, 9
   179da: 63 e5 77 3d  	bltu	a5, s7, 0x17da4 <_svfprintf_r+0xe06>
   179de: 9b 8b 0b 03  	addiw	s7, s7, 48
   179e2: a3 0d 71 17  	sb	s7, 379(sp)
   179e6: 66 89        	mv	s2, s9
   179e8: 85 4d        	addi	s11, zero, 1
   179ea: 13 0b b1 17  	addi	s6, sp, 379
   179ee: 7d b6        	j	0x1759c <_svfprintf_r+0x5fe>
   179f0: 82 65        	ld	a1, 0(sp)
   179f2: 10 02        	addi	a2, sp, 256
   179f4: 4e 85        	mv	a0, s3
   179f6: 1a e9        	sd	t1, 144(sp)
   179f8: ef 20 70 4b  	jal	0x1a6ae <__ssprint_r>
   179fc: 1d f5        	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   179fe: d2 6b        	ld	s7, 272(sp)
   17a00: 4a 63        	ld	t1, 144(sp)
   17a02: 00 03        	addi	s0, sp, 384
   17a04: 6f f0 ef f7  	j	0x17182 <_svfprintf_r+0x1e4>
   17a08: f9 68        	lui	a7, 30
   17a0a: c1 46        	addi	a3, zero, 16
   17a0c: 83 27 81 10  	lw	a5, 264(sp)
   17a10: 93 8b 08 0c  	addi	s7, a7, 192
   17a14: 63 d3 26 05  	bge	a3, s2, 0x17a5a <_svfprintf_r+0xabc>
   17a18: 82 6a        	ld	s5, 0(sp)
   17a1a: c1 4c        	addi	s9, zero, 16
   17a1c: 1d 4d        	addi	s10, zero, 7
   17a1e: 21 a0        	j	0x17a26 <_svfprintf_r+0xa88>
   17a20: 41 39        	addiw	s2, s2, -16
   17a22: 63 dc 2c 03  	bge	s9, s2, 0x17a5a <_svfprintf_r+0xabc>
   17a26: 85 27        	addiw	a5, a5, 1
   17a28: 41 07        	addi	a4, a4, 16
   17a2a: 23 30 74 01  	sd	s7, 0(s0)
   17a2e: 23 34 94 01  	sd	s9, 8(s0)
   17a32: 3a ea        	sd	a4, 272(sp)
   17a34: 23 24 f1 10  	sw	a5, 264(sp)
   17a38: 41 04        	addi	s0, s0, 16
   17a3a: e3 53 fd fe  	bge	s10, a5, 0x17a20 <_svfprintf_r+0xa82>
   17a3e: 10 02        	addi	a2, sp, 256
   17a40: d6 85        	mv	a1, s5
   17a42: 4e 85        	mv	a0, s3
   17a44: ef 20 b0 46  	jal	0x1a6ae <__ssprint_r>
   17a48: e3 11 05 ee  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17a4c: 41 39        	addiw	s2, s2, -16
   17a4e: 52 67        	ld	a4, 272(sp)
   17a50: 83 27 81 10  	lw	a5, 264(sp)
   17a54: 00 03        	addi	s0, sp, 384
   17a56: e3 c8 2c fd  	blt	s9, s2, 0x17a26 <_svfprintf_r+0xa88>
   17a5a: 9b 86 17 00  	addiw	a3, a5, 1
   17a5e: 4a 97        	add	a4, a4, s2
   17a60: 23 30 74 01  	sd	s7, 0(s0)
   17a64: 23 34 24 01  	sd	s2, 8(s0)
   17a68: 3a ea        	sd	a4, 272(sp)
   17a6a: 23 24 d1 10  	sw	a3, 264(sp)
   17a6e: 9d 47        	addi	a5, zero, 7
   17a70: 63 de d7 f4  	bge	a5, a3, 0x171cc <_svfprintf_r+0x22e>
   17a74: 82 65        	ld	a1, 0(sp)
   17a76: 10 02        	addi	a2, sp, 256
   17a78: 4e 85        	mv	a0, s3
   17a7a: ef 20 50 43  	jal	0x1a6ae <__ssprint_r>
   17a7e: e3 16 05 ea  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17a82: 52 67        	ld	a4, 272(sp)
   17a84: 6f f0 8f f4  	j	0x171cc <_svfprintf_r+0x22e>
   17a88: 6e 46        	lw	a2, 216(sp)
   17a8a: 63 53 c0 5e  	blez	a2, 0x18070 <_svfprintf_r+0x10d2>
   17a8e: e2 77        	ld	a5, 56(sp)
   17a90: 42 77        	ld	a4, 48(sp)
   17a92: b3 0a fb 00  	add	s5, s6, a5
   17a96: be 86        	mv	a3, a5
   17a98: 63 4c f7 2e  	blt	a4, a5, 0x17d90 <_svfprintf_r+0xdf2>
   17a9c: 1b 8d 06 00  	sext.w	s10, a3
   17aa0: 63 52 a0 03  	blez	s10, 0x17ac4 <_svfprintf_r+0xb26>
   17aa4: 03 26 81 10  	lw	a2, 264(sp)
   17aa8: ea 9b        	add	s7, s7, s10
   17aaa: 23 30 64 01  	sd	s6, 0(s0)
   17aae: 9b 05 16 00  	addiw	a1, a2, 1
   17ab2: 23 34 a4 01  	sd	s10, 8(s0)
   17ab6: 5e ea        	sd	s7, 272(sp)
   17ab8: 23 24 b1 10  	sw	a1, 264(sp)
   17abc: 1d 46        	addi	a2, zero, 7
   17abe: 41 04        	addi	s0, s0, 16
   17ac0: e3 47 b6 06  	blt	a2, a1, 0x1832e <_svfprintf_r+0x1390>
   17ac4: 13 46 fd ff  	not	a2, s10
   17ac8: c2 77        	ld	a5, 48(sp)
   17aca: 7d 96        	srai	a2, a2, 63
   17acc: b3 76 cd 00  	and	a3, s10, a2
   17ad0: 3b 8d d7 40  	subw	s10, a5, a3
   17ad4: 63 4d a0 3d  	bgtz	s10, 0x17eae <_svfprintf_r+0xf10>
   17ad8: c2 77        	ld	a5, 48(sp)
   17ada: 93 76 09 40  	andi	a3, s2, 1024
   17ade: 3e 9b        	add	s6, s6, a5
   17ae0: 63 97 06 42  	bnez	a3, 0x17f0e <_svfprintf_r+0xf70>
   17ae4: ee 46        	lw	a3, 216(sp)
   17ae6: e2 77        	ld	a5, 56(sp)
   17ae8: 63 c6 f6 00  	blt	a3, a5, 0x17af4 <_svfprintf_r+0xb56>
   17aec: 13 76 19 00  	andi	a2, s2, 1
   17af0: e3 0c 06 04  	beqz	a2, 0x18348 <_svfprintf_r+0x13aa>
   17af4: 03 26 81 10  	lw	a2, 264(sp)
   17af8: e6 67        	ld	a5, 88(sp)
   17afa: 86 75        	ld	a1, 96(sp)
   17afc: 41 04        	addi	s0, s0, 16
   17afe: 33 87 77 01  	add	a4, a5, s7
   17b02: 23 38 b4 fe  	sd	a1, -16(s0)
   17b06: 9b 05 16 00  	addiw	a1, a2, 1
   17b0a: 23 3c f4 fe  	sd	a5, -8(s0)
   17b0e: 3a ea        	sd	a4, 272(sp)
   17b10: 23 24 b1 10  	sw	a1, 264(sp)
   17b14: 1d 46        	addi	a2, zero, 7
   17b16: e3 44 b6 50  	blt	a2, a1, 0x1881e <_svfprintf_r+0x1880>
   17b1a: e2 77        	ld	a5, 56(sp)
   17b1c: b3 8c 6a 41  	sub	s9, s5, s6
   17b20: 9b 85 0c 00  	sext.w	a1, s9
   17b24: 3b 86 d7 40  	subw	a2, a5, a3
   17b28: 63 53 b6 00  	bge	a2, a1, 0x17b2e <_svfprintf_r+0xb90>
   17b2c: b2 8c        	mv	s9, a2
   17b2e: 81 2c        	sext.w	s9, s9
   17b30: 63 51 90 03  	blez	s9, 0x17b52 <_svfprintf_r+0xbb4>
   17b34: 83 26 81 10  	lw	a3, 264(sp)
   17b38: 66 97        	add	a4, a4, s9
   17b3a: 23 30 64 01  	sd	s6, 0(s0)
   17b3e: 85 26        	addiw	a3, a3, 1
   17b40: 23 34 94 01  	sd	s9, 8(s0)
   17b44: 3a ea        	sd	a4, 272(sp)
   17b46: 23 24 d1 10  	sw	a3, 264(sp)
   17b4a: 9d 47        	addi	a5, zero, 7
   17b4c: 41 04        	addi	s0, s0, 16
   17b4e: e3 c1 d7 54  	blt	a5, a3, 0x18890 <_svfprintf_r+0x18f2>
   17b52: 93 c7 fc ff  	not	a5, s9
   17b56: fd 97        	srai	a5, a5, 63
   17b58: b3 fc fc 00  	and	s9, s9, a5
   17b5c: bb 0c 96 41  	subw	s9, a2, s9
   17b60: 63 5e 90 e5  	blez	s9, 0x171bc <_svfprintf_r+0x21e>
   17b64: 79 6f        	lui	t5, 30
   17b66: c1 46        	addi	a3, zero, 16
   17b68: 83 27 81 10  	lw	a5, 264(sp)
   17b6c: 93 0b 0f 0b  	addi	s7, t5, 176
   17b70: 63 dc 96 5b  	bge	a3, s9, 0x18128 <_svfprintf_r+0x118a>
   17b74: 82 6a        	ld	s5, 0(sp)
   17b76: 41 4d        	addi	s10, zero, 16
   17b78: 9d 4d        	addi	s11, zero, 7
   17b7a: 21 a0        	j	0x17b82 <_svfprintf_r+0xbe4>
   17b7c: c1 3c        	addiw	s9, s9, -16
   17b7e: 63 55 9d 5b  	bge	s10, s9, 0x18128 <_svfprintf_r+0x118a>
   17b82: 85 27        	addiw	a5, a5, 1
   17b84: 41 07        	addi	a4, a4, 16
   17b86: 23 30 74 01  	sd	s7, 0(s0)
   17b8a: 23 34 a4 01  	sd	s10, 8(s0)
   17b8e: 3a ea        	sd	a4, 272(sp)
   17b90: 23 24 f1 10  	sw	a5, 264(sp)
   17b94: 41 04        	addi	s0, s0, 16
   17b96: e3 d3 fd fe  	bge	s11, a5, 0x17b7c <_svfprintf_r+0xbde>
   17b9a: 10 02        	addi	a2, sp, 256
   17b9c: d6 85        	mv	a1, s5
   17b9e: 4e 85        	mv	a0, s3
   17ba0: ef 20 f0 30  	jal	0x1a6ae <__ssprint_r>
   17ba4: e3 13 05 d8  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17ba8: 52 67        	ld	a4, 272(sp)
   17baa: 83 27 81 10  	lw	a5, 264(sp)
   17bae: 00 03        	addi	s0, sp, 384
   17bb0: f1 b7        	j	0x17b7c <_svfprintf_r+0xbde>
   17bb2: 13 76 19 00  	andi	a2, s2, 1
   17bb6: 63 1f 06 f8  	bnez	a2, 0x17354 <_svfprintf_r+0x3b6>
   17bba: 03 26 81 10  	lw	a2, 264(sp)
   17bbe: 13 8d 1b 00  	addi	s10, s7, 1
   17bc2: 23 30 64 01  	sd	s6, 0(s0)
   17bc6: 9b 07 16 00  	addiw	a5, a2, 1
   17bca: 14 e4        	sd	a3, 8(s0)
   17bcc: 23 24 f1 10  	sw	a5, 264(sp)
   17bd0: 1b 84 07 00  	sext.w	s0, a5
   17bd4: 6a ea        	sd	s10, 272(sp)
   17bd6: 9d 47        	addi	a5, zero, 7
   17bd8: 63 da 87 fe  	bge	a5, s0, 0x173cc <_svfprintf_r+0x42e>
   17bdc: 82 65        	ld	a1, 0(sp)
   17bde: 10 02        	addi	a2, sp, 256
   17be0: 4e 85        	mv	a0, s3
   17be2: ef 20 d0 2c  	jal	0x1a6ae <__ssprint_r>
   17be6: e3 12 05 d4  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17bea: 52 6d        	ld	s10, 272(sp)
   17bec: 03 24 81 10  	lw	s0, 264(sp)
   17bf0: 93 0c 01 18  	addi	s9, sp, 384
   17bf4: 6f f0 8f fd  	j	0x173cc <_svfprintf_r+0x42e>
   17bf8: 62 77        	ld	a4, 56(sp)
   17bfa: 85 47        	addi	a5, zero, 1
   17bfc: 63 d8 e7 fc  	bge	a5, a4, 0x173cc <_svfprintf_r+0x42e>
   17c00: 79 6f        	lui	t5, 30
   17c02: c5 47        	addi	a5, zero, 17
   17c04: 93 0b 0f 0b  	addi	s7, t5, 176
   17c08: 63 da e7 5c  	bge	a5, a4, 0x181dc <_svfprintf_r+0x123e>
   17c0c: 82 6a        	ld	s5, 0(sp)
   17c0e: 41 47        	addi	a4, zero, 16
   17c10: 1d 4b        	addi	s6, zero, 7
   17c12: 21 a0        	j	0x17c1a <_svfprintf_r+0xc7c>
   17c14: c1 3d        	addiw	s11, s11, -16
   17c16: 63 53 b7 5d  	bge	a4, s11, 0x181dc <_svfprintf_r+0x123e>
   17c1a: 05 24        	addiw	s0, s0, 1
   17c1c: 41 0d        	addi	s10, s10, 16
   17c1e: 23 b0 7c 01  	sd	s7, 0(s9)
   17c22: 23 b4 ec 00  	sd	a4, 8(s9)
   17c26: 6a ea        	sd	s10, 272(sp)
   17c28: 23 24 81 10  	sw	s0, 264(sp)
   17c2c: c1 0c        	addi	s9, s9, 16
   17c2e: e3 53 8b fe  	bge	s6, s0, 0x17c14 <_svfprintf_r+0xc76>
   17c32: 10 02        	addi	a2, sp, 256
   17c34: d6 85        	mv	a1, s5
   17c36: 4e 85        	mv	a0, s3
   17c38: ef 20 70 27  	jal	0x1a6ae <__ssprint_r>
   17c3c: e3 17 05 ce  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17c40: 52 6d        	ld	s10, 272(sp)
   17c42: 03 24 81 10  	lw	s0, 264(sp)
   17c46: 93 0c 01 18  	addi	s9, sp, 384
   17c4a: 41 47        	addi	a4, zero, 16
   17c4c: e1 b7        	j	0x17c14 <_svfprintf_r+0xc76>
   17c4e: 82 65        	ld	a1, 0(sp)
   17c50: 10 02        	addi	a2, sp, 256
   17c52: 4e 85        	mv	a0, s3
   17c54: ef 20 b0 25  	jal	0x1a6ae <__ssprint_r>
   17c58: e3 19 05 cc  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17c5c: d2 6b        	ld	s7, 272(sp)
   17c5e: 00 03        	addi	s0, sp, 384
   17c60: 6f f0 2f d3  	j	0x17192 <_svfprintf_r+0x1f4>
   17c64: 93 77 09 01  	andi	a5, s2, 16
   17c68: e5 e7        	bnez	a5, 0x17d50 <_svfprintf_r+0xdb2>
   17c6a: 82 76        	ld	a3, 32(sp)
   17c6c: 93 77 09 04  	andi	a5, s2, 64
   17c70: 83 ab 06 00  	lw	s7, 0(a3)
   17c74: 63 8b 07 5e  	beqz	a5, 0x1826a <_svfprintf_r+0x12cc>
   17c78: c2 1b        	slli	s7, s7, 48
   17c7a: 93 db 0b 03  	srli	s7, s7, 48
   17c7e: 55 b2        	j	0x17622 <_svfprintf_r+0x684>
   17c80: 93 f7 0c 01  	andi	a5, s9, 16
   17c84: dd ef        	bnez	a5, 0x17d42 <_svfprintf_r+0xda4>
   17c86: 82 76        	ld	a3, 32(sp)
   17c88: 93 f7 0c 04  	andi	a5, s9, 64
   17c8c: 83 ab 06 00  	lw	s7, 0(a3)
   17c90: 63 85 07 5e  	beqz	a5, 0x1827a <_svfprintf_r+0x12dc>
   17c94: c2 1b        	slli	s7, s7, 48
   17c96: 93 db 0b 03  	srli	s7, s7, 48
   17c9a: 3a f0        	sd	a4, 32(sp)
   17c9c: 85 47        	addi	a5, zero, 1
   17c9e: d1 b8        	j	0x17572 <_svfprintf_r+0x5d4>
   17ca0: 93 77 09 01  	andi	a5, s2, 16
   17ca4: d9 eb        	bnez	a5, 0x17d3a <_svfprintf_r+0xd9c>
   17ca6: 82 76        	ld	a3, 32(sp)
   17ca8: 93 77 09 04  	andi	a5, s2, 64
   17cac: 83 ab 06 00  	lw	s7, 0(a3)
   17cb0: 63 8f 07 5c  	beqz	a5, 0x1828e <_svfprintf_r+0x12f0>
   17cb4: 9b 9b 0b 01  	slliw	s7, s7, 16
   17cb8: 9b db 0b 41  	sraiw	s7, s7, 16
   17cbc: de 87        	mv	a5, s7
   17cbe: e3 d3 07 92  	bgez	a5, 0x175e4 <_svfprintf_r+0x646>
   17cc2: 3a f0        	sd	a4, 32(sp)
   17cc4: b3 0b 70 41  	neg	s7, s7
   17cc8: ca 8c        	mv	s9, s2
   17cca: 13 07 d0 02  	addi	a4, zero, 45
   17cce: 85 47        	addi	a5, zero, 1
   17cd0: 55 b0        	j	0x17574 <_svfprintf_r+0x5d6>
   17cd2: 81 4d        	mv	s11, zero
   17cd4: 13 0b c1 17  	addi	s6, sp, 380
   17cd8: d1 b0        	j	0x1759c <_svfprintf_r+0x5fe>
   17cda: 82 65        	ld	a1, 0(sp)
   17cdc: 10 02        	addi	a2, sp, 256
   17cde: 4e 85        	mv	a0, s3
   17ce0: ef 20 f0 1c  	jal	0x1a6ae <__ssprint_r>
   17ce4: e3 13 05 c4  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17ce8: 52 67        	ld	a4, 272(sp)
   17cea: 83 26 81 10  	lw	a3, 264(sp)
   17cee: 93 0c 01 18  	addi	s9, sp, 384
   17cf2: 6f f0 4f e8  	j	0x17376 <_svfprintf_r+0x3d8>
   17cf6: 82 65        	ld	a1, 0(sp)
   17cf8: 10 02        	addi	a2, sp, 256
   17cfa: 4e 85        	mv	a0, s3
   17cfc: ef 20 30 1b  	jal	0x1a6ae <__ssprint_r>
   17d00: e3 15 05 c2  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17d04: 52 6d        	ld	s10, 272(sp)
   17d06: 03 24 81 10  	lw	s0, 264(sp)
   17d0a: 93 0c 01 18  	addi	s9, sp, 384
   17d0e: 6f f0 af e8  	j	0x17398 <_svfprintf_r+0x3fa>
   17d12: 93 77 19 00  	andi	a5, s2, 1
   17d16: 5e 87        	mv	a4, s7
   17d18: 63 82 07 ca  	beqz	a5, 0x171bc <_svfprintf_r+0x21e>
   17d1c: d1 ba        	j	0x176f0 <_svfprintf_r+0x752>
   17d1e: 56 83        	mv	t1, s5
   17d20: 7d b0        	j	0x175ce <_svfprintf_r+0x630>
   17d22: 03 ce 04 00  	lbu	t3, 0(s1)
   17d26: 3e f0        	sd	a5, 32(sp)
   17d28: 6f f0 8f b7  	j	0x170a0 <_svfprintf_r+0x102>
   17d2c: 93 07 00 03  	addi	a5, zero, 48
   17d30: a3 0d f1 16  	sb	a5, 379(sp)
   17d34: 13 0b b1 17  	addi	s6, sp, 379
   17d38: 95 b0        	j	0x1759c <_svfprintf_r+0x5fe>
   17d3a: 82 77        	ld	a5, 32(sp)
   17d3c: 9c 63        	ld	a5, 0(a5)
   17d3e: be 8b        	mv	s7, a5
   17d40: 45 b0        	j	0x175e0 <_svfprintf_r+0x642>
   17d42: 82 77        	ld	a5, 32(sp)
   17d44: 3a f0        	sd	a4, 32(sp)
   17d46: 83 bb 07 00  	ld	s7, 0(a5)
   17d4a: 85 47        	addi	a5, zero, 1
   17d4c: 6f f0 7f 82  	j	0x17572 <_svfprintf_r+0x5d4>
   17d50: 82 77        	ld	a5, 32(sp)
   17d52: 83 bb 07 00  	ld	s7, 0(a5)
   17d56: f1 b0        	j	0x17622 <_svfprintf_r+0x684>
   17d58: 93 77 09 01  	andi	a5, s2, 16
   17d5c: e3 95 07 8e  	bnez	a5, 0x17646 <_svfprintf_r+0x6a8>
   17d60: 93 77 09 04  	andi	a5, s2, 64
   17d64: 63 92 07 78  	bnez	a5, 0x184e8 <_svfprintf_r+0x154a>
   17d68: 13 79 09 20  	andi	s2, s2, 512
   17d6c: c2 67        	ld	a5, 16(sp)
   17d6e: e3 02 09 46  	beqz	s2, 0x189d2 <_svfprintf_r+0x1a34>
   17d72: 23 00 f7 00  	sb	a5, 0(a4)
   17d76: d1 b8        	j	0x1764a <_svfprintf_r+0x6ac>
   17d78: 82 77        	ld	a5, 32(sp)
   17d7a: 56 ec        	sd	s5, 24(sp)
   17d7c: 88 23        	fld	fa0, 0(a5)
   17d7e: 93 8a 87 00  	addi	s5, a5, 8
   17d82: 56 f0        	sd	s5, 32(sp)
   17d84: ef 50 20 04  	jal	0x1cdc6 <__extenddftf2>
   17d88: 62 63        	ld	t1, 24(sp)
   17d8a: 2a 87        	mv	a4, a0
   17d8c: 6f f0 ef d0  	j	0x1729a <_svfprintf_r+0x2fc>
   17d90: ba 86        	mv	a3, a4
   17d92: 1b 8d 06 00  	sext.w	s10, a3
   17d96: e3 47 a0 d1  	bgtz	s10, 0x17aa4 <_svfprintf_r+0xb06>
   17d9a: 2d b3        	j	0x17ac4 <_svfprintf_r+0xb26>
   17d9c: 3a f0        	sd	a4, 32(sp)
   17d9e: a5 47        	addi	a5, zero, 9
   17da0: e3 ff 77 c3  	bgeu	a5, s7, 0x179de <_svfprintf_r+0xa40>
   17da4: 93 0a c1 17  	addi	s5, sp, 380
   17da8: 66 78        	ld	a6, 120(sp)
   17daa: 93 f7 0c 40  	andi	a5, s9, 1024
   17dae: 81 46        	mv	a3, zero
   17db0: d6 85        	mv	a1, s5
   17db2: 29 49        	addi	s2, zero, 10
   17db4: a5 4d        	addi	s11, zero, 9
   17db6: 13 0c f0 0f  	addi	s8, zero, 255
   17dba: 39 a0        	j	0x17dc8 <_svfprintf_r+0xe2a>
   17dbc: 33 d7 2b 03  	divu	a4, s7, s2
   17dc0: 63 f6 7d 03  	bgeu	s11, s7, 0x17dec <_svfprintf_r+0xe4e>
   17dc4: ba 8b        	mv	s7, a4
   17dc6: da 85        	mv	a1, s6
   17dc8: 13 8b f5 ff  	addi	s6, a1, -1
   17dcc: 85 26        	addiw	a3, a3, 1
   17dce: 33 f7 2b 03  	remu	a4, s7, s2
   17dd2: 1b 07 07 03  	addiw	a4, a4, 48
   17dd6: a3 8f e5 fe  	sb	a4, -1(a1)
   17dda: ed d3        	beqz	a5, 0x17dbc <_svfprintf_r+0xe1e>
   17ddc: 03 47 08 00  	lbu	a4, 0(a6)
   17de0: e3 9e e6 fc  	bne	a3, a4, 0x17dbc <_svfprintf_r+0xe1e>
   17de4: e3 0c 87 fd  	beq	a4, s8, 0x17dbc <_svfprintf_r+0xe1e>
   17de8: 63 e7 7d 51  	bltu	s11, s7, 0x182f6 <_svfprintf_r+0x1358>
   17dec: c2 fc        	sd	a6, 120(sp)
   17dee: 36 fc        	sd	a3, 56(sp)
   17df0: bb 8d 6a 41  	subw	s11, s5, s6
   17df4: 66 89        	mv	s2, s9
   17df6: 6f f0 6f fa  	j	0x1759c <_svfprintf_r+0x5fe>
   17dfa: f5 67        	lui	a5, 29
   17dfc: 93 87 07 76  	addi	a5, a5, 1888
   17e00: 56 83        	mv	t1, s5
   17e02: be f4        	sd	a5, 104(sp)
   17e04: 82 76        	ld	a3, 32(sp)
   17e06: 93 77 09 02  	andi	a5, s2, 32
   17e0a: 13 87 86 00  	addi	a4, a3, 8
   17e0e: 63 8e 07 22  	beqz	a5, 0x1804a <_svfprintf_r+0x10ac>
   17e12: 83 bb 06 00  	ld	s7, 0(a3)
   17e16: 93 77 19 00  	andi	a5, s2, 1
   17e1a: 81 cf        	beqz	a5, 0x17e32 <_svfprintf_r+0xe94>
   17e1c: 63 8b 0b 00  	beqz	s7, 0x17e32 <_svfprintf_r+0xe94>
   17e20: 13 69 29 00  	ori	s2, s2, 2
   17e24: 93 07 00 03  	addi	a5, zero, 48
   17e28: 23 08 f1 0c  	sb	a5, 208(sp)
   17e2c: a3 08 a1 0d  	sb	s10, 209(sp)
   17e30: 01 29        	sext.w	s2, s2
   17e32: 93 7c f9 bf  	andi	s9, s2, -1025
   17e36: 81 2c        	sext.w	s9, s9
   17e38: 3a f0        	sd	a4, 32(sp)
   17e3a: 89 47        	addi	a5, zero, 2
   17e3c: 6f f0 6f f3  	j	0x17572 <_svfprintf_r+0x5d4>
   17e40: 56 83        	mv	t1, s5
   17e42: ca 8c        	mv	s9, s2
   17e44: 6f f0 8f f1  	j	0x1755c <_svfprintf_r+0x5be>
   17e48: 82 65        	ld	a1, 0(sp)
   17e4a: 10 02        	addi	a2, sp, 256
   17e4c: 4e 85        	mv	a0, s3
   17e4e: ef 20 10 06  	jal	0x1a6ae <__ssprint_r>
   17e52: e3 1c 05 ac  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17e56: 52 67        	ld	a4, 272(sp)
   17e58: 00 03        	addi	s0, sp, 384
   17e5a: 6d b8        	j	0x17714 <_svfprintf_r+0x776>
   17e5c: f5 67        	lui	a5, 29
   17e5e: 93 87 87 77  	addi	a5, a5, 1912
   17e62: 56 83        	mv	t1, s5
   17e64: be f4        	sd	a5, 104(sp)
   17e66: 79 bf        	j	0x17e04 <_svfprintf_r+0xe66>
   17e68: 56 83        	mv	t1, s5
   17e6a: 6f f0 6f fa  	j	0x17610 <_svfprintf_r+0x672>
   17e6e: 13 69 09 20  	ori	s2, s2, 512
   17e72: 03 ce 14 00  	lbu	t3, 1(s1)
   17e76: 01 29        	sext.w	s2, s2
   17e78: 85 04        	addi	s1, s1, 1
   17e7a: 6f f0 6f a2  	j	0x170a0 <_svfprintf_r+0x102>
   17e7e: 13 69 09 02  	ori	s2, s2, 32
   17e82: 03 ce 14 00  	lbu	t3, 1(s1)
   17e86: 01 29        	sext.w	s2, s2
   17e88: 85 04        	addi	s1, s1, 1
   17e8a: 6f f0 6f a1  	j	0x170a0 <_svfprintf_r+0x102>
   17e8e: 46 65        	ld	a0, 80(sp)
   17e90: 93 05 00 04  	addi	a1, zero, 64
   17e94: ef d0 7f 88  	jal	0x1571a <_malloc_r>
   17e98: 82 67        	ld	a5, 0(sp)
   17e9a: 88 e3        	sd	a0, 0(a5)
   17e9c: 88 ef        	sd	a0, 24(a5)
   17e9e: e3 00 05 32  	beqz	a0, 0x189be <_svfprintf_r+0x1a20>
   17ea2: 82 67        	ld	a5, 0(sp)
   17ea4: 13 07 00 04  	addi	a4, zero, 64
   17ea8: 98 d3        	sw	a4, 32(a5)
   17eaa: 6f f0 af 95  	j	0x17004 <_svfprintf_r+0x66>
   17eae: 79 6f        	lui	t5, 30
   17eb0: c1 45        	addi	a1, zero, 16
   17eb2: 5e 87        	mv	a4, s7
   17eb4: 03 26 81 10  	lw	a2, 264(sp)
   17eb8: 93 0b 0f 0b  	addi	s7, t5, 176
   17ebc: 63 dc a5 41  	bge	a1, s10, 0x182d4 <_svfprintf_r+0x1336>
   17ec0: 82 6c        	ld	s9, 0(sp)
   17ec2: c1 48        	addi	a7, zero, 16
   17ec4: 9d 4d        	addi	s11, zero, 7
   17ec6: 21 a0        	j	0x17ece <_svfprintf_r+0xf30>
   17ec8: 41 3d        	addiw	s10, s10, -16
   17eca: 63 d5 a8 41  	bge	a7, s10, 0x182d4 <_svfprintf_r+0x1336>
   17ece: 05 26        	addiw	a2, a2, 1
   17ed0: 41 07        	addi	a4, a4, 16
   17ed2: 23 30 74 01  	sd	s7, 0(s0)
   17ed6: 23 34 14 01  	sd	a7, 8(s0)
   17eda: 3a ea        	sd	a4, 272(sp)
   17edc: 23 24 c1 10  	sw	a2, 264(sp)
   17ee0: 41 04        	addi	s0, s0, 16
   17ee2: e3 d3 cd fe  	bge	s11, a2, 0x17ec8 <_svfprintf_r+0xf2a>
   17ee6: 10 02        	addi	a2, sp, 256
   17ee8: e6 85        	mv	a1, s9
   17eea: 4e 85        	mv	a0, s3
   17eec: ef 20 20 7c  	jal	0x1a6ae <__ssprint_r>
   17ef0: e3 1d 05 a2  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   17ef4: 52 67        	ld	a4, 272(sp)
   17ef6: 03 26 81 10  	lw	a2, 264(sp)
   17efa: 00 03        	addi	s0, sp, 384
   17efc: c1 48        	addi	a7, zero, 16
   17efe: e9 b7        	j	0x17ec8 <_svfprintf_r+0xf2a>
   17f00: 5a 85        	mv	a0, s6
   17f02: ef e0 ff f7  	jal	0x16e80 <strlen>
   17f06: 9b 0d 05 00  	sext.w	s11, a0
   17f0a: 6f f0 cf d4  	j	0x17456 <_svfprintf_r+0x4b8>
   17f0e: 06 6d        	ld	s10, 64(sp)
   17f10: 79 6e        	lui	t3, 30
   17f12: 26 e9        	sd	s1, 144(sp)
   17f14: 4a ed        	sd	s2, 152(sp)
   17f16: 52 f1        	sd	s4, 160(sp)
   17f18: e2 e0        	sd	s8, 64(sp)
   17f1a: 5a 8a        	mv	s4, s6
   17f1c: 66 79        	ld	s2, 120(sp)
   17f1e: aa 64        	ld	s1, 136(sp)
   17f20: 0a 6b        	ld	s6, 128(sp)
   17f22: 26 6c        	ld	s8, 72(sp)
   17f24: 82 6c        	ld	s9, 0(sp)
   17f26: 1d 43        	addi	t1, zero, 7
   17f28: c1 48        	addi	a7, zero, 16
   17f2a: 93 0d 0e 0b  	addi	s11, t3, 176
   17f2e: 5e 87        	mv	a4, s7
   17f30: 63 00 0d 08  	beqz	s10, 0x17fb0 <_svfprintf_r+0x1012>
   17f34: 63 08 0c 10  	beqz	s8, 0x18044 <_svfprintf_r+0x10a6>
   17f38: 7d 3c        	addiw	s8, s8, -1
   17f3a: 83 27 81 10  	lw	a5, 264(sp)
   17f3e: 5a 97        	add	a4, a4, s6
   17f40: 04 e0        	sd	s1, 0(s0)
   17f42: 9b 86 17 00  	addiw	a3, a5, 1
   17f46: 23 34 64 01  	sd	s6, 8(s0)
   17f4a: 3a ea        	sd	a4, 272(sp)
   17f4c: 23 24 d1 10  	sw	a3, 264(sp)
   17f50: 41 04        	addi	s0, s0, 16
   17f52: 63 43 d3 10  	blt	t1, a3, 0x18058 <_svfprintf_r+0x10ba>
   17f56: 83 46 09 00  	lbu	a3, 0(s2)
   17f5a: b3 8b 4a 41  	sub	s7, s5, s4
   17f5e: 9b 87 0b 00  	sext.w	a5, s7
   17f62: 1b 86 06 00  	sext.w	a2, a3
   17f66: 63 d3 f6 00  	bge	a3, a5, 0x17f6c <_svfprintf_r+0xfce>
   17f6a: b2 8b        	mv	s7, a2
   17f6c: 81 2b        	sext.w	s7, s7
   17f6e: 63 55 70 03  	blez	s7, 0x17f98 <_svfprintf_r+0xffa>
   17f72: 83 27 81 10  	lw	a5, 264(sp)
   17f76: 5e 97        	add	a4, a4, s7
   17f78: 23 30 44 01  	sd	s4, 0(s0)
   17f7c: 9b 86 17 00  	addiw	a3, a5, 1
   17f80: 23 34 74 01  	sd	s7, 8(s0)
   17f84: 3a ea        	sd	a4, 272(sp)
   17f86: 23 24 d1 10  	sw	a3, 264(sp)
   17f8a: 63 40 d3 1c  	blt	t1, a3, 0x1814a <_svfprintf_r+0x11ac>
   17f8e: 83 46 09 00  	lbu	a3, 0(s2)
   17f92: 41 04        	addi	s0, s0, 16
   17f94: 1b 86 06 00  	sext.w	a2, a3
   17f98: 93 c7 fb ff  	not	a5, s7
   17f9c: fd 97        	srai	a5, a5, 63
   17f9e: b3 f7 fb 00  	and	a5, s7, a5
   17fa2: bb 0b f6 40  	subw	s7, a2, a5
   17fa6: 63 42 70 03  	bgtz	s7, 0x17fca <_svfprintf_r+0x102c>
   17faa: 36 9a        	add	s4, s4, a3
   17fac: e3 14 0d f8  	bnez	s10, 0x17f34 <_svfprintf_r+0xf96>
   17fb0: e3 14 0c f8  	bnez	s8, 0x17f38 <_svfprintf_r+0xf9a>
   17fb4: ca fc        	sd	s2, 120(sp)
   17fb6: 52 8b        	mv	s6, s4
   17fb8: ca 64        	ld	s1, 144(sp)
   17fba: 6a 69        	ld	s2, 152(sp)
   17fbc: 0a 7a        	ld	s4, 160(sp)
   17fbe: 06 6c        	ld	s8, 64(sp)
   17fc0: ba 8b        	mv	s7, a4
   17fc2: e3 f1 6a b3  	bgeu	s5, s6, 0x17ae4 <_svfprintf_r+0xb46>
   17fc6: 56 8b        	mv	s6, s5
   17fc8: 31 be        	j	0x17ae4 <_svfprintf_r+0xb46>
   17fca: 79 66        	lui	a2, 30
   17fcc: 83 27 81 10  	lw	a5, 264(sp)
   17fd0: 13 06 06 0b  	addi	a2, a2, 176
   17fd4: 63 d8 78 05  	bge	a7, s7, 0x18024 <_svfprintf_r+0x1086>
   17fd8: 26 f8        	sd	s1, 48(sp)
   17fda: de 84        	mv	s1, s7
   17fdc: ee 8b        	mv	s7, s11
   17fde: 21 a0        	j	0x17fe6 <_svfprintf_r+0x1048>
   17fe0: c1 34        	addiw	s1, s1, -16
   17fe2: 63 de 98 02  	bge	a7, s1, 0x1801e <_svfprintf_r+0x1080>
   17fe6: 85 27        	addiw	a5, a5, 1
   17fe8: 41 07        	addi	a4, a4, 16
   17fea: 23 30 b4 01  	sd	s11, 0(s0)
   17fee: 23 34 14 01  	sd	a7, 8(s0)
   17ff2: 3a ea        	sd	a4, 272(sp)
   17ff4: 23 24 f1 10  	sw	a5, 264(sp)
   17ff8: 41 04        	addi	s0, s0, 16
   17ffa: e3 53 f3 fe  	bge	t1, a5, 0x17fe0 <_svfprintf_r+0x1042>
   17ffe: 10 02        	addi	a2, sp, 256
   18000: e6 85        	mv	a1, s9
   18002: 4e 85        	mv	a0, s3
   18004: ef 20 a0 6a  	jal	0x1a6ae <__ssprint_r>
   18008: e3 11 05 92  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   1800c: c1 48        	addi	a7, zero, 16
   1800e: c1 34        	addiw	s1, s1, -16
   18010: 52 67        	ld	a4, 272(sp)
   18012: 83 27 81 10  	lw	a5, 264(sp)
   18016: 00 03        	addi	s0, sp, 384
   18018: 1d 43        	addi	t1, zero, 7
   1801a: e3 c6 98 fc  	blt	a7, s1, 0x17fe6 <_svfprintf_r+0x1048>
   1801e: 5e 86        	mv	a2, s7
   18020: a6 8b        	mv	s7, s1
   18022: c2 74        	ld	s1, 48(sp)
   18024: 9b 86 17 00  	addiw	a3, a5, 1
   18028: 5e 97        	add	a4, a4, s7
   1802a: 10 e0        	sd	a2, 0(s0)
   1802c: 23 34 74 01  	sd	s7, 8(s0)
   18030: 3a ea        	sd	a4, 272(sp)
   18032: 23 24 d1 10  	sw	a3, 264(sp)
   18036: 63 49 d3 72  	blt	t1, a3, 0x18768 <_svfprintf_r+0x17ca>
   1803a: 83 46 09 00  	lbu	a3, 0(s2)
   1803e: 41 04        	addi	s0, s0, 16
   18040: 36 9a        	add	s4, s4, a3
   18042: ad b7        	j	0x17fac <_svfprintf_r+0x100e>
   18044: 7d 19        	addi	s2, s2, -1
   18046: 7d 3d        	addiw	s10, s10, -1
   18048: cd bd        	j	0x17f3a <_svfprintf_r+0xf9c>
   1804a: 93 77 09 01  	andi	a5, s2, 16
   1804e: dd c7        	beqz	a5, 0x180fc <_svfprintf_r+0x115e>
   18050: 82 77        	ld	a5, 32(sp)
   18052: 83 bb 07 00  	ld	s7, 0(a5)
   18056: c1 b3        	j	0x17e16 <_svfprintf_r+0xe78>
   18058: 10 02        	addi	a2, sp, 256
   1805a: e6 85        	mv	a1, s9
   1805c: 4e 85        	mv	a0, s3
   1805e: ef 20 00 65  	jal	0x1a6ae <__ssprint_r>
   18062: e3 14 05 8c  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18066: 52 67        	ld	a4, 272(sp)
   18068: 00 03        	addi	s0, sp, 384
   1806a: c1 48        	addi	a7, zero, 16
   1806c: 1d 43        	addi	t1, zero, 7
   1806e: e5 b5        	j	0x17f56 <_svfprintf_r+0xfb8>
   18070: 83 26 81 10  	lw	a3, 264(sp)
   18074: f5 65        	lui	a1, 29
   18076: 93 85 85 79  	addi	a1, a1, 1944
   1807a: 85 26        	addiw	a3, a3, 1
   1807c: 0c e0        	sd	a1, 0(s0)
   1807e: 13 87 1b 00  	addi	a4, s7, 1
   18082: 85 45        	addi	a1, zero, 1
   18084: 0c e4        	sd	a1, 8(s0)
   18086: 23 24 d1 10  	sw	a3, 264(sp)
   1808a: 9b 85 06 00  	sext.w	a1, a3
   1808e: 3a ea        	sd	a4, 272(sp)
   18090: 9d 46        	addi	a3, zero, 7
   18092: 41 04        	addi	s0, s0, 16
   18094: 63 c9 b6 04  	blt	a3, a1, 0x180e6 <_svfprintf_r+0x1148>
   18098: 63 1f 06 14  	bnez	a2, 0x181f6 <_svfprintf_r+0x1258>
   1809c: e2 77        	ld	a5, 56(sp)
   1809e: 93 76 19 00  	andi	a3, s2, 1
   180a2: dd 8e        	or	a3, a3, a5
   180a4: 63 8c 06 90  	beqz	a3, 0x171bc <_svfprintf_r+0x21e>
   180a8: 03 26 81 10  	lw	a2, 264(sp)
   180ac: e6 67        	ld	a5, 88(sp)
   180ae: 86 76        	ld	a3, 96(sp)
   180b0: 3e 97        	add	a4, a4, a5
   180b2: 14 e0        	sd	a3, 0(s0)
   180b4: 9b 06 16 00  	addiw	a3, a2, 1
   180b8: 1c e4        	sd	a5, 8(s0)
   180ba: 3a ea        	sd	a4, 272(sp)
   180bc: 23 24 d1 10  	sw	a3, 264(sp)
   180c0: 1d 46        	addi	a2, zero, 7
   180c2: 63 49 d6 44  	blt	a2, a3, 0x18514 <_svfprintf_r+0x1576>
   180c6: 41 04        	addi	s0, s0, 16
   180c8: e2 77        	ld	a5, 56(sp)
   180ca: 1b 86 16 00  	addiw	a2, a3, 1
   180ce: 23 30 64 01  	sd	s6, 0(s0)
   180d2: 3e 97        	add	a4, a4, a5
   180d4: 1c e4        	sd	a5, 8(s0)
   180d6: 3a ea        	sd	a4, 272(sp)
   180d8: 23 24 c1 10  	sw	a2, 264(sp)
   180dc: 9d 47        	addi	a5, zero, 7
   180de: 63 de c7 8c  	bge	a5, a2, 0x171ba <_svfprintf_r+0x21c>
   180e2: 6f f0 0f b1  	j	0x173f2 <_svfprintf_r+0x454>
   180e6: 82 65        	ld	a1, 0(sp)
   180e8: 10 02        	addi	a2, sp, 256
   180ea: 4e 85        	mv	a0, s3
   180ec: ef 20 20 5c  	jal	0x1a6ae <__ssprint_r>
   180f0: e3 1d 05 82  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   180f4: 6e 46        	lw	a2, 216(sp)
   180f6: 52 67        	ld	a4, 272(sp)
   180f8: 00 03        	addi	s0, sp, 384
   180fa: 79 bf        	j	0x18098 <_svfprintf_r+0x10fa>
   180fc: 82 76        	ld	a3, 32(sp)
   180fe: 93 77 09 04  	andi	a5, s2, 64
   18102: 83 ab 06 00  	lw	s7, 0(a3)
   18106: 63 8f 07 18  	beqz	a5, 0x182a4 <_svfprintf_r+0x1306>
   1810a: c2 1b        	slli	s7, s7, 48
   1810c: 93 db 0b 03  	srli	s7, s7, 48
   18110: 19 b3        	j	0x17e16 <_svfprintf_r+0xe78>
   18112: 82 65        	ld	a1, 0(sp)
   18114: 10 02        	addi	a2, sp, 256
   18116: 4e 85        	mv	a0, s3
   18118: ef 20 60 59  	jal	0x1a6ae <__ssprint_r>
   1811c: e3 17 05 80  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18120: d2 6b        	ld	s7, 272(sp)
   18122: 00 03        	addi	s0, sp, 384
   18124: 6f f0 4f dc  	j	0x176e8 <_svfprintf_r+0x74a>
   18128: 9b 86 17 00  	addiw	a3, a5, 1
   1812c: 23 30 74 01  	sd	s7, 0(s0)
   18130: 23 34 94 01  	sd	s9, 8(s0)
   18134: 66 97        	add	a4, a4, s9
   18136: 3a ea        	sd	a4, 272(sp)
   18138: 23 24 d1 10  	sw	a3, 264(sp)
   1813c: 9d 47        	addi	a5, zero, 7
   1813e: 63 c4 d7 00  	blt	a5, a3, 0x18146 <_svfprintf_r+0x11a8>
   18142: 6f f0 8f 87  	j	0x171ba <_svfprintf_r+0x21c>
   18146: 6f f0 cf aa  	j	0x173f2 <_svfprintf_r+0x454>
   1814a: 10 02        	addi	a2, sp, 256
   1814c: e6 85        	mv	a1, s9
   1814e: 4e 85        	mv	a0, s3
   18150: ef 20 e0 55  	jal	0x1a6ae <__ssprint_r>
   18154: 63 1b 05 fc  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18158: 83 46 09 00  	lbu	a3, 0(s2)
   1815c: 52 67        	ld	a4, 272(sp)
   1815e: 00 03        	addi	s0, sp, 384
   18160: 1b 86 06 00  	sext.w	a2, a3
   18164: c1 48        	addi	a7, zero, 16
   18166: 1d 43        	addi	t1, zero, 7
   18168: 05 bd        	j	0x17f98 <_svfprintf_r+0xffa>
   1816a: 82 65        	ld	a1, 0(sp)
   1816c: 10 02        	addi	a2, sp, 256
   1816e: 4e 85        	mv	a0, s3
   18170: 1a e9        	sd	t1, 144(sp)
   18172: ef 20 c0 53  	jal	0x1a6ae <__ssprint_r>
   18176: 63 1a 05 fa  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   1817a: d2 6b        	ld	s7, 272(sp)
   1817c: 4a 63        	ld	t1, 144(sp)
   1817e: 00 03        	addi	s0, sp, 384
   18180: 6f f0 af 80  	j	0x1718a <_svfprintf_r+0x1ec>
   18184: 4e 75        	ld	a0, 240(sp)
   18186: ee 75        	ld	a1, 248(sp)
   18188: 01 46        	mv	a2, zero
   1818a: 81 46        	mv	a3, zero
   1818c: ef 30 b0 33  	jal	0x1bcc6 <__lttf2>
   18190: 63 43 05 6a  	bltz	a0, 0x18836 <_svfprintf_r+0x1898>
   18194: 03 47 f1 0c  	lbu	a4, 207(sp)
   18198: 93 07 70 04  	addi	a5, zero, 71
   1819c: 63 d5 a7 19  	bge	a5, s10, 0x18326 <_svfprintf_r+0x1388>
   181a0: f5 67        	lui	a5, 29
   181a2: 13 8b 87 74  	addi	s6, a5, 1864
   181a6: 13 79 f9 f7  	andi	s2, s2, -129
   181aa: 02 ec        	sd	zero, 24(sp)
   181ac: 82 e4        	sd	zero, 72(sp)
   181ae: 82 e0        	sd	zero, 64(sp)
   181b0: 02 f8        	sd	zero, 48(sp)
   181b2: 01 29        	sext.w	s2, s2
   181b4: 0d 4c        	addi	s8, zero, 3
   181b6: 8d 4d        	addi	s11, zero, 3
   181b8: 01 43        	mv	t1, zero
   181ba: 63 1e 07 aa  	bnez	a4, 0x17476 <_svfprintf_r+0x4d8>
   181be: 6f e0 9f f5  	j	0x17116 <_svfprintf_r+0x178>
   181c2: 99 47        	addi	a5, zero, 6
   181c4: 9a 86        	mv	a3, t1
   181c6: 63 ef 67 16  	bltu	a5, t1, 0x18344 <_svfprintf_r+0x13a6>
   181ca: 9b 8d 06 00  	sext.w	s11, a3
   181ce: f5 67        	lui	a5, 29
   181d0: 6e 8c        	mv	s8, s11
   181d2: 56 f0        	sd	s5, 32(sp)
   181d4: 13 8b 07 79  	addi	s6, a5, 1936
   181d8: 6f e0 5f f3  	j	0x1710c <_svfprintf_r+0x16e>
   181dc: 05 24        	addiw	s0, s0, 1
   181de: 6e 9d        	add	s10, s10, s11
   181e0: 23 b0 7c 01  	sd	s7, 0(s9)
   181e4: 23 b4 bc 01  	sd	s11, 8(s9)
   181e8: 6a ea        	sd	s10, 272(sp)
   181ea: 23 24 81 10  	sw	s0, 264(sp)
   181ee: 9d 47        	addi	a5, zero, 7
   181f0: 63 dd 87 9c  	bge	a5, s0, 0x173ca <_svfprintf_r+0x42c>
   181f4: e5 b2        	j	0x17bdc <_svfprintf_r+0xc3e>
   181f6: 83 25 81 10  	lw	a1, 264(sp)
   181fa: e6 67        	ld	a5, 88(sp)
   181fc: 86 76        	ld	a3, 96(sp)
   181fe: 41 04        	addi	s0, s0, 16
   18200: 3e 97        	add	a4, a4, a5
   18202: 23 38 d4 fe  	sd	a3, -16(s0)
   18206: 9b 86 15 00  	addiw	a3, a1, 1
   1820a: 23 3c f4 fe  	sd	a5, -8(s0)
   1820e: 3a ea        	sd	a4, 272(sp)
   18210: 23 24 d1 10  	sw	a3, 264(sp)
   18214: 9d 45        	addi	a1, zero, 7
   18216: 63 cf d5 2e  	blt	a1, a3, 0x18514 <_svfprintf_r+0x1576>
   1821a: e3 57 06 ea  	bgez	a2, 0x180c8 <_svfprintf_r+0x112a>
   1821e: 79 6f        	lui	t5, 30
   18220: c1 55        	addi	a1, zero, -16
   18222: bb 0c c0 40  	negw	s9, a2
   18226: 93 0b 0f 0b  	addi	s7, t5, 176
   1822a: 41 4d        	addi	s10, zero, 16
   1822c: 9d 4d        	addi	s11, zero, 7
   1822e: 63 46 b6 00  	blt	a2, a1, 0x1823a <_svfprintf_r+0x129c>
   18232: f9 ad        	j	0x18910 <_svfprintf_r+0x1972>
   18234: c1 3c        	addiw	s9, s9, -16
   18236: 63 5d 9d 6d  	bge	s10, s9, 0x18910 <_svfprintf_r+0x1972>
   1823a: 85 26        	addiw	a3, a3, 1
   1823c: 41 07        	addi	a4, a4, 16
   1823e: 23 30 74 01  	sd	s7, 0(s0)
   18242: 23 34 a4 01  	sd	s10, 8(s0)
   18246: 3a ea        	sd	a4, 272(sp)
   18248: 23 24 d1 10  	sw	a3, 264(sp)
   1824c: 41 04        	addi	s0, s0, 16
   1824e: e3 d3 dd fe  	bge	s11, a3, 0x18234 <_svfprintf_r+0x1296>
   18252: 82 65        	ld	a1, 0(sp)
   18254: 10 02        	addi	a2, sp, 256
   18256: 4e 85        	mv	a0, s3
   18258: ef 20 60 45  	jal	0x1a6ae <__ssprint_r>
   1825c: 63 17 05 ec  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18260: 52 67        	ld	a4, 272(sp)
   18262: 83 26 81 10  	lw	a3, 264(sp)
   18266: 00 03        	addi	s0, sp, 384
   18268: f1 b7        	j	0x18234 <_svfprintf_r+0x1296>
   1826a: 93 77 09 20  	andi	a5, s2, 512
   1826e: 63 88 07 26  	beqz	a5, 0x184de <_svfprintf_r+0x1540>
   18272: 93 fb fb 0f  	andi	s7, s7, 255
   18276: 6f f0 cf ba  	j	0x17622 <_svfprintf_r+0x684>
   1827a: 93 f7 0c 20  	andi	a5, s9, 512
   1827e: 63 89 07 24  	beqz	a5, 0x184d0 <_svfprintf_r+0x1532>
   18282: 93 fb fb 0f  	andi	s7, s7, 255
   18286: 3a f0        	sd	a4, 32(sp)
   18288: 85 47        	addi	a5, zero, 1
   1828a: 6f f0 8f ae  	j	0x17572 <_svfprintf_r+0x5d4>
   1828e: 93 77 09 20  	andi	a5, s2, 512
   18292: 63 84 07 26  	beqz	a5, 0x184fa <_svfprintf_r+0x155c>
   18296: 9b 9b 8b 01  	slliw	s7, s7, 24
   1829a: 9b db 8b 41  	sraiw	s7, s7, 24
   1829e: de 87        	mv	a5, s7
   182a0: 6f f0 0f b4  	j	0x175e0 <_svfprintf_r+0x642>
   182a4: 93 77 09 20  	andi	a5, s2, 512
   182a8: 63 85 07 24  	beqz	a5, 0x184f2 <_svfprintf_r+0x1554>
   182ac: 93 fb fb 0f  	andi	s7, s7, 255
   182b0: 9d b6        	j	0x17e16 <_svfprintf_r+0xe78>
   182b2: ee 77        	ld	a5, 248(sp)
   182b4: 63 d6 07 24  	bgez	a5, 0x18500 <_svfprintf_r+0x1562>
   182b8: 93 07 d0 02  	addi	a5, zero, 45
   182bc: a3 07 f1 0c  	sb	a5, 207(sp)
   182c0: 13 07 d0 02  	addi	a4, zero, 45
   182c4: 93 07 70 04  	addi	a5, zero, 71
   182c8: 63 d4 a7 5f  	bge	a5, s10, 0x188b0 <_svfprintf_r+0x1912>
   182cc: f5 67        	lui	a5, 29
   182ce: 13 8b 87 75  	addi	s6, a5, 1880
   182d2: d1 bd        	j	0x181a6 <_svfprintf_r+0x1208>
   182d4: 9b 05 16 00  	addiw	a1, a2, 1
   182d8: 6a 97        	add	a4, a4, s10
   182da: 23 30 74 01  	sd	s7, 0(s0)
   182de: 23 34 a4 01  	sd	s10, 8(s0)
   182e2: 3a ea        	sd	a4, 272(sp)
   182e4: 23 24 b1 10  	sw	a1, 264(sp)
   182e8: 9d 46        	addi	a3, zero, 7
   182ea: 63 c7 b6 24  	blt	a3, a1, 0x18538 <_svfprintf_r+0x159a>
   182ee: 41 04        	addi	s0, s0, 16
   182f0: ba 8b        	mv	s7, a4
   182f2: 6f f0 6f fe  	j	0x17ad8 <_svfprintf_r+0xb3a>
   182f6: 3e fc        	sd	a5, 56(sp)
   182f8: 8a 67        	ld	a5, 128(sp)
   182fa: aa 65        	ld	a1, 136(sp)
   182fc: 1a f8        	sd	t1, 48(sp)
   182fe: 33 0b fb 40  	sub	s6, s6, a5
   18302: 3e 86        	mv	a2, a5
   18304: 5a 85        	mv	a0, s6
   18306: 42 ec        	sd	a6, 24(sp)
   18308: ef e0 9f c0  	jal	0x16f10 <strncpy>
   1830c: 62 68        	ld	a6, 24(sp)
   1830e: 29 47        	addi	a4, zero, 10
   18310: 33 d7 eb 02  	divu	a4, s7, a4
   18314: 83 46 18 00  	lbu	a3, 1(a6)
   18318: 42 73        	ld	t1, 48(sp)
   1831a: e2 77        	ld	a5, 56(sp)
   1831c: b3 36 d0 00  	snez	a3, a3
   18320: 36 98        	add	a6, a6, a3
   18322: 81 46        	mv	a3, zero
   18324: 45 b4        	j	0x17dc4 <_svfprintf_r+0xe26>
   18326: f5 67        	lui	a5, 29
   18328: 13 8b 07 74  	addi	s6, a5, 1856
   1832c: ad bd        	j	0x181a6 <_svfprintf_r+0x1208>
   1832e: 82 65        	ld	a1, 0(sp)
   18330: 10 02        	addi	a2, sp, 256
   18332: 4e 85        	mv	a0, s3
   18334: ef 20 a0 37  	jal	0x1a6ae <__ssprint_r>
   18338: 63 19 05 de  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   1833c: d2 6b        	ld	s7, 272(sp)
   1833e: 00 03        	addi	s0, sp, 384
   18340: 6f f0 4f f8  	j	0x17ac4 <_svfprintf_r+0xb26>
   18344: 99 46        	addi	a3, zero, 6
   18346: 51 b5        	j	0x181ca <_svfprintf_r+0x122c>
   18348: b3 8c 6a 41  	sub	s9, s5, s6
   1834c: 3b 86 d7 40  	subw	a2, a5, a3
   18350: 9b 87 0c 00  	sext.w	a5, s9
   18354: 5e 87        	mv	a4, s7
   18356: 63 53 f6 00  	bge	a2, a5, 0x1835c <_svfprintf_r+0x13be>
   1835a: b2 8c        	mv	s9, a2
   1835c: 81 2c        	sext.w	s9, s9
   1835e: 6f f0 4f ff  	j	0x17b52 <_svfprintf_r+0xbb4>
   18362: 1a 89        	mv	s2, t1
   18364: 89 46        	addi	a3, zero, 2
   18366: 4a 87        	mv	a4, s2
   18368: 93 08 81 0e  	addi	a7, sp, 232
   1836c: 13 08 c1 0d  	addi	a6, sp, 220
   18370: bc 09        	addi	a5, sp, 216
   18372: e2 85        	mv	a1, s8
   18374: 66 86        	mv	a2, s9
   18376: 4e 85        	mv	a0, s3
   18378: 1a f8        	sd	t1, 48(sp)
   1837a: ef b0 3f fe  	jal	0x1435c <_ldtoa_r>
   1837e: 13 07 70 06  	addi	a4, zero, 103
   18382: 42 73        	ld	t1, 48(sp)
   18384: 2a 8b        	mv	s6, a0
   18386: 63 08 ed 3c  	beq	s10, a4, 0x18756 <_svfprintf_r+0x17b8>
   1838a: 13 07 70 04  	addi	a4, zero, 71
   1838e: 63 04 ed 3c  	beq	s10, a4, 0x18756 <_svfprintf_r+0x17b8>
   18392: 93 7d fd fd  	andi	s11, s10, -33
   18396: 13 07 60 04  	addi	a4, zero, 70
   1839a: b3 0a 25 01  	add	s5, a0, s2
   1839e: 63 81 ed 4c  	beq	s11, a4, 0x18860 <_svfprintf_r+0x18c2>
   183a2: 62 85        	mv	a0, s8
   183a4: 01 46        	mv	a2, zero
   183a6: 81 46        	mv	a3, zero
   183a8: e6 85        	mv	a1, s9
   183aa: 1a f8        	sd	t1, 48(sp)
   183ac: ef 30 00 7c  	jal	0x1bb6c <__netf2>
   183b0: 42 73        	ld	t1, 48(sp)
   183b2: 56 8c        	mv	s8, s5
   183b4: 11 cd        	beqz	a0, 0x183d0 <_svfprintf_r+0x1432>
   183b6: 2e 7c        	ld	s8, 232(sp)
   183b8: 63 7c 5c 01  	bgeu	s8, s5, 0x183d0 <_svfprintf_r+0x1432>
   183bc: 93 06 00 03  	addi	a3, zero, 48
   183c0: 93 07 1c 00  	addi	a5, s8, 1
   183c4: be f5        	sd	a5, 232(sp)
   183c6: 23 00 dc 00  	sb	a3, 0(s8)
   183ca: 2e 7c        	ld	s8, 232(sp)
   183cc: e3 6a 5c ff  	bltu	s8, s5, 0x183c0 <_svfprintf_r+0x1422>
   183d0: bb 07 6c 41  	subw	a5, s8, s6
   183d4: 3e fc        	sd	a5, 56(sp)
   183d6: 6e 46        	lw	a2, 216(sp)
   183d8: 13 07 70 04  	addi	a4, zero, 71
   183dc: 32 f8        	sd	a2, 48(sp)
   183de: 63 8f ed 2c  	beq	s11, a4, 0x186bc <_svfprintf_r+0x171e>
   183e2: 13 07 60 04  	addi	a4, zero, 70
   183e6: 63 87 ed 50  	beq	s11, a4, 0x188f4 <_svfprintf_r+0x1956>
   183ea: 1b 07 f6 ff  	addiw	a4, a2, -1
   183ee: ba cd        	sw	a4, 216(sp)
   183f0: 93 06 10 06  	addi	a3, zero, 97
   183f4: 63 0b dd 66  	beq	s10, a3, 0x18a6a <_svfprintf_r+0x1acc>
   183f8: 93 05 10 04  	addi	a1, zero, 65
   183fc: 93 06 00 05  	addi	a3, zero, 80
   18400: 63 18 bd 2c  	bne	s10, a1, 0x186d0 <_svfprintf_r+0x1732>
   18404: 85 45        	addi	a1, zero, 1
   18406: 23 00 d1 0e  	sb	a3, 224(sp)
   1840a: 93 06 b0 02  	addi	a3, zero, 43
   1840e: 63 56 07 00  	bgez	a4, 0x1841a <_svfprintf_r+0x147c>
   18412: 05 47        	addi	a4, zero, 1
   18414: 11 9f        	subw	a4, a4, a2
   18416: 93 06 d0 02  	addi	a3, zero, 45
   1841a: a3 00 d1 0e  	sb	a3, 225(sp)
   1841e: a5 46        	addi	a3, zero, 9
   18420: 63 d2 e6 52  	bge	a3, a4, 0x18944 <_svfprintf_r+0x19a6>
   18424: 13 08 f1 0e  	addi	a6, sp, 239
   18428: c2 88        	mv	a7, a6
   1842a: 29 43        	addi	t1, zero, 10
   1842c: 13 0f 30 06  	addi	t5, zero, 99
   18430: bb 67 67 02  	remw	a5, a4, t1
   18434: 46 86        	mv	a2, a7
   18436: ba 86        	mv	a3, a4
   18438: fd 18        	addi	a7, a7, -1
   1843a: 9b 87 07 03  	addiw	a5, a5, 48
   1843e: a3 0f f6 fe  	sb	a5, -1(a2)
   18442: 3b 47 67 02  	divw	a4, a4, t1
   18446: e3 45 df fe  	blt	t5, a3, 0x18430 <_svfprintf_r+0x1492>
   1844a: 1b 05 07 03  	addiw	a0, a4, 48
   1844e: a3 8f a8 fe  	sb	a0, -1(a7)
   18452: 13 07 e6 ff  	addi	a4, a2, -2
   18456: 63 74 07 67  	bgeu	a4, a6, 0x18abe <_svfprintf_r+0x1b20>
   1845a: 93 06 21 0e  	addi	a3, sp, 226
   1845e: 83 47 07 00  	lbu	a5, 0(a4)
   18462: 05 07        	addi	a4, a4, 1
   18464: 85 06        	addi	a3, a3, 1
   18466: a3 8f f6 fe  	sb	a5, -1(a3)
   1846a: e3 1a 07 ff  	bne	a4, a6, 0x1845e <_svfprintf_r+0x14c0>
   1846e: 13 07 11 0f  	addi	a4, sp, 241
   18472: 93 07 21 0e  	addi	a5, sp, 226
   18476: 11 8f        	sub	a4, a4, a2
   18478: 3e 97        	add	a4, a4, a5
   1847a: 94 11        	addi	a3, sp, 224
   1847c: bb 07 d7 40  	subw	a5, a4, a3
   18480: be f8        	sd	a5, 112(sp)
   18482: 46 77        	ld	a4, 112(sp)
   18484: e2 77        	ld	a5, 56(sp)
   18486: 05 46        	addi	a2, zero, 1
   18488: bb 0d f7 00  	addw	s11, a4, a5
   1848c: 6e 87        	mv	a4, s11
   1848e: 63 5b f6 4e  	bge	a2, a5, 0x18984 <_svfprintf_r+0x19e6>
   18492: e6 67        	ld	a5, 88(sp)
   18494: bb 0d f7 00  	addw	s11, a4, a5
   18498: 86 67        	ld	a5, 64(sp)
   1849a: 13 c8 fd ff  	not	a6, s11
   1849e: 13 58 f8 43  	srai	a6, a6, 63
   184a2: 13 f9 f7 bf  	andi	s2, a5, -1025
   184a6: 01 29        	sext.w	s2, s2
   184a8: 33 f8 0d 01  	and	a6, s11, a6
   184ac: 13 69 09 10  	ori	s2, s2, 256
   184b0: 1b 0c 08 00  	sext.w	s8, a6
   184b4: 82 e4        	sd	zero, 72(sp)
   184b6: 82 e0        	sd	zero, 64(sp)
   184b8: 02 f8        	sd	zero, 48(sp)
   184ba: ea 67        	ld	a5, 152(sp)
   184bc: 63 86 07 24  	beqz	a5, 0x18708 <_svfprintf_r+0x176a>
   184c0: 13 07 d0 02  	addi	a4, zero, 45
   184c4: a3 07 e1 0c  	sb	a4, 207(sp)
   184c8: 01 43        	mv	t1, zero
   184ca: 05 2c        	addiw	s8, s8, 1
   184cc: 6f e0 bf c4  	j	0x17116 <_svfprintf_r+0x178>
   184d0: 82 1b        	slli	s7, s7, 32
   184d2: 93 db 0b 02  	srli	s7, s7, 32
   184d6: 3a f0        	sd	a4, 32(sp)
   184d8: 85 47        	addi	a5, zero, 1
   184da: 6f f0 8f 89  	j	0x17572 <_svfprintf_r+0x5d4>
   184de: 82 1b        	slli	s7, s7, 32
   184e0: 93 db 0b 02  	srli	s7, s7, 32
   184e4: 6f f0 ef 93  	j	0x17622 <_svfprintf_r+0x684>
   184e8: c2 67        	ld	a5, 16(sp)
   184ea: 23 10 f7 00  	sh	a5, 0(a4)
   184ee: 6f f0 cf 95  	j	0x1764a <_svfprintf_r+0x6ac>
   184f2: 82 1b        	slli	s7, s7, 32
   184f4: 93 db 0b 02  	srli	s7, s7, 32
   184f8: 39 ba        	j	0x17e16 <_svfprintf_r+0xe78>
   184fa: de 87        	mv	a5, s7
   184fc: 6f f0 4f 8e  	j	0x175e0 <_svfprintf_r+0x642>
   18500: 03 47 f1 0c  	lbu	a4, 207(sp)
   18504: c1 b3        	j	0x182c4 <_svfprintf_r+0x1326>
   18506: 82 65        	ld	a1, 0(sp)
   18508: 46 65        	ld	a0, 80(sp)
   1850a: 10 02        	addi	a2, sp, 256
   1850c: ef 20 20 1a  	jal	0x1a6ae <__ssprint_r>
   18510: 6f e0 3f cf  	j	0x17202 <_svfprintf_r+0x264>
   18514: 82 65        	ld	a1, 0(sp)
   18516: 10 02        	addi	a2, sp, 256
   18518: 4e 85        	mv	a0, s3
   1851a: ef 20 40 19  	jal	0x1a6ae <__ssprint_r>
   1851e: 63 16 05 c0  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18522: 6e 46        	lw	a2, 216(sp)
   18524: 52 67        	ld	a4, 272(sp)
   18526: 83 26 81 10  	lw	a3, 264(sp)
   1852a: 00 03        	addi	s0, sp, 384
   1852c: e3 5e 06 b8  	bgez	a2, 0x180c8 <_svfprintf_r+0x112a>
   18530: fd b1        	j	0x1821e <_svfprintf_r+0x1280>
   18532: 19 43        	addi	t1, zero, 6
   18534: 6f e0 9f db  	j	0x172ec <_svfprintf_r+0x34e>
   18538: 82 65        	ld	a1, 0(sp)
   1853a: 10 02        	addi	a2, sp, 256
   1853c: 4e 85        	mv	a0, s3
   1853e: ef 20 00 17  	jal	0x1a6ae <__ssprint_r>
   18542: 63 14 05 be  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18546: d2 6b        	ld	s7, 272(sp)
   18548: 00 03        	addi	s0, sp, 384
   1854a: 6f f0 ef d8  	j	0x17ad8 <_svfprintf_r+0xb3a>
   1854e: 1b 09 13 00  	addiw	s2, t1, 1
   18552: 89 46        	addi	a3, zero, 2
   18554: 09 bd        	j	0x18366 <_svfprintf_r+0x13c8>
   18556: 1a 89        	mv	s2, t1
   18558: 8d 46        	addi	a3, zero, 3
   1855a: 31 b5        	j	0x18366 <_svfprintf_r+0x13c8>
   1855c: e6 85        	mv	a1, s9
   1855e: 62 85        	mv	a0, s8
   18560: 1a f8        	sd	t1, 48(sp)
   18562: ef 40 90 12  	jal	0x1ce8a <__trunctfdf2>
   18566: a8 09        	addi	a0, sp, 216
   18568: ef e0 2f da  	jal	0x16b0a <frexp>
   1856c: ef 40 b0 05  	jal	0x1cdc6 <__extenddftf2>
   18570: 79 67        	lui	a4, 30
   18572: 83 36 87 9c  	ld	a3, -1592(a4)
   18576: 01 46        	mv	a2, zero
   18578: ef 30 f0 00  	jal	0x1bd86 <__multf3>
   1857c: 01 46        	mv	a2, zero
   1857e: 81 46        	mv	a3, zero
   18580: aa 8d        	mv	s11, a0
   18582: 2e 89        	mv	s2, a1
   18584: ef 30 80 5e  	jal	0x1bb6c <__netf2>
   18588: 42 73        	ld	t1, 48(sp)
   1858a: 19 e1        	bnez	a0, 0x18590 <_svfprintf_r+0x15f2>
   1858c: 05 47        	addi	a4, zero, 1
   1858e: ba cd        	sw	a4, 216(sp)
   18590: 75 67        	lui	a4, 29
   18592: 93 07 07 76  	addi	a5, a4, 1888
   18596: 3e fc        	sd	a5, 56(sp)
   18598: f9 66        	lui	a3, 30
   1859a: 83 b7 86 9d  	ld	a5, -1576(a3)
   1859e: 13 17 03 02  	slli	a4, t1, 32
   185a2: 01 93        	srli	a4, a4, 32
   185a4: 05 07        	addi	a4, a4, 1
   185a6: 3e f8        	sd	a5, 48(sp)
   185a8: 1b 06 03 00  	sext.w	a2, t1
   185ac: b3 07 eb 00  	add	a5, s6, a4
   185b0: a6 e4        	sd	s1, 72(sp)
   185b2: 52 f1        	sd	s4, 160(sp)
   185b4: 22 f9        	sd	s0, 176(sp)
   185b6: e2 74        	ld	s1, 56(sp)
   185b8: 1a f5        	sd	t1, 168(sp)
   185ba: 5a 8a        	mv	s4, s6
   185bc: 32 fd        	sd	a2, 184(sp)
   185be: 3e 84        	mv	s0, a5
   185c0: ce 8b        	mv	s7, s3
   185c2: da 8c        	mv	s9, s6
   185c4: 39 a0        	j	0x185d2 <_svfprintf_r+0x1634>
   185c6: 01 46        	mv	a2, zero
   185c8: 81 46        	mv	a3, zero
   185ca: ef 30 20 5a  	jal	0x1bb6c <__netf2>
   185ce: 63 01 05 3c  	beqz	a0, 0x18990 <_svfprintf_r+0x19f2>
   185d2: c2 76        	ld	a3, 48(sp)
   185d4: 01 46        	mv	a2, zero
   185d6: 6e 85        	mv	a0, s11
   185d8: ca 85        	mv	a1, s2
   185da: ef 30 c0 7a  	jal	0x1bd86 <__multf3>
   185de: 2e 89        	mv	s2, a1
   185e0: aa 89        	mv	s3, a0
   185e2: ef 40 e0 6f  	jal	0x1cce0 <__fixtfsi>
   185e6: 1b 0c 05 00  	sext.w	s8, a0
   185ea: 62 85        	mv	a0, s8
   185ec: ef 40 40 78  	jal	0x1cd70 <__floatsitf>
   185f0: ae 86        	mv	a3, a1
   185f2: 2a 86        	mv	a2, a0
   185f4: ca 85        	mv	a1, s2
   185f6: 4e 85        	mv	a0, s3
   185f8: ef 30 10 5b  	jal	0x1c3a8 <__subtf3>
   185fc: 33 87 84 01  	add	a4, s1, s8
   18600: 83 46 07 00  	lbu	a3, 0(a4)
   18604: d2 89        	mv	s3, s4
   18606: 05 0a        	addi	s4, s4, 1
   18608: a3 0f da fe  	sb	a3, -1(s4)
   1860c: aa 8a        	mv	s5, a0
   1860e: aa 8d        	mv	s11, a0
   18610: 2e 89        	mv	s2, a1
   18612: e3 1a 8a fa  	bne	s4, s0, 0x185c6 <_svfprintf_r+0x1628>
   18616: d2 8a        	mv	s5, s4
   18618: a6 64        	ld	s1, 72(sp)
   1861a: 2a 73        	ld	t1, 168(sp)
   1861c: 4a 74        	ld	s0, 176(sp)
   1861e: 0a 7a        	ld	s4, 160(sp)
   18620: 4e f8        	sd	s3, 48(sp)
   18622: 66 8b        	mv	s6, s9
   18624: de 89        	mv	s3, s7
   18626: ae 8c        	mv	s9, a1
   18628: aa 8b        	mv	s7, a0
   1862a: fd 5d        	addi	s11, zero, -1
   1862c: f9 66        	lui	a3, 30
   1862e: 03 b9 86 9e  	ld	s2, -1560(a3)
   18632: 01 46        	mv	a2, zero
   18634: 5e 85        	mv	a0, s7
   18636: ca 86        	mv	a3, s2
   18638: e6 85        	mv	a1, s9
   1863a: 9a e4        	sd	t1, 72(sp)
   1863c: ef 30 a0 5c  	jal	0x1bc06 <__gttf2>
   18640: 26 63        	ld	t1, 72(sp)
   18642: 63 4e a0 00  	bgtz	a0, 0x1865e <_svfprintf_r+0x16c0>
   18646: 01 46        	mv	a2, zero
   18648: ca 86        	mv	a3, s2
   1864a: 5e 85        	mv	a0, s7
   1864c: e6 85        	mv	a1, s9
   1864e: ef 30 e0 51  	jal	0x1bb6c <__netf2>
   18652: 26 63        	ld	t1, 72(sp)
   18654: 51 e1        	bnez	a0, 0x186d8 <_svfprintf_r+0x173a>
   18656: 13 7c 1c 00  	andi	s8, s8, 1
   1865a: 63 0f 0c 06  	beqz	s8, 0x186d8 <_svfprintf_r+0x173a>
   1865e: c2 77        	ld	a5, 48(sp)
   18660: d6 86        	mv	a3, s5
   18662: be f5        	sd	a5, 232(sp)
   18664: e2 77        	ld	a5, 56(sp)
   18666: 03 c6 fa ff  	lbu	a2, -1(s5)
   1866a: 83 c5 f7 00  	lbu	a1, 15(a5)
   1866e: 63 9e c5 00  	bne	a1, a2, 0x1868a <_svfprintf_r+0x16ec>
   18672: 13 05 00 03  	addi	a0, zero, 48
   18676: a3 8f a6 fe  	sb	a0, -1(a3)
   1867a: ae 76        	ld	a3, 232(sp)
   1867c: 93 87 f6 ff  	addi	a5, a3, -1
   18680: be f5        	sd	a5, 232(sp)
   18682: 03 c6 f6 ff  	lbu	a2, -1(a3)
   18686: e3 88 c5 fe  	beq	a1, a2, 0x18676 <_svfprintf_r+0x16d8>
   1868a: 9b 05 16 00  	addiw	a1, a2, 1
   1868e: 13 05 90 03  	addi	a0, zero, 57
   18692: 93 f5 f5 0f  	andi	a1, a1, 255
   18696: 63 0a a6 00  	beq	a2, a0, 0x186aa <_svfprintf_r+0x170c>
   1869a: a3 8f b6 fe  	sb	a1, -1(a3)
   1869e: bb 87 6a 41  	subw	a5, s5, s6
   186a2: 3e fc        	sd	a5, 56(sp)
   186a4: 93 7d fd fd  	andi	s11, s10, -33
   186a8: 3d b3        	j	0x183d6 <_svfprintf_r+0x1438>
   186aa: e2 77        	ld	a5, 56(sp)
   186ac: 83 c5 a7 00  	lbu	a1, 10(a5)
   186b0: ed b7        	j	0x1869a <_svfprintf_r+0x16fc>
   186b2: 2e 7c        	ld	s8, 232(sp)
   186b4: 6e 46        	lw	a2, 216(sp)
   186b6: bb 07 6c 41  	subw	a5, s8, s6
   186ba: 3e fc        	sd	a5, 56(sp)
   186bc: 75 57        	addi	a4, zero, -3
   186be: 32 f8        	sd	a2, 48(sp)
   186c0: 63 44 e6 00  	blt	a2, a4, 0x186c8 <_svfprintf_r+0x172a>
   186c4: 63 5a c3 04  	bge	t1, a2, 0x18718 <_svfprintf_r+0x177a>
   186c8: 1b 07 f6 ff  	addiw	a4, a2, -1
   186cc: 79 3d        	addiw	s10, s10, -2
   186ce: ba cd        	sw	a4, 216(sp)
   186d0: 93 76 fd 0f  	andi	a3, s10, 255
   186d4: 81 45        	mv	a1, zero
   186d6: 05 bb        	j	0x18406 <_svfprintf_r+0x1468>
   186d8: e3 c3 0d fc  	bltz	s11, 0x1869e <_svfprintf_r+0x1700>
   186dc: 85 2d        	addiw	s11, s11, 1
   186de: d7 75 00 04  	<unknown>
   186e2: 93 96 0d 02  	slli	a3, s11, 32
   186e6: 93 05 00 03  	addi	a1, zero, 48
   186ea: 81 92        	srli	a3, a3, 32
   186ec: 56 86        	mv	a2, s5
   186ee: 73 25 20 c2  	csrr	a0, vlenb
   186f2: 57 cc 05 5e  	<unknown>
   186f6: d7 f7 06 04  	<unknown>
   186fa: 27 0c 06 02  	<unknown>
   186fe: 9d 8e        	sub	a3, a3, a5
   18700: 2a 96        	add	a2, a2, a0
   18702: f5 fa        	bnez	a3, 0x186f6 <_svfprintf_r+0x1758>
   18704: ee 9a        	add	s5, s5, s11
   18706: 61 bf        	j	0x1869e <_svfprintf_r+0x1700>
   18708: 03 47 f1 0c  	lbu	a4, 207(sp)
   1870c: 01 43        	mv	t1, zero
   1870e: 19 c3        	beqz	a4, 0x18714 <_svfprintf_r+0x1776>
   18710: 6f e0 7f d6  	j	0x17476 <_svfprintf_r+0x4d8>
   18714: 6f e0 3f a0  	j	0x17116 <_svfprintf_r+0x178>
   18718: e2 77        	ld	a5, 56(sp)
   1871a: 63 45 f6 1a  	blt	a2, a5, 0x188c4 <_svfprintf_r+0x1926>
   1871e: 86 67        	ld	a5, 64(sp)
   18720: b2 8d        	mv	s11, a2
   18722: 13 f7 17 00  	andi	a4, a5, 1
   18726: 01 c7        	beqz	a4, 0x1872e <_svfprintf_r+0x1790>
   18728: e6 67        	ld	a5, 88(sp)
   1872a: bb 8d c7 00  	addw	s11, a5, a2
   1872e: 86 67        	ld	a5, 64(sp)
   18730: 13 f7 07 40  	andi	a4, a5, 1024
   18734: 19 c3        	beqz	a4, 0x1873a <_svfprintf_r+0x179c>
   18736: 63 41 c0 2a  	bgtz	a2, 0x189d8 <_svfprintf_r+0x1a3a>
   1873a: 13 c8 fd ff  	not	a6, s11
   1873e: 13 58 f8 43  	srai	a6, a6, 63
   18742: 4a 69        	ld	s2, 144(sp)
   18744: 33 f8 0d 01  	and	a6, s11, a6
   18748: 1b 0c 08 00  	sext.w	s8, a6
   1874c: 13 0d 70 06  	addi	s10, zero, 103
   18750: 82 e4        	sd	zero, 72(sp)
   18752: 82 e0        	sd	zero, 64(sp)
   18754: 9d b3        	j	0x184ba <_svfprintf_r+0x151c>
   18756: 86 67        	ld	a5, 64(sp)
   18758: 13 f7 17 00  	andi	a4, a5, 1
   1875c: 39 db        	beqz	a4, 0x186b2 <_svfprintf_r+0x1714>
   1875e: b3 0a 2b 01  	add	s5, s6, s2
   18762: 93 0d 70 04  	addi	s11, zero, 71
   18766: 35 b9        	j	0x183a2 <_svfprintf_r+0x1404>
   18768: 10 02        	addi	a2, sp, 256
   1876a: e6 85        	mv	a1, s9
   1876c: 4e 85        	mv	a0, s3
   1876e: ef 10 10 74  	jal	0x1a6ae <__ssprint_r>
   18772: 63 1c 05 9a  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   18776: 83 46 09 00  	lbu	a3, 0(s2)
   1877a: 52 67        	ld	a4, 272(sp)
   1877c: 00 03        	addi	s0, sp, 384
   1877e: c1 48        	addi	a7, zero, 16
   18780: 1d 43        	addi	t1, zero, 7
   18782: 36 9a        	add	s4, s4, a3
   18784: 6f f0 9f 82  	j	0x17fac <_svfprintf_r+0x100e>
   18788: 13 07 80 07  	addi	a4, zero, 120
   1878c: 93 67 29 00  	ori	a5, s2, 2
   18790: 93 06 00 03  	addi	a3, zero, 48
   18794: 81 27        	sext.w	a5, a5
   18796: a3 08 e1 0c  	sb	a4, 209(sp)
   1879a: 23 08 d1 0c  	sb	a3, 208(sp)
   1879e: 13 07 30 06  	addi	a4, zero, 99
   187a2: be e0        	sd	a5, 64(sp)
   187a4: 02 ec        	sd	zero, 24(sp)
   187a6: 13 0b 81 11  	addi	s6, sp, 280
   187aa: 63 41 67 1c  	blt	a4, t1, 0x1896c <_svfprintf_r+0x19ce>
   187ae: 13 69 29 10  	ori	s2, s2, 258
   187b2: ee 7c        	ld	s9, 248(sp)
   187b4: 9b 07 09 00  	sext.w	a5, s2
   187b8: 3e e9        	sd	a5, 144(sp)
   187ba: 02 ed        	sd	zero, 152(sp)
   187bc: 4e 7c        	ld	s8, 240(sp)
   187be: 63 c8 0c 04  	bltz	s9, 0x1880e <_svfprintf_r+0x1870>
   187c2: 13 07 10 06  	addi	a4, zero, 97
   187c6: 63 04 ed 00  	beq	s10, a4, 0x187ce <_svfprintf_r+0x1830>
   187ca: 6f e0 ff b3  	j	0x17308 <_svfprintf_r+0x36a>
   187ce: e6 85        	mv	a1, s9
   187d0: 62 85        	mv	a0, s8
   187d2: 1a f8        	sd	t1, 48(sp)
   187d4: ef 40 60 6b  	jal	0x1ce8a <__trunctfdf2>
   187d8: a8 09        	addi	a0, sp, 216
   187da: ef e0 0f b3  	jal	0x16b0a <frexp>
   187de: ef 40 80 5e  	jal	0x1cdc6 <__extenddftf2>
   187e2: 79 67        	lui	a4, 30
   187e4: 83 36 87 9c  	ld	a3, -1592(a4)
   187e8: 01 46        	mv	a2, zero
   187ea: ef 30 c0 59  	jal	0x1bd86 <__multf3>
   187ee: 01 46        	mv	a2, zero
   187f0: 81 46        	mv	a3, zero
   187f2: aa 8d        	mv	s11, a0
   187f4: 2e 89        	mv	s2, a1
   187f6: ef 30 60 37  	jal	0x1bb6c <__netf2>
   187fa: 42 73        	ld	t1, 48(sp)
   187fc: 19 e1        	bnez	a0, 0x18802 <_svfprintf_r+0x1864>
   187fe: 05 47        	addi	a4, zero, 1
   18800: ba cd        	sw	a4, 216(sp)
   18802: 75 67        	lui	a4, 29
   18804: 93 07 87 77  	addi	a5, a4, 1912
   18808: 3e fc        	sd	a5, 56(sp)
   1880a: 79 b3        	j	0x18598 <_svfprintf_r+0x15fa>
   1880c: 02 ec        	sd	zero, 24(sp)
   1880e: 7d 57        	addi	a4, zero, -1
   18810: 7e 17        	slli	a4, a4, 63
   18812: 93 07 d0 02  	addi	a5, zero, 45
   18816: b3 cc ec 00  	xor	s9, s9, a4
   1881a: 3e ed        	sd	a5, 152(sp)
   1881c: 5d b7        	j	0x187c2 <_svfprintf_r+0x1824>
   1881e: 82 65        	ld	a1, 0(sp)
   18820: 10 02        	addi	a2, sp, 256
   18822: 4e 85        	mv	a0, s3
   18824: ef 10 b0 68  	jal	0x1a6ae <__ssprint_r>
   18828: 63 11 05 90  	bnez	a0, 0x1792a <_svfprintf_r+0x98c>
   1882c: ee 46        	lw	a3, 216(sp)
   1882e: 52 67        	ld	a4, 272(sp)
   18830: 00 03        	addi	s0, sp, 384
   18832: 6f f0 8f ae  	j	0x17b1a <_svfprintf_r+0xb7c>
   18836: 93 07 d0 02  	addi	a5, zero, 45
   1883a: a3 07 f1 0c  	sb	a5, 207(sp)
   1883e: 13 07 d0 02  	addi	a4, zero, 45
   18842: 99 ba        	j	0x18198 <_svfprintf_r+0x11fa>
   18844: 03 47 f1 0c  	lbu	a4, 207(sp)
   18848: 56 f0        	sd	s5, 32(sp)
   1884a: 82 e4        	sd	zero, 72(sp)
   1884c: 82 e0        	sd	zero, 64(sp)
   1884e: 02 f8        	sd	zero, 48(sp)
   18850: 1a 8c        	mv	s8, t1
   18852: 9a 8d        	mv	s11, t1
   18854: 01 43        	mv	t1, zero
   18856: 19 c3        	beqz	a4, 0x1885c <_svfprintf_r+0x18be>
   18858: 6f e0 ff c1  	j	0x17476 <_svfprintf_r+0x4d8>
   1885c: 6f e0 bf 8b  	j	0x17116 <_svfprintf_r+0x178>
   18860: 83 46 05 00  	lbu	a3, 0(a0)
   18864: 13 07 00 03  	addi	a4, zero, 48
   18868: 63 85 e6 00  	beq	a3, a4, 0x18872 <_svfprintf_r+0x18d4>
   1886c: ee 46        	lw	a3, 216(sp)
   1886e: b6 9a        	add	s5, s5, a3
   18870: 0d be        	j	0x183a2 <_svfprintf_r+0x1404>
   18872: 01 46        	mv	a2, zero
   18874: 81 46        	mv	a3, zero
   18876: 62 85        	mv	a0, s8
   18878: e6 85        	mv	a1, s9
   1887a: 1a f8        	sd	t1, 48(sp)
   1887c: ef 30 00 2f  	jal	0x1bb6c <__netf2>
   18880: 42 73        	ld	t1, 48(sp)
   18882: 6d d5        	beqz	a0, 0x1886c <_svfprintf_r+0x18ce>
   18884: 05 47        	addi	a4, zero, 1
   18886: bb 06 27 41  	subw	a3, a4, s2
   1888a: b6 cd        	sw	a3, 216(sp)
   1888c: b6 9a        	add	s5, s5, a3
   1888e: 11 be        	j	0x183a2 <_svfprintf_r+0x1404>
   18890: 82 65        	ld	a1, 0(sp)
   18892: 10 02        	addi	a2, sp, 256
   18894: 4e 85        	mv	a0, s3
   18896: ef 10 90 61  	jal	0x1a6ae <__ssprint_r>
   1889a: 19 c1        	beqz	a0, 0x188a0 <_svfprintf_r+0x1902>
   1889c: 6f f0 ef 88  	j	0x1792a <_svfprintf_r+0x98c>
   188a0: 6e 46        	lw	a2, 216(sp)
   188a2: e2 77        	ld	a5, 56(sp)
   188a4: 52 67        	ld	a4, 272(sp)
   188a6: 00 03        	addi	s0, sp, 384
   188a8: 3b 86 c7 40  	subw	a2, a5, a2
   188ac: 6f f0 6f aa  	j	0x17b52 <_svfprintf_r+0xbb4>
   188b0: f5 67        	lui	a5, 29
   188b2: 13 8b 07 75  	addi	s6, a5, 1872
   188b6: c5 b8        	j	0x181a6 <_svfprintf_r+0x1208>
   188b8: 13 07 80 05  	addi	a4, zero, 88
   188bc: c1 bd        	j	0x1878c <_svfprintf_r+0x17ee>
   188be: ca 8c        	mv	s9, s2
   188c0: 6f f0 ef 91  	j	0x179de <_svfprintf_r+0xa40>
   188c4: e6 67        	ld	a5, 88(sp)
   188c6: 62 77        	ld	a4, 56(sp)
   188c8: 13 0d 70 06  	addi	s10, zero, 103
   188cc: bb 8d e7 00  	addw	s11, a5, a4
   188d0: c2 77        	ld	a5, 48(sp)
   188d2: 63 57 f0 16  	blez	a5, 0x18a40 <_svfprintf_r+0x1aa2>
   188d6: 86 67        	ld	a5, 64(sp)
   188d8: 13 f7 07 40  	andi	a4, a5, 1024
   188dc: 63 10 07 10  	bnez	a4, 0x189dc <_svfprintf_r+0x1a3e>
   188e0: 13 c8 fd ff  	not	a6, s11
   188e4: 13 58 f8 43  	srai	a6, a6, 63
   188e8: 33 f8 0d 01  	and	a6, s11, a6
   188ec: 4a 69        	ld	s2, 144(sp)
   188ee: 1b 0c 08 00  	sext.w	s8, a6
   188f2: b9 bd        	j	0x18750 <_svfprintf_r+0x17b2>
   188f4: 86 67        	ld	a5, 64(sp)
   188f6: 13 f7 17 00  	andi	a4, a5, 1
   188fa: c2 77        	ld	a5, 48(sp)
   188fc: 33 67 e3 00  	or	a4, t1, a4
   18900: 63 57 f0 18  	blez	a5, 0x18a8e <_svfprintf_r+0x1af0>
   18904: 63 1b 07 14  	bnez	a4, 0x18a5a <_svfprintf_r+0x1abc>
   18908: c2 7d        	ld	s11, 48(sp)
   1890a: 13 0d 60 06  	addi	s10, zero, 102
   1890e: e1 b7        	j	0x188d6 <_svfprintf_r+0x1938>
   18910: 85 26        	addiw	a3, a3, 1
   18912: 66 97        	add	a4, a4, s9
   18914: 23 30 74 01  	sd	s7, 0(s0)
   18918: 23 34 94 01  	sd	s9, 8(s0)
   1891c: 3a ea        	sd	a4, 272(sp)
   1891e: 23 24 d1 10  	sw	a3, 264(sp)
   18922: 1d 46        	addi	a2, zero, 7
   18924: 63 51 d6 fa  	bge	a2, a3, 0x180c6 <_svfprintf_r+0x1128>
   18928: 82 65        	ld	a1, 0(sp)
   1892a: 10 02        	addi	a2, sp, 256
   1892c: 4e 85        	mv	a0, s3
   1892e: ef 10 10 58  	jal	0x1a6ae <__ssprint_r>
   18932: 19 c1        	beqz	a0, 0x18938 <_svfprintf_r+0x199a>
   18934: 6f e0 7f ff  	j	0x1792a <_svfprintf_r+0x98c>
   18938: 52 67        	ld	a4, 272(sp)
   1893a: 83 26 81 10  	lw	a3, 264(sp)
   1893e: 00 03        	addi	s0, sp, 384
   18940: 6f f0 8f f8  	j	0x180c8 <_svfprintf_r+0x112a>
   18944: 93 06 21 0e  	addi	a3, sp, 226
   18948: 99 e5        	bnez	a1, 0x18956 <_svfprintf_r+0x19b8>
   1894a: 93 06 00 03  	addi	a3, zero, 48
   1894e: 23 01 d1 0e  	sb	a3, 226(sp)
   18952: 93 06 31 0e  	addi	a3, sp, 227
   18956: 1c 04        	addi	a5, sp, 512
   18958: 33 86 f6 40  	sub	a2, a3, a5
   1895c: 1b 07 07 03  	addiw	a4, a4, 48
   18960: 9b 07 16 12  	addiw	a5, a2, 289
   18964: 23 80 e6 00  	sb	a4, 0(a3)
   18968: be f8        	sd	a5, 112(sp)
   1896a: 21 be        	j	0x18482 <_svfprintf_r+0x14e4>
   1896c: 9b 05 13 00  	addiw	a1, t1, 1
   18970: 4e 85        	mv	a0, s3
   18972: 1a ec        	sd	t1, 24(sp)
   18974: ef c0 7f da  	jal	0x1571a <_malloc_r>
   18978: 62 63        	ld	t1, 24(sp)
   1897a: 2a 8b        	mv	s6, a0
   1897c: 63 07 05 14  	beqz	a0, 0x18aca <_svfprintf_r+0x1b2c>
   18980: 2a ec        	sd	a0, 24(sp)
   18982: 35 b5        	j	0x187ae <_svfprintf_r+0x1810>
   18984: 86 67        	ld	a5, 64(sp)
   18986: 13 f6 17 00  	andi	a2, a5, 1
   1898a: e3 07 06 b0  	beqz	a2, 0x18498 <_svfprintf_r+0x14fa>
   1898e: 11 b6        	j	0x18492 <_svfprintf_r+0x14f4>
   18990: 6a 76        	ld	a2, 184(sp)
   18992: 66 8b        	mv	s6, s9
   18994: 4e f8        	sd	s3, 48(sp)
   18996: 3b 06 cb 00  	addw	a2, s6, a2
   1899a: de 89        	mv	s3, s7
   1899c: d6 8b        	mv	s7, s5
   1899e: d2 8a        	mv	s5, s4
   189a0: a6 64        	ld	s1, 72(sp)
   189a2: 2a 73        	ld	t1, 168(sp)
   189a4: 4a 74        	ld	s0, 176(sp)
   189a6: 0a 7a        	ld	s4, 160(sp)
   189a8: ca 8c        	mv	s9, s2
   189aa: bb 0d 56 41  	subw	s11, a2, s5
   189ae: bd b9        	j	0x1862c <_svfprintf_r+0x168e>
   189b0: 63 04 03 00  	beqz	t1, 0x189b8 <_svfprintf_r+0x1a1a>
   189b4: 6f e0 9f 93  	j	0x172ec <_svfprintf_r+0x34e>
   189b8: 05 43        	addi	t1, zero, 1
   189ba: 6f e0 3f 93  	j	0x172ec <_svfprintf_r+0x34e>
   189be: 46 67        	ld	a4, 80(sp)
   189c0: b1 47        	addi	a5, zero, 12
   189c2: 1c c3        	sw	a5, 0(a4)
   189c4: fd 57        	addi	a5, zero, -1
   189c6: 3e e8        	sd	a5, 16(sp)
   189c8: 6f e0 bf 84  	j	0x17212 <_svfprintf_r+0x274>
   189cc: 3a f0        	sd	a4, 32(sp)
   189ce: 6f f0 0f 81  	j	0x179de <_svfprintf_r+0xa40>
   189d2: 1c c3        	sw	a5, 0(a4)
   189d4: 6f e0 7f c7  	j	0x1764a <_svfprintf_r+0x6ac>
   189d8: 13 0d 70 06  	addi	s10, zero, 103
   189dc: e6 76        	ld	a3, 120(sp)
   189de: 13 06 f0 0f  	addi	a2, zero, 255
   189e2: 03 c7 06 00  	lbu	a4, 0(a3)
   189e6: 63 0f c7 0c  	beq	a4, a2, 0x18ac4 <_svfprintf_r+0x1b26>
   189ea: c2 77        	ld	a5, 48(sp)
   189ec: 01 48        	mv	a6, zero
   189ee: 01 45        	mv	a0, zero
   189f0: 93 05 f0 0f  	addi	a1, zero, 255
   189f4: 1b 06 07 00  	sext.w	a2, a4
   189f8: 63 5a f7 00  	bge	a4, a5, 0x18a0c <_svfprintf_r+0x1a6e>
   189fc: 03 c7 16 00  	lbu	a4, 1(a3)
   18a00: 91 9f        	subw	a5, a5, a2
   18a02: 1d cb        	beqz	a4, 0x18a38 <_svfprintf_r+0x1a9a>
   18a04: 05 25        	addiw	a0, a0, 1
   18a06: 85 06        	addi	a3, a3, 1
   18a08: e3 16 b7 fe  	bne	a4, a1, 0x189f4 <_svfprintf_r+0x1a56>
   18a0c: b6 fc        	sd	a3, 120(sp)
   18a0e: 3e f8        	sd	a5, 48(sp)
   18a10: aa e0        	sd	a0, 64(sp)
   18a12: c2 e4        	sd	a6, 72(sp)
   18a14: a6 67        	ld	a5, 72(sp)
   18a16: 06 67        	ld	a4, 64(sp)
   18a18: 4a 69        	ld	s2, 144(sp)
   18a1a: 3d 9f        	addw	a4, a4, a5
   18a1c: 8a 67        	ld	a5, 128(sp)
   18a1e: 3b 07 f7 02  	mulw	a4, a4, a5
   18a22: 3b 07 b7 01  	addw	a4, a4, s11
   18a26: 9b 0d 07 00  	sext.w	s11, a4
   18a2a: 13 c6 fd ff  	not	a2, s11
   18a2e: 7d 96        	srai	a2, a2, 63
   18a30: 71 8f        	and	a4, a4, a2
   18a32: 1b 0c 07 00  	sext.w	s8, a4
   18a36: 51 b4        	j	0x184ba <_svfprintf_r+0x151c>
   18a38: 03 c7 06 00  	lbu	a4, 0(a3)
   18a3c: 05 28        	addiw	a6, a6, 1
   18a3e: e9 b7        	j	0x18a08 <_svfprintf_r+0x1a6a>
   18a40: 3b 87 fd 40  	subw	a4, s11, a5
   18a44: 05 27        	addiw	a4, a4, 1
   18a46: 9b 0d 07 00  	sext.w	s11, a4
   18a4a: 13 c6 fd ff  	not	a2, s11
   18a4e: 7d 96        	srai	a2, a2, 63
   18a50: 71 8f        	and	a4, a4, a2
   18a52: 4a 69        	ld	s2, 144(sp)
   18a54: 1b 0c 07 00  	sext.w	s8, a4
   18a58: e5 b9        	j	0x18750 <_svfprintf_r+0x17b2>
   18a5a: 66 67        	ld	a4, 88(sp)
   18a5c: 13 0d 60 06  	addi	s10, zero, 102
   18a60: bb 06 f7 00  	addw	a3, a4, a5
   18a64: bb 8d 66 00  	addw	s11, a3, t1
   18a68: bd b5        	j	0x188d6 <_svfprintf_r+0x1938>
   18a6a: 93 06 00 07  	addi	a3, zero, 112
   18a6e: 59 ba        	j	0x18404 <_svfprintf_r+0x1466>
   18a70: 82 76        	ld	a3, 32(sp)
   18a72: 98 42        	lw	a4, 0(a3)
   18a74: a1 06        	addi	a3, a3, 8
   18a76: 3a 83        	mv	t1, a4
   18a78: 63 53 07 00  	bgez	a4, 0x18a7e <_svfprintf_r+0x1ae0>
   18a7c: 7d 53        	addi	t1, zero, -1
   18a7e: 03 ce 14 00  	lbu	t3, 1(s1)
   18a82: 9b 0a 03 00  	sext.w	s5, t1
   18a86: 36 f0        	sd	a3, 32(sp)
   18a88: be 84        	mv	s1, a5
   18a8a: 6f e0 6f e1  	j	0x170a0 <_svfprintf_r+0x102>
   18a8e: 19 e7        	bnez	a4, 0x18a9c <_svfprintf_r+0x1afe>
   18a90: 4a 69        	ld	s2, 144(sp)
   18a92: 05 4c        	addi	s8, zero, 1
   18a94: 13 0d 60 06  	addi	s10, zero, 102
   18a98: 85 4d        	addi	s11, zero, 1
   18a9a: 5d b9        	j	0x18750 <_svfprintf_r+0x17b2>
   18a9c: e6 67        	ld	a5, 88(sp)
   18a9e: 4a 69        	ld	s2, 144(sp)
   18aa0: 13 0d 60 06  	addi	s10, zero, 102
   18aa4: 1b 87 17 00  	addiw	a4, a5, 1
   18aa8: 3b 07 67 00  	addw	a4, a4, t1
   18aac: 9b 0d 07 00  	sext.w	s11, a4
   18ab0: 13 c6 fd ff  	not	a2, s11
   18ab4: 7d 96        	srai	a2, a2, 63
   18ab6: 71 8f        	and	a4, a4, a2
   18ab8: 1b 0c 07 00  	sext.w	s8, a4
   18abc: 51 b9        	j	0x18750 <_svfprintf_r+0x17b2>
   18abe: 89 47        	addi	a5, zero, 2
   18ac0: be f8        	sd	a5, 112(sp)
   18ac2: c1 b2        	j	0x18482 <_svfprintf_r+0x14e4>
   18ac4: 82 e4        	sd	zero, 72(sp)
   18ac6: 82 e0        	sd	zero, 64(sp)
   18ac8: b1 b7        	j	0x18a14 <_svfprintf_r+0x1a76>
   18aca: 02 67        	ld	a4, 0(sp)
   18acc: 83 57 07 01  	lhu	a5, 16(a4)
   18ad0: 93 e7 07 04  	ori	a5, a5, 64
   18ad4: 23 18 f7 00  	sh	a5, 16(a4)
   18ad8: 6f e0 af f2  	j	0x17202 <_svfprintf_r+0x264>

0000000000018adc <__sprint_r.part.0>:
   18adc: 83 a7 c5 0a  	lw	a5, 172(a1)
   18ae0: 5d 71        	addi	sp, sp, -80
   18ae2: 56 ec        	sd	s5, 24(sp)
   18ae4: b5 83        	srli	a5, a5, 13
   18ae6: 86 e4        	sd	ra, 72(sp)
   18ae8: a2 e0        	sd	s0, 64(sp)
   18aea: 26 fc        	sd	s1, 56(sp)
   18aec: 4a f8        	sd	s2, 48(sp)
   18aee: 4e f4        	sd	s3, 40(sp)
   18af0: 52 f0        	sd	s4, 32(sp)
   18af2: 5a e8        	sd	s6, 16(sp)
   18af4: 5e e4        	sd	s7, 8(sp)
   18af6: 85 8b        	andi	a5, a5, 1
   18af8: b2 8a        	mv	s5, a2
   18afa: c1 c3        	beqz	a5, 0x18b7a <__sprint_r.part.0+0x9e>
   18afc: 1c 6a        	ld	a5, 16(a2)
   18afe: 03 3b 06 00  	ld	s6, 0(a2)
   18b02: 2e 89        	mv	s2, a1
   18b04: aa 89        	mv	s3, a0
   18b06: 7d 5a        	addi	s4, zero, -1
   18b08: bd c7        	beqz	a5, 0x18b76 <__sprint_r.part.0+0x9a>
   18b0a: 83 34 8b 00  	ld	s1, 8(s6)
   18b0e: 03 34 0b 00  	ld	s0, 0(s6)
   18b12: 89 80        	srli	s1, s1, 2
   18b14: 9b 8b 04 00  	sext.w	s7, s1
   18b18: 63 58 70 05  	blez	s7, 0x18b68 <__sprint_r.part.0+0x8c>
   18b1c: fd 34        	addiw	s1, s1, -1
   18b1e: 93 97 04 02  	slli	a5, s1, 32
   18b22: 93 d4 e7 01  	srli	s1, a5, 30
   18b26: 93 07 44 00  	addi	a5, s0, 4
   18b2a: be 94        	add	s1, s1, a5
   18b2c: 19 a0        	j	0x18b32 <__sprint_r.part.0+0x56>
   18b2e: 63 8b 84 02  	beq	s1, s0, 0x18b64 <__sprint_r.part.0+0x88>
   18b32: 0c 40        	lw	a1, 0(s0)
   18b34: 4a 86        	mv	a2, s2
   18b36: 4e 85        	mv	a0, s3
   18b38: ef 10 40 11  	jal	0x19c4c <_fputwc_r>
   18b3c: 01 25        	sext.w	a0, a0
   18b3e: 11 04        	addi	s0, s0, 4
   18b40: e3 17 45 ff  	bne	a0, s4, 0x18b2e <__sprint_r.part.0+0x52>
   18b44: 7d 55        	addi	a0, zero, -1
   18b46: a6 60        	ld	ra, 72(sp)
   18b48: 06 64        	ld	s0, 64(sp)
   18b4a: 23 b8 0a 00  	sd	zero, 16(s5)
   18b4e: 23 a4 0a 00  	sw	zero, 8(s5)
   18b52: e2 74        	ld	s1, 56(sp)
   18b54: 42 79        	ld	s2, 48(sp)
   18b56: a2 79        	ld	s3, 40(sp)
   18b58: 02 7a        	ld	s4, 32(sp)
   18b5a: e2 6a        	ld	s5, 24(sp)
   18b5c: 42 6b        	ld	s6, 16(sp)
   18b5e: a2 6b        	ld	s7, 8(sp)
   18b60: 61 61        	addi	sp, sp, 80
   18b62: 82 80        	ret
   18b64: 83 b7 0a 01  	ld	a5, 16(s5)
   18b68: 8a 0b        	slli	s7, s7, 2
   18b6a: b3 87 77 41  	sub	a5, a5, s7
   18b6e: 23 b8 fa 00  	sd	a5, 16(s5)
   18b72: 41 0b        	addi	s6, s6, 16
   18b74: d9 fb        	bnez	a5, 0x18b0a <__sprint_r.part.0+0x2e>
   18b76: 01 45        	mv	a0, zero
   18b78: f9 b7        	j	0x18b46 <__sprint_r.part.0+0x6a>
   18b7a: ef 10 00 18  	jal	0x19cfa <__sfvwrite_r>
   18b7e: e1 b7        	j	0x18b46 <__sprint_r.part.0+0x6a>

0000000000018b80 <__sprint_r>:
   18b80: 18 6a        	ld	a4, 16(a2)
   18b82: 11 c3        	beqz	a4, 0x18b86 <__sprint_r+0x6>
   18b84: a1 bf        	j	0x18adc <__sprint_r.part.0>
   18b86: 23 24 06 00  	sw	zero, 8(a2)
   18b8a: 01 45        	mv	a0, zero
   18b8c: 82 80        	ret

0000000000018b8e <_vfiprintf_r>:
   18b8e: 05 71        	addi	sp, sp, -480
   18b90: 52 fb        	sd	s4, 432(sp)
   18b92: 56 f7        	sd	s5, 424(sp)
   18b94: 5e ef        	sd	s7, 408(sp)
   18b96: ee fe        	sd	s11, 376(sp)
   18b98: 86 ef        	sd	ra, 472(sp)
   18b9a: a2 eb        	sd	s0, 464(sp)
   18b9c: a6 e7        	sd	s1, 456(sp)
   18b9e: ca e3        	sd	s2, 448(sp)
   18ba0: 4e ff        	sd	s3, 440(sp)
   18ba2: 5a f3        	sd	s6, 416(sp)
   18ba4: 62 eb        	sd	s8, 400(sp)
   18ba6: 66 e7        	sd	s9, 392(sp)
   18ba8: 6a e3        	sd	s10, 384(sp)
   18baa: aa 8a        	mv	s5, a0
   18bac: 2e 8a        	mv	s4, a1
   18bae: b2 8d        	mv	s11, a2
   18bb0: b6 8b        	mv	s7, a3
   18bb2: 01 c5        	beqz	a0, 0x18bba <_vfiprintf_r+0x2c>
   18bb4: 3c 49        	lw	a5, 80(a0)
   18bb6: 63 88 07 38  	beqz	a5, 0x18f46 <_vfiprintf_r+0x3b8>
   18bba: 83 17 0a 01  	lh	a5, 16(s4)
   18bbe: 13 97 27 03  	slli	a4, a5, 50
   18bc2: 63 41 07 02  	bltz	a4, 0x18be4 <_vfiprintf_r+0x56>
   18bc6: 03 27 ca 0a  	lw	a4, 172(s4)
   18bca: 89 66        	lui	a3, 2
   18bcc: d5 8f        	or	a5, a5, a3
   18bce: f9 76        	lui	a3, 1048574
   18bd0: 9b 97 07 01  	slliw	a5, a5, 16
   18bd4: fd 16        	addi	a3, a3, -1
   18bd6: 9b d7 07 41  	sraiw	a5, a5, 16
   18bda: 75 8f        	and	a4, a4, a3
   18bdc: 23 18 fa 00  	sh	a5, 16(s4)
   18be0: 23 26 ea 0a  	sw	a4, 172(s4)
   18be4: 13 f7 87 00  	andi	a4, a5, 8
   18be8: 79 c7        	beqz	a4, 0x18cb6 <_vfiprintf_r+0x128>
   18bea: 03 37 8a 01  	ld	a4, 24(s4)
   18bee: 61 c7        	beqz	a4, 0x18cb6 <_vfiprintf_r+0x128>
   18bf0: e9 8b        	andi	a5, a5, 26
   18bf2: 29 47        	addi	a4, zero, 10
   18bf4: 63 8d e7 0c  	beq	a5, a4, 0x18cce <_vfiprintf_r+0x140>
   18bf8: 79 6d        	lui	s10, 30
   18bfa: 93 09 01 0f  	addi	s3, sp, 240
   18bfe: 01 4c        	mv	s8, zero
   18c00: 93 07 0d 0d  	addi	a5, s10, 208
   18c04: 79 6b        	lui	s6, 30
   18c06: e2 8c        	mv	s9, s8
   18c08: ce f8        	sd	s3, 112(sp)
   18c0a: 02 e1        	sd	zero, 128(sp)
   18c0c: 82 dc        	sw	zero, 120(sp)
   18c0e: 4e 84        	mv	s0, s3
   18c10: 02 e8        	sd	zero, 16(sp)
   18c12: 02 fc        	sd	zero, 56(sp)
   18c14: 02 f8        	sd	zero, 48(sp)
   18c16: 82 e0        	sd	zero, 64(sp)
   18c18: 3e e4        	sd	a5, 8(sp)
   18c1a: 41 49        	addi	s2, zero, 16
   18c1c: 13 0b 0b 24  	addi	s6, s6, 576
   18c20: 5e 8c        	mv	s8, s7
   18c22: 83 c7 0d 00  	lbu	a5, 0(s11)
   18c26: 63 89 07 24  	beqz	a5, 0x18e78 <_vfiprintf_r+0x2ea>
   18c2a: ee 84        	mv	s1, s11
   18c2c: 13 07 50 02  	addi	a4, zero, 37
   18c30: 63 80 e7 2a  	beq	a5, a4, 0x18ed0 <_vfiprintf_r+0x342>
   18c34: 83 c7 14 00  	lbu	a5, 1(s1)
   18c38: 85 04        	addi	s1, s1, 1
   18c3a: fd fb        	bnez	a5, 0x18c30 <_vfiprintf_r+0xa2>
   18c3c: bb 8b b4 41  	subw	s7, s1, s11
   18c40: 63 8c 0b 22  	beqz	s7, 0x18e78 <_vfiprintf_r+0x2ea>
   18c44: e6 57        	lw	a5, 120(sp)
   18c46: 0a 67        	ld	a4, 128(sp)
   18c48: 23 30 b4 01  	sd	s11, 0(s0)
   18c4c: 9b 86 17 00  	addiw	a3, a5, 1
   18c50: 5e 97        	add	a4, a4, s7
   18c52: 23 34 74 01  	sd	s7, 8(s0)
   18c56: 3a e1        	sd	a4, 128(sp)
   18c58: b6 dc        	sw	a3, 120(sp)
   18c5a: 9d 47        	addi	a5, zero, 7
   18c5c: 41 04        	addi	s0, s0, 16
   18c5e: 63 db d7 00  	bge	a5, a3, 0x18c74 <_vfiprintf_r+0xe6>
   18c62: e3 0b 07 0e  	beqz	a4, 0x19558 <_vfiprintf_r+0x9ca>
   18c66: 90 18        	addi	a2, sp, 112
   18c68: d2 85        	mv	a1, s4
   18c6a: 56 85        	mv	a0, s5
   18c6c: ef f0 1f e7  	jal	0x18adc <__sprint_r.part.0>
   18c70: 51 ed        	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   18c72: 4e 84        	mv	s0, s3
   18c74: 83 c7 04 00  	lbu	a5, 0(s1)
   18c78: bb 8c 7c 01  	addw	s9, s9, s7
   18c7c: 63 8e 07 1e  	beqz	a5, 0x18e78 <_vfiprintf_r+0x2ea>
   18c80: 83 c7 14 00  	lbu	a5, 1(s1)
   18c84: 93 8d 14 00  	addi	s11, s1, 1
   18c88: a3 03 01 06  	sb	zero, 103(sp)
   18c8c: 7d 5d        	addi	s10, zero, -1
   18c8e: 81 48        	mv	a7, zero
   18c90: 81 4b        	mv	s7, zero
   18c92: 93 04 a0 05  	addi	s1, zero, 90
   18c96: 85 0d        	addi	s11, s11, 1
   18c98: 81 27        	sext.w	a5, a5
   18c9a: 1b 87 07 fe  	addiw	a4, a5, -32
   18c9e: 9b 06 07 00  	sext.w	a3, a4
   18ca2: 63 e7 d4 0c  	bltu	s1, a3, 0x18d70 <_vfiprintf_r+0x1e2>
   18ca6: 93 16 07 02  	slli	a3, a4, 32
   18caa: 13 d7 e6 01  	srli	a4, a3, 30
   18cae: a2 66        	ld	a3, 8(sp)
   18cb0: 36 97        	add	a4, a4, a3
   18cb2: 18 43        	lw	a4, 0(a4)
   18cb4: 02 87        	jr	a4
   18cb6: d2 85        	mv	a1, s4
   18cb8: 56 85        	mv	a0, s5
   18cba: ef 90 4f eb  	jal	0x1236e <__swsetup_r>
   18cbe: e3 18 05 36  	bnez	a0, 0x1982e <_vfiprintf_r+0xca0>
   18cc2: 83 17 0a 01  	lh	a5, 16(s4)
   18cc6: 29 47        	addi	a4, zero, 10
   18cc8: e9 8b        	andi	a5, a5, 26
   18cca: e3 97 e7 f2  	bne	a5, a4, 0x18bf8 <_vfiprintf_r+0x6a>
   18cce: 83 17 2a 01  	lh	a5, 18(s4)
   18cd2: e3 c3 07 f2  	bltz	a5, 0x18bf8 <_vfiprintf_r+0x6a>
   18cd6: 5e 64        	ld	s0, 464(sp)
   18cd8: fe 60        	ld	ra, 472(sp)
   18cda: be 64        	ld	s1, 456(sp)
   18cdc: 1e 69        	ld	s2, 448(sp)
   18cde: fa 79        	ld	s3, 440(sp)
   18ce0: 1a 7b        	ld	s6, 416(sp)
   18ce2: 5a 6c        	ld	s8, 400(sp)
   18ce4: ba 6c        	ld	s9, 392(sp)
   18ce6: 1a 6d        	ld	s10, 384(sp)
   18ce8: de 86        	mv	a3, s7
   18cea: 6e 86        	mv	a2, s11
   18cec: fa 6b        	ld	s7, 408(sp)
   18cee: f6 7d        	ld	s11, 376(sp)
   18cf0: d2 85        	mv	a1, s4
   18cf2: 56 85        	mv	a0, s5
   18cf4: 5a 7a        	ld	s4, 432(sp)
   18cf6: ba 7a        	ld	s5, 424(sp)
   18cf8: 3d 61        	addi	sp, sp, 480
   18cfa: 6f 00 f0 36  	j	0x19868 <__sbprintf>
   18cfe: 90 18        	addi	a2, sp, 112
   18d00: d2 85        	mv	a1, s4
   18d02: 56 85        	mv	a0, s5
   18d04: ef f0 9f dd  	jal	0x18adc <__sprint_r.part.0>
   18d08: 63 02 05 16  	beqz	a0, 0x18e6c <_vfiprintf_r+0x2de>
   18d0c: 66 8c        	mv	s8, s9
   18d0e: 83 57 0a 01  	lhu	a5, 16(s4)
   18d12: 93 f7 07 04  	andi	a5, a5, 64
   18d16: e3 9c 07 30  	bnez	a5, 0x1982e <_vfiprintf_r+0xca0>
   18d1a: fe 60        	ld	ra, 472(sp)
   18d1c: 5e 64        	ld	s0, 464(sp)
   18d1e: be 64        	ld	s1, 456(sp)
   18d20: 1e 69        	ld	s2, 448(sp)
   18d22: fa 79        	ld	s3, 440(sp)
   18d24: 5a 7a        	ld	s4, 432(sp)
   18d26: ba 7a        	ld	s5, 424(sp)
   18d28: 1a 7b        	ld	s6, 416(sp)
   18d2a: fa 6b        	ld	s7, 408(sp)
   18d2c: ba 6c        	ld	s9, 392(sp)
   18d2e: 1a 6d        	ld	s10, 384(sp)
   18d30: f6 7d        	ld	s11, 376(sp)
   18d32: 62 85        	mv	a0, s8
   18d34: 5a 6c        	ld	s8, 400(sp)
   18d36: 3d 61        	addi	sp, sp, 480
   18d38: 82 80        	ret
   18d3a: 81 48        	mv	a7, zero
   18d3c: 25 46        	addi	a2, zero, 9
   18d3e: 83 c6 0d 00  	lbu	a3, 0(s11)
   18d42: 1b 97 28 00  	slliw	a4, a7, 2
   18d46: bb 08 17 01  	addw	a7, a4, a7
   18d4a: 9b 87 07 fd  	addiw	a5, a5, -48
   18d4e: 9b 98 18 00  	slliw	a7, a7, 1
   18d52: 1b 87 06 fd  	addiw	a4, a3, -48
   18d56: bb 88 17 01  	addw	a7, a5, a7
   18d5a: 85 0d        	addi	s11, s11, 1
   18d5c: 9b 87 06 00  	sext.w	a5, a3
   18d60: e3 7f e6 fc  	bgeu	a2, a4, 0x18d3e <_vfiprintf_r+0x1b0>
   18d64: 1b 87 07 fe  	addiw	a4, a5, -32
   18d68: 9b 06 07 00  	sext.w	a3, a4
   18d6c: e3 fd d4 f2  	bgeu	s1, a3, 0x18ca6 <_vfiprintf_r+0x118>
   18d70: 63 84 07 10  	beqz	a5, 0x18e78 <_vfiprintf_r+0x2ea>
   18d74: 23 04 f1 08  	sb	a5, 136(sp)
   18d78: a3 03 01 06  	sb	zero, 103(sp)
   18d7c: 05 4e        	addi	t3, zero, 1
   18d7e: 05 43        	addi	t1, zero, 1
   18d80: 24 01        	addi	s1, sp, 136
   18d82: 01 4d        	mv	s10, zero
   18d84: 93 ff 2b 00  	andi	t6, s7, 2
   18d88: 63 83 0f 00  	beqz	t6, 0x18d8e <_vfiprintf_r+0x200>
   18d8c: 09 2e        	addiw	t3, t3, 2
   18d8e: 66 57        	lw	a4, 120(sp)
   18d90: 93 f2 4b 08  	andi	t0, s7, 132
   18d94: 8a 67        	ld	a5, 128(sp)
   18d96: 9b 05 17 00  	addiw	a1, a4, 1
   18d9a: 2e 86        	mv	a2, a1
   18d9c: 63 96 02 00  	bnez	t0, 0x18da8 <_vfiprintf_r+0x21a>
   18da0: bb 86 c8 41  	subw	a3, a7, t3
   18da4: 63 43 d0 62  	bgtz	a3, 0x193ca <_vfiprintf_r+0x83c>
   18da8: 83 46 71 06  	lbu	a3, 103(sp)
   18dac: 99 ce        	beqz	a3, 0x18dca <_vfiprintf_r+0x23c>
   18dae: 13 07 71 06  	addi	a4, sp, 103
   18db2: 85 07        	addi	a5, a5, 1
   18db4: 18 e0        	sd	a4, 0(s0)
   18db6: 05 47        	addi	a4, zero, 1
   18db8: 18 e4        	sd	a4, 8(s0)
   18dba: 3e e1        	sd	a5, 128(sp)
   18dbc: ae dc        	sw	a1, 120(sp)
   18dbe: 1d 47        	addi	a4, zero, 7
   18dc0: 63 46 b7 58  	blt	a4, a1, 0x1934c <_vfiprintf_r+0x7be>
   18dc4: 2e 87        	mv	a4, a1
   18dc6: 41 04        	addi	s0, s0, 16
   18dc8: 85 25        	addiw	a1, a1, 1
   18dca: 63 81 0f 04  	beqz	t6, 0x18e0c <_vfiprintf_r+0x27e>
   18dce: b8 10        	addi	a4, sp, 104
   18dd0: 89 07        	addi	a5, a5, 2
   18dd2: 18 e0        	sd	a4, 0(s0)
   18dd4: 09 47        	addi	a4, zero, 2
   18dd6: 18 e4        	sd	a4, 8(s0)
   18dd8: 3e e1        	sd	a5, 128(sp)
   18dda: ae dc        	sw	a1, 120(sp)
   18ddc: 1d 47        	addi	a4, zero, 7
   18dde: 63 57 b7 5a  	bge	a4, a1, 0x1938c <_vfiprintf_r+0x7fe>
   18de2: 63 8f 07 76  	beqz	a5, 0x19560 <_vfiprintf_r+0x9d2>
   18de6: 90 18        	addi	a2, sp, 112
   18de8: d2 85        	mv	a1, s4
   18dea: 56 85        	mv	a0, s5
   18dec: 9a e4        	sd	t1, 72(sp)
   18dee: 72 f4        	sd	t3, 40(sp)
   18df0: 46 f0        	sd	a7, 32(sp)
   18df2: 16 ec        	sd	t0, 24(sp)
   18df4: ef f0 9f ce  	jal	0x18adc <__sprint_r.part.0>
   18df8: 11 f9        	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   18dfa: 66 57        	lw	a4, 120(sp)
   18dfc: 8a 67        	ld	a5, 128(sp)
   18dfe: 26 63        	ld	t1, 72(sp)
   18e00: 22 7e        	ld	t3, 40(sp)
   18e02: 82 78        	ld	a7, 32(sp)
   18e04: e2 62        	ld	t0, 24(sp)
   18e06: 4e 84        	mv	s0, s3
   18e08: 9b 05 17 00  	addiw	a1, a4, 1
   18e0c: 93 06 00 08  	addi	a3, zero, 128
   18e10: 63 82 d2 3e  	beq	t0, a3, 0x191f4 <_vfiprintf_r+0x666>
   18e14: 3b 0d 6d 40  	subw	s10, s10, t1
   18e18: 63 4f a0 49  	bgtz	s10, 0x192b6 <_vfiprintf_r+0x728>
   18e1c: 9a 97        	add	a5, a5, t1
   18e1e: 04 e0        	sd	s1, 0(s0)
   18e20: 23 34 64 00  	sd	t1, 8(s0)
   18e24: 3e e1        	sd	a5, 128(sp)
   18e26: ae dc        	sw	a1, 120(sp)
   18e28: 1d 47        	addi	a4, zero, 7
   18e2a: 63 53 b7 3c  	bge	a4, a1, 0x191f0 <_vfiprintf_r+0x662>
   18e2e: 63 80 07 64  	beqz	a5, 0x1946e <_vfiprintf_r+0x8e0>
   18e32: 90 18        	addi	a2, sp, 112
   18e34: d2 85        	mv	a1, s4
   18e36: 56 85        	mv	a0, s5
   18e38: 72 f0        	sd	t3, 32(sp)
   18e3a: 46 ec        	sd	a7, 24(sp)
   18e3c: ef f0 1f ca  	jal	0x18adc <__sprint_r.part.0>
   18e40: e3 16 05 ec  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   18e44: 8a 67        	ld	a5, 128(sp)
   18e46: 02 7e        	ld	t3, 32(sp)
   18e48: e2 68        	ld	a7, 24(sp)
   18e4a: 4e 84        	mv	s0, s3
   18e4c: 93 fb 4b 00  	andi	s7, s7, 4
   18e50: 63 86 0b 00  	beqz	s7, 0x18e5c <_vfiprintf_r+0x2ce>
   18e54: bb 84 c8 41  	subw	s1, a7, t3
   18e58: 63 45 90 62  	bgtz	s1, 0x19482 <_vfiprintf_r+0x8f4>
   18e5c: 46 87        	mv	a4, a7
   18e5e: 63 d3 c8 01  	bge	a7, t3, 0x18e64 <_vfiprintf_r+0x2d6>
   18e62: 72 87        	mv	a4, t3
   18e64: bb 0c 97 01  	addw	s9, a4, s9
   18e68: e3 9b 07 e8  	bnez	a5, 0x18cfe <_vfiprintf_r+0x170>
   18e6c: 83 c7 0d 00  	lbu	a5, 0(s11)
   18e70: 82 dc        	sw	zero, 120(sp)
   18e72: 4e 84        	mv	s0, s3
   18e74: e3 9b 07 da  	bnez	a5, 0x18c2a <_vfiprintf_r+0x9c>
   18e78: 8a 67        	ld	a5, 128(sp)
   18e7a: 66 8c        	mv	s8, s9
   18e7c: e3 89 07 e8  	beqz	a5, 0x18d0e <_vfiprintf_r+0x180>
   18e80: 90 18        	addi	a2, sp, 112
   18e82: d2 85        	mv	a1, s4
   18e84: 56 85        	mv	a0, s5
   18e86: ef f0 7f c5  	jal	0x18adc <__sprint_r.part.0>
   18e8a: 51 b5        	j	0x18d0e <_vfiprintf_r+0x180>
   18e8c: 93 eb 0b 01  	ori	s7, s7, 16
   18e90: 83 c7 0d 00  	lbu	a5, 0(s11)
   18e94: 81 2b        	sext.w	s7, s7
   18e96: 01 b5        	j	0x18c96 <_vfiprintf_r+0x108>
   18e98: 93 eb 0b 01  	ori	s7, s7, 16
   18e9c: 81 2b        	sext.w	s7, s7
   18e9e: 93 f7 0b 02  	andi	a5, s7, 32
   18ea2: 93 0e 8c 00  	addi	t4, s8, 8
   18ea6: 85 cf        	beqz	a5, 0x18ede <_vfiprintf_r+0x350>
   18ea8: 83 37 0c 00  	ld	a5, 0(s8)
   18eac: 3e 8c        	mv	s8, a5
   18eae: 63 c8 07 04  	bltz	a5, 0x18efe <_vfiprintf_r+0x370>
   18eb2: fd 57        	addi	a5, zero, -1
   18eb4: 63 01 fd 06  	beq	s10, a5, 0x18f16 <_vfiprintf_r+0x388>
   18eb8: 93 fb fb f7  	andi	s7, s7, -129
   18ebc: 81 2b        	sext.w	s7, s7
   18ebe: 63 1c 0c 04  	bnez	s8, 0x18f16 <_vfiprintf_r+0x388>
   18ec2: e3 16 0d 14  	bnez	s10, 0x1980e <_vfiprintf_r+0xc80>
   18ec6: 01 4e        	mv	t3, zero
   18ec8: 01 43        	mv	t1, zero
   18eca: 76 8c        	mv	s8, t4
   18ecc: e4 11        	addi	s1, sp, 236
   18ece: b5 a0        	j	0x18f3a <_vfiprintf_r+0x3ac>
   18ed0: bb 8b b4 41  	subw	s7, s1, s11
   18ed4: e3 98 0b d6  	bnez	s7, 0x18c44 <_vfiprintf_r+0xb6>
   18ed8: 83 c7 04 00  	lbu	a5, 0(s1)
   18edc: 45 b3        	j	0x18c7c <_vfiprintf_r+0xee>
   18ede: 93 f7 0b 01  	andi	a5, s7, 16
   18ee2: f9 f3        	bnez	a5, 0x18ea8 <_vfiprintf_r+0x31a>
   18ee4: 93 f7 0b 04  	andi	a5, s7, 64
   18ee8: 03 2c 0c 00  	lw	s8, 0(s8)
   18eec: e3 81 07 0c  	beqz	a5, 0x197ae <_vfiprintf_r+0xc20>
   18ef0: 1b 1c 0c 01  	slliw	s8, s8, 16
   18ef4: 1b 5c 0c 41  	sraiw	s8, s8, 16
   18ef8: e2 87        	mv	a5, s8
   18efa: e3 dc 07 fa  	bgez	a5, 0x18eb2 <_vfiprintf_r+0x324>
   18efe: 93 07 d0 02  	addi	a5, zero, 45
   18f02: a3 03 f1 06  	sb	a5, 103(sp)
   18f06: fd 57        	addi	a5, zero, -1
   18f08: 33 0c 80 41  	neg	s8, s8
   18f0c: 63 05 fd 00  	beq	s10, a5, 0x18f16 <_vfiprintf_r+0x388>
   18f10: 93 fb fb f7  	andi	s7, s7, -129
   18f14: 81 2b        	sext.w	s7, s7
   18f16: a5 47        	addi	a5, zero, 9
   18f18: 63 e6 87 69  	bltu	a5, s8, 0x195a4 <_vfiprintf_r+0xa16>
   18f1c: 9b 07 0c 03  	addiw	a5, s8, 48
   18f20: 93 f7 f7 0f  	andi	a5, a5, 255
   18f24: 76 8c        	mv	s8, t4
   18f26: a3 05 f1 0e  	sb	a5, 235(sp)
   18f2a: 1b 0e 0d 00  	sext.w	t3, s10
   18f2e: 63 43 a0 01  	bgtz	s10, 0x18f34 <_vfiprintf_r+0x3a6>
   18f32: 05 4e        	addi	t3, zero, 1
   18f34: 05 43        	addi	t1, zero, 1
   18f36: 93 04 b1 0e  	addi	s1, sp, 235
   18f3a: 83 47 71 06  	lbu	a5, 103(sp)
   18f3e: e3 83 07 e4  	beqz	a5, 0x18d84 <_vfiprintf_r+0x1f6>
   18f42: 05 2e        	addiw	t3, t3, 1
   18f44: 81 b5        	j	0x18d84 <_vfiprintf_r+0x1f6>
   18f46: ef 90 bf a1  	jal	0x12960 <__sinit>
   18f4a: 85 b9        	j	0x18bba <_vfiprintf_r+0x2c>
   18f4c: 83 34 0c 00  	ld	s1, 0(s8)
   18f50: a3 03 01 06  	sb	zero, 103(sp)
   18f54: 21 0c        	addi	s8, s8, 8
   18f56: e3 8d 04 00  	beqz	s1, 0x19770 <_vfiprintf_r+0xbe2>
   18f5a: fd 57        	addi	a5, zero, -1
   18f5c: 46 ec        	sd	a7, 24(sp)
   18f5e: 63 0b fd 76  	beq	s10, a5, 0x196d4 <_vfiprintf_r+0xb46>
   18f62: 6a 86        	mv	a2, s10
   18f64: 81 45        	mv	a1, zero
   18f66: 26 85        	mv	a0, s1
   18f68: ef c0 ff de  	jal	0x15d56 <memchr>
   18f6c: e2 68        	ld	a7, 24(sp)
   18f6e: e3 07 05 08  	beqz	a0, 0x197fc <_vfiprintf_r+0xc6e>
   18f72: 3b 03 95 40  	subw	t1, a0, s1
   18f76: 13 4e f3 ff  	not	t3, t1
   18f7a: 13 5e fe 43  	srai	t3, t3, 63
   18f7e: 33 7e 6e 00  	and	t3, t3, t1
   18f82: 01 4d        	mv	s10, zero
   18f84: 5d bf        	j	0x18f3a <_vfiprintf_r+0x3ac>
   18f86: 83 27 0c 00  	lw	a5, 0(s8)
   18f8a: a3 03 01 06  	sb	zero, 103(sp)
   18f8e: 21 0c        	addi	s8, s8, 8
   18f90: 23 04 f1 08  	sb	a5, 136(sp)
   18f94: 05 4e        	addi	t3, zero, 1
   18f96: 05 43        	addi	t1, zero, 1
   18f98: 24 01        	addi	s1, sp, 136
   18f9a: e5 b3        	j	0x18d82 <_vfiprintf_r+0x1f4>
   18f9c: 83 28 0c 00  	lw	a7, 0(s8)
   18fa0: 21 0c        	addi	s8, s8, 8
   18fa2: 63 db 08 08  	bgez	a7, 0x19038 <_vfiprintf_r+0x4aa>
   18fa6: bb 08 10 41  	negw	a7, a7
   18faa: 93 eb 4b 00  	ori	s7, s7, 4
   18fae: 83 c7 0d 00  	lbu	a5, 0(s11)
   18fb2: 81 2b        	sext.w	s7, s7
   18fb4: cd b1        	j	0x18c96 <_vfiprintf_r+0x108>
   18fb6: 13 07 b0 02  	addi	a4, zero, 43
   18fba: 83 c7 0d 00  	lbu	a5, 0(s11)
   18fbe: a3 03 e1 06  	sb	a4, 103(sp)
   18fc2: d1 b9        	j	0x18c96 <_vfiprintf_r+0x108>
   18fc4: 93 eb 0b 08  	ori	s7, s7, 128
   18fc8: 83 c7 0d 00  	lbu	a5, 0(s11)
   18fcc: 81 2b        	sext.w	s7, s7
   18fce: e1 b1        	j	0x18c96 <_vfiprintf_r+0x108>
   18fd0: 83 c7 0d 00  	lbu	a5, 0(s11)
   18fd4: 13 06 a0 02  	addi	a2, zero, 42
   18fd8: 13 87 1d 00  	addi	a4, s11, 1
   18fdc: e3 81 c7 06  	beq	a5, a2, 0x1983e <_vfiprintf_r+0xcb0>
   18fe0: 9b 86 07 fd  	addiw	a3, a5, -48
   18fe4: a5 45        	addi	a1, zero, 9
   18fe6: ba 8d        	mv	s11, a4
   18fe8: 01 4d        	mv	s10, zero
   18fea: 25 46        	addi	a2, zero, 9
   18fec: e3 e7 d5 ca  	bltu	a1, a3, 0x18c9a <_vfiprintf_r+0x10c>
   18ff0: 83 c7 0d 00  	lbu	a5, 0(s11)
   18ff4: 1b 17 2d 00  	slliw	a4, s10, 2
   18ff8: 3b 0d a7 01  	addw	s10, a4, s10
   18ffc: 1b 1d 1d 00  	slliw	s10, s10, 1
   19000: 3b 0d dd 00  	addw	s10, s10, a3
   19004: 9b 86 07 fd  	addiw	a3, a5, -48
   19008: 85 0d        	addi	s11, s11, 1
   1900a: e3 73 d6 fe  	bgeu	a2, a3, 0x18ff0 <_vfiprintf_r+0x462>
   1900e: 71 b1        	j	0x18c9a <_vfiprintf_r+0x10c>
   19010: 56 85        	mv	a0, s5
   19012: 46 ec        	sd	a7, 24(sp)
   19014: ef c0 4f cb  	jal	0x154c8 <_localeconv_r>
   19018: 1c 65        	ld	a5, 8(a0)
   1901a: 3e 85        	mv	a0, a5
   1901c: be e0        	sd	a5, 64(sp)
   1901e: ef d0 3f e6  	jal	0x16e80 <strlen>
   19022: aa 87        	mv	a5, a0
   19024: 56 85        	mv	a0, s5
   19026: 3e f8        	sd	a5, 48(sp)
   19028: ef c0 0f ca  	jal	0x154c8 <_localeconv_r>
   1902c: 18 69        	ld	a4, 16(a0)
   1902e: c2 77        	ld	a5, 48(sp)
   19030: e2 68        	ld	a7, 24(sp)
   19032: 3a fc        	sd	a4, 56(sp)
   19034: 63 94 07 60  	bnez	a5, 0x1963c <_vfiprintf_r+0xaae>
   19038: 83 c7 0d 00  	lbu	a5, 0(s11)
   1903c: a9 b9        	j	0x18c96 <_vfiprintf_r+0x108>
   1903e: 93 eb 1b 00  	ori	s7, s7, 1
   19042: 83 c7 0d 00  	lbu	a5, 0(s11)
   19046: 81 2b        	sext.w	s7, s7
   19048: b9 b1        	j	0x18c96 <_vfiprintf_r+0x108>
   1904a: 03 47 71 06  	lbu	a4, 103(sp)
   1904e: 83 c7 0d 00  	lbu	a5, 0(s11)
   19052: e3 12 07 c4  	bnez	a4, 0x18c96 <_vfiprintf_r+0x108>
   19056: 13 07 00 02  	addi	a4, zero, 32
   1905a: a3 03 e1 06  	sb	a4, 103(sp)
   1905e: 25 b9        	j	0x18c96 <_vfiprintf_r+0x108>
   19060: 93 e7 0b 01  	ori	a5, s7, 16
   19064: 81 27        	sext.w	a5, a5
   19066: 13 f7 07 02  	andi	a4, a5, 32
   1906a: 13 06 8c 00  	addi	a2, s8, 8
   1906e: 09 e7        	bnez	a4, 0x19078 <_vfiprintf_r+0x4ea>
   19070: 13 f7 07 01  	andi	a4, a5, 16
   19074: 63 0b 07 5a  	beqz	a4, 0x1962a <_vfiprintf_r+0xa9c>
   19078: 03 37 0c 00  	ld	a4, 0(s8)
   1907c: 93 fb f7 bf  	andi	s7, a5, -1025
   19080: a3 03 01 06  	sb	zero, 103(sp)
   19084: fd 56        	addi	a3, zero, -1
   19086: 81 2b        	sext.w	s7, s7
   19088: 32 8c        	mv	s8, a2
   1908a: 63 01 dd 0a  	beq	s10, a3, 0x1912c <_vfiprintf_r+0x59e>
   1908e: 93 f7 f7 b7  	andi	a5, a5, -1153
   19092: 81 27        	sext.w	a5, a5
   19094: 63 19 07 6a  	bnez	a4, 0x19746 <_vfiprintf_r+0xbb8>
   19098: 5e 83        	mv	t1, s7
   1909a: be 8b        	mv	s7, a5
   1909c: 63 11 0d 08  	bnez	s10, 0x1911e <_vfiprintf_r+0x590>
   190a0: 63 1e 07 4e  	bnez	a4, 0x1959c <_vfiprintf_r+0xa0e>
   190a4: 13 73 13 00  	andi	t1, t1, 1
   190a8: 63 0e 03 54  	beqz	t1, 0x19604 <_vfiprintf_r+0xa76>
   190ac: 93 07 00 03  	addi	a5, zero, 48
   190b0: a3 05 f1 0e  	sb	a5, 235(sp)
   190b4: 1a 8e        	mv	t3, t1
   190b6: 93 04 b1 0e  	addi	s1, sp, 235
   190ba: 41 b5        	j	0x18f3a <_vfiprintf_r+0x3ac>
   190bc: 83 c7 0d 00  	lbu	a5, 0(s11)
   190c0: 13 07 c0 06  	addi	a4, zero, 108
   190c4: 63 8e e7 68  	beq	a5, a4, 0x19760 <_vfiprintf_r+0xbd2>
   190c8: 93 eb 0b 01  	ori	s7, s7, 16
   190cc: 81 2b        	sext.w	s7, s7
   190ce: e1 b6        	j	0x18c96 <_vfiprintf_r+0x108>
   190d0: 83 c7 0d 00  	lbu	a5, 0(s11)
   190d4: 13 07 80 06  	addi	a4, zero, 104
   190d8: 63 8c e7 66  	beq	a5, a4, 0x19750 <_vfiprintf_r+0xbc2>
   190dc: 93 eb 0b 04  	ori	s7, s7, 64
   190e0: 81 2b        	sext.w	s7, s7
   190e2: 55 be        	j	0x18c96 <_vfiprintf_r+0x108>
   190e4: 13 e3 0b 01  	ori	t1, s7, 16
   190e8: 01 23        	sext.w	t1, t1
   190ea: 93 77 03 02  	andi	a5, t1, 32
   190ee: 93 0e 8c 00  	addi	t4, s8, 8
   190f2: 89 e7        	bnez	a5, 0x190fc <_vfiprintf_r+0x56e>
   190f4: 93 77 03 01  	andi	a5, t1, 16
   190f8: 63 8f 07 50  	beqz	a5, 0x19616 <_vfiprintf_r+0xa88>
   190fc: 03 3c 0c 00  	ld	s8, 0(s8)
   19100: a3 03 01 06  	sb	zero, 103(sp)
   19104: fd 57        	addi	a5, zero, -1
   19106: 9a 8b        	mv	s7, t1
   19108: e3 07 fd e0  	beq	s10, a5, 0x18f16 <_vfiprintf_r+0x388>
   1910c: 93 7b f3 f7  	andi	s7, t1, -129
   19110: 81 2b        	sext.w	s7, s7
   19112: e3 12 0c e0  	bnez	s8, 0x18f16 <_vfiprintf_r+0x388>
   19116: 76 8c        	mv	s8, t4
   19118: 05 47        	addi	a4, zero, 1
   1911a: e3 03 0d f8  	beqz	s10, 0x190a0 <_vfiprintf_r+0x512>
   1911e: 85 47        	addi	a5, zero, 1
   19120: 63 08 f7 6e  	beq	a4, a5, 0x19810 <_vfiprintf_r+0xc82>
   19124: 89 47        	addi	a5, zero, 2
   19126: 63 00 f7 66  	beq	a4, a5, 0x19786 <_vfiprintf_r+0xbf8>
   1912a: 01 47        	mv	a4, zero
   1912c: 13 03 c1 0e  	addi	t1, sp, 236
   19130: 9a 84        	mv	s1, t1
   19132: 93 77 77 00  	andi	a5, a4, 7
   19136: 93 87 07 03  	addi	a5, a5, 48
   1913a: a3 8f f4 fe  	sb	a5, -1(s1)
   1913e: 0d 83        	srli	a4, a4, 3
   19140: a6 86        	mv	a3, s1
   19142: fd 14        	addi	s1, s1, -1
   19144: 7d f7        	bnez	a4, 0x19132 <_vfiprintf_r+0x5a4>
   19146: 13 f7 1b 00  	andi	a4, s7, 1
   1914a: 63 07 07 26  	beqz	a4, 0x193b8 <_vfiprintf_r+0x82a>
   1914e: 13 07 00 03  	addi	a4, zero, 48
   19152: 63 83 e7 26  	beq	a5, a4, 0x193b8 <_vfiprintf_r+0x82a>
   19156: f9 16        	addi	a3, a3, -2
   19158: 3b 03 d3 40  	subw	t1, t1, a3
   1915c: a3 8f e4 fe  	sb	a4, -1(s1)
   19160: 1b 0e 03 00  	sext.w	t3, t1
   19164: 63 4e a3 4f  	blt	t1, s10, 0x19660 <_vfiprintf_r+0xad2>
   19168: b6 84        	mv	s1, a3
   1916a: c1 bb        	j	0x18f3a <_vfiprintf_r+0x3ac>
   1916c: 93 eb 0b 02  	ori	s7, s7, 32
   19170: 83 c7 0d 00  	lbu	a5, 0(s11)
   19174: 81 2b        	sext.w	s7, s7
   19176: 05 b6        	j	0x18c96 <_vfiprintf_r+0x108>
   19178: e1 77        	lui	a5, 1048568
   1917a: 93 c7 07 83  	xori	a5, a5, -2000
   1917e: 23 14 f1 06  	sh	a5, 104(sp)
   19182: f5 67        	lui	a5, 29
   19184: 93 eb 2b 00  	ori	s7, s7, 2
   19188: 93 87 87 77  	addi	a5, a5, 1912
   1918c: 03 37 0c 00  	ld	a4, 0(s8)
   19190: 81 2b        	sext.w	s7, s7
   19192: 21 0c        	addi	s8, s8, 8
   19194: 3e e8        	sd	a5, 16(sp)
   19196: a3 03 01 06  	sb	zero, 103(sp)
   1919a: fd 57        	addi	a5, zero, -1
   1919c: 63 00 fd 20  	beq	s10, a5, 0x1939c <_vfiprintf_r+0x80e>
   191a0: 93 f7 fb f7  	andi	a5, s7, -129
   191a4: 81 27        	sext.w	a5, a5
   191a6: 63 17 07 1e  	bnez	a4, 0x19394 <_vfiprintf_r+0x806>
   191aa: 5e 83        	mv	t1, s7
   191ac: 09 47        	addi	a4, zero, 2
   191ae: be 8b        	mv	s7, a5
   191b0: f5 b5        	j	0x1909c <_vfiprintf_r+0x50e>
   191b2: 93 f7 0b 02  	andi	a5, s7, 32
   191b6: 03 37 0c 00  	ld	a4, 0(s8)
   191ba: 85 e3        	bnez	a5, 0x191da <_vfiprintf_r+0x64c>
   191bc: 93 f7 0b 01  	andi	a5, s7, 16
   191c0: 89 ef        	bnez	a5, 0x191da <_vfiprintf_r+0x64c>
   191c2: 93 f7 0b 04  	andi	a5, s7, 64
   191c6: 63 99 07 60  	bnez	a5, 0x197d8 <_vfiprintf_r+0xc4a>
   191ca: 93 fb 0b 20  	andi	s7, s7, 512
   191ce: 63 83 0b 66  	beqz	s7, 0x19834 <_vfiprintf_r+0xca6>
   191d2: 23 00 97 01  	sb	s9, 0(a4)
   191d6: 21 0c        	addi	s8, s8, 8
   191d8: a9 b4        	j	0x18c22 <_vfiprintf_r+0x94>
   191da: 23 30 97 01  	sd	s9, 0(a4)
   191de: 21 0c        	addi	s8, s8, 8
   191e0: 89 b4        	j	0x18c22 <_vfiprintf_r+0x94>
   191e2: 05 47        	addi	a4, zero, 1
   191e4: 9a 87        	mv	a5, t1
   191e6: a6 f9        	sd	s1, 240(sp)
   191e8: 9a fd        	sd	t1, 248(sp)
   191ea: 1a e1        	sd	t1, 128(sp)
   191ec: ba dc        	sw	a4, 120(sp)
   191ee: 4e 84        	mv	s0, s3
   191f0: 41 04        	addi	s0, s0, 16
   191f2: a9 b9        	j	0x18e4c <_vfiprintf_r+0x2be>
   191f4: bb 8e c8 41  	subw	t4, a7, t3
   191f8: e3 5e d0 c1  	blez	t4, 0x18e14 <_vfiprintf_r+0x286>
   191fc: c1 46        	addi	a3, zero, 16
   191fe: 63 d3 d6 63  	bge	a3, t4, 0x19824 <_vfiprintf_r+0xc96>
   19202: 79 6f        	lui	t5, 30
   19204: 13 0f 0f 24  	addi	t5, t5, 576
   19208: 26 ec        	sd	s1, 24(sp)
   1920a: 22 86        	mv	a2, s0
   1920c: c1 4f        	addi	t6, zero, 16
   1920e: 9d 42        	addi	t0, zero, 7
   19210: 46 f0        	sd	a7, 32(sp)
   19212: 72 f4        	sd	t3, 40(sp)
   19214: 9a e4        	sd	t1, 72(sp)
   19216: 76 84        	mv	s0, t4
   19218: fa 84        	mv	s1, t5
   1921a: 01 a8        	j	0x1922a <_vfiprintf_r+0x69c>
   1921c: 9b 06 27 00  	addiw	a3, a4, 2
   19220: 41 06        	addi	a2, a2, 16
   19222: 2e 87        	mv	a4, a1
   19224: 41 34        	addiw	s0, s0, -16
   19226: 63 df 8f 02  	bge	t6, s0, 0x19264 <_vfiprintf_r+0x6d6>
   1922a: 9b 05 17 00  	addiw	a1, a4, 1
   1922e: c1 07        	addi	a5, a5, 16
   19230: 04 e2        	sd	s1, 0(a2)
   19232: 23 34 f6 01  	sd	t6, 8(a2)
   19236: 3e e1        	sd	a5, 128(sp)
   19238: ae dc        	sw	a1, 120(sp)
   1923a: e3 d1 b2 fe  	bge	t0, a1, 0x1921c <_vfiprintf_r+0x68e>
   1923e: 63 83 07 10  	beqz	a5, 0x19344 <_vfiprintf_r+0x7b6>
   19242: 90 18        	addi	a2, sp, 112
   19244: d2 85        	mv	a1, s4
   19246: 56 85        	mv	a0, s5
   19248: ef f0 5f 89  	jal	0x18adc <__sprint_r.part.0>
   1924c: e3 10 05 ac  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19250: 66 57        	lw	a4, 120(sp)
   19252: c1 4f        	addi	t6, zero, 16
   19254: 41 34        	addiw	s0, s0, -16
   19256: 8a 67        	ld	a5, 128(sp)
   19258: 4e 86        	mv	a2, s3
   1925a: 9b 06 17 00  	addiw	a3, a4, 1
   1925e: 9d 42        	addi	t0, zero, 7
   19260: e3 c5 8f fc  	blt	t6, s0, 0x1922a <_vfiprintf_r+0x69c>
   19264: 26 8f        	mv	t5, s1
   19266: 82 78        	ld	a7, 32(sp)
   19268: 22 7e        	ld	t3, 40(sp)
   1926a: 26 63        	ld	t1, 72(sp)
   1926c: e2 64        	ld	s1, 24(sp)
   1926e: a2 8e        	mv	t4, s0
   19270: 32 84        	mv	s0, a2
   19272: f6 97        	add	a5, a5, t4
   19274: 23 30 e4 01  	sd	t5, 0(s0)
   19278: 23 34 d4 01  	sd	t4, 8(s0)
   1927c: 3e e1        	sd	a5, 128(sp)
   1927e: b6 dc        	sw	a3, 120(sp)
   19280: 1d 47        	addi	a4, zero, 7
   19282: 63 54 d7 38  	bge	a4, a3, 0x1960a <_vfiprintf_r+0xa7c>
   19286: 63 8e 07 52  	beqz	a5, 0x197c2 <_vfiprintf_r+0xc34>
   1928a: 90 18        	addi	a2, sp, 112
   1928c: d2 85        	mv	a1, s4
   1928e: 56 85        	mv	a0, s5
   19290: 1a f4        	sd	t1, 40(sp)
   19292: 72 f0        	sd	t3, 32(sp)
   19294: 46 ec        	sd	a7, 24(sp)
   19296: ef f0 7f 84  	jal	0x18adc <__sprint_r.part.0>
   1929a: e3 19 05 a6  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   1929e: 22 73        	ld	t1, 40(sp)
   192a0: 66 57        	lw	a4, 120(sp)
   192a2: 8a 67        	ld	a5, 128(sp)
   192a4: 3b 0d 6d 40  	subw	s10, s10, t1
   192a8: 02 7e        	ld	t3, 32(sp)
   192aa: e2 68        	ld	a7, 24(sp)
   192ac: 4e 84        	mv	s0, s3
   192ae: 9b 05 17 00  	addiw	a1, a4, 1
   192b2: e3 55 a0 b7  	blez	s10, 0x18e1c <_vfiprintf_r+0x28e>
   192b6: 79 6f        	lui	t5, 30
   192b8: 13 0f 0f 24  	addi	t5, t5, 576
   192bc: 63 53 a9 07  	bge	s2, s10, 0x19322 <_vfiprintf_r+0x794>
   192c0: a2 86        	mv	a3, s0
   192c2: 9d 4e        	addi	t4, zero, 7
   192c4: 6a 84        	mv	s0, s10
   192c6: 46 ec        	sd	a7, 24(sp)
   192c8: 72 f0        	sd	t3, 32(sp)
   192ca: 1a f4        	sd	t1, 40(sp)
   192cc: 5a 8d        	mv	s10, s6
   192ce: 01 a8        	j	0x192de <_vfiprintf_r+0x750>
   192d0: 9b 05 27 00  	addiw	a1, a4, 2
   192d4: c1 06        	addi	a3, a3, 16
   192d6: 2a 87        	mv	a4, a0
   192d8: 41 34        	addiw	s0, s0, -16
   192da: 63 5e 89 02  	bge	s2, s0, 0x19316 <_vfiprintf_r+0x788>
   192de: 1b 05 17 00  	addiw	a0, a4, 1
   192e2: c1 07        	addi	a5, a5, 16
   192e4: 23 b0 66 01  	sd	s6, 0(a3)
   192e8: 23 b4 26 01  	sd	s2, 8(a3)
   192ec: 3e e1        	sd	a5, 128(sp)
   192ee: aa dc        	sw	a0, 120(sp)
   192f0: e3 d0 ae fe  	bge	t4, a0, 0x192d0 <_vfiprintf_r+0x742>
   192f4: a1 c7        	beqz	a5, 0x1933c <_vfiprintf_r+0x7ae>
   192f6: 90 18        	addi	a2, sp, 112
   192f8: d2 85        	mv	a1, s4
   192fa: 56 85        	mv	a0, s5
   192fc: ef f0 0f fe  	jal	0x18adc <__sprint_r.part.0>
   19300: e3 16 05 a0  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19304: 66 57        	lw	a4, 120(sp)
   19306: 41 34        	addiw	s0, s0, -16
   19308: 8a 67        	ld	a5, 128(sp)
   1930a: ce 86        	mv	a3, s3
   1930c: 9b 05 17 00  	addiw	a1, a4, 1
   19310: 9d 4e        	addi	t4, zero, 7
   19312: e3 46 89 fc  	blt	s2, s0, 0x192de <_vfiprintf_r+0x750>
   19316: e2 68        	ld	a7, 24(sp)
   19318: 02 7e        	ld	t3, 32(sp)
   1931a: 22 73        	ld	t1, 40(sp)
   1931c: 6a 8f        	mv	t5, s10
   1931e: 22 8d        	mv	s10, s0
   19320: 36 84        	mv	s0, a3
   19322: ea 97        	add	a5, a5, s10
   19324: 23 30 e4 01  	sd	t5, 0(s0)
   19328: 23 34 a4 01  	sd	s10, 8(s0)
   1932c: 3e e1        	sd	a5, 128(sp)
   1932e: ae dc        	sw	a1, 120(sp)
   19330: 1d 47        	addi	a4, zero, 7
   19332: 63 4f b7 1e  	blt	a4, a1, 0x19530 <_vfiprintf_r+0x9a2>
   19336: 41 04        	addi	s0, s0, 16
   19338: 85 25        	addiw	a1, a1, 1
   1933a: cd b4        	j	0x18e1c <_vfiprintf_r+0x28e>
   1933c: 85 45        	addi	a1, zero, 1
   1933e: 01 47        	mv	a4, zero
   19340: ce 86        	mv	a3, s3
   19342: 59 bf        	j	0x192d8 <_vfiprintf_r+0x74a>
   19344: 85 46        	addi	a3, zero, 1
   19346: 01 47        	mv	a4, zero
   19348: 4e 86        	mv	a2, s3
   1934a: e9 bd        	j	0x19224 <_vfiprintf_r+0x696>
   1934c: 85 cb        	beqz	a5, 0x1937c <_vfiprintf_r+0x7ee>
   1934e: 90 18        	addi	a2, sp, 112
   19350: d2 85        	mv	a1, s4
   19352: 56 85        	mv	a0, s5
   19354: 9a e8        	sd	t1, 80(sp)
   19356: f2 e4        	sd	t3, 72(sp)
   19358: 46 f4        	sd	a7, 40(sp)
   1935a: 16 f0        	sd	t0, 32(sp)
   1935c: 7e ec        	sd	t6, 24(sp)
   1935e: ef f0 ef f7  	jal	0x18adc <__sprint_r.part.0>
   19362: e3 15 05 9a  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19366: 66 57        	lw	a4, 120(sp)
   19368: 8a 67        	ld	a5, 128(sp)
   1936a: 46 63        	ld	t1, 80(sp)
   1936c: 26 6e        	ld	t3, 72(sp)
   1936e: a2 78        	ld	a7, 40(sp)
   19370: 82 72        	ld	t0, 32(sp)
   19372: e2 6f        	ld	t6, 24(sp)
   19374: 4e 84        	mv	s0, s3
   19376: 9b 05 17 00  	addiw	a1, a4, 1
   1937a: 81 bc        	j	0x18dca <_vfiprintf_r+0x23c>
   1937c: 63 8d 0f 2c  	beqz	t6, 0x19656 <_vfiprintf_r+0xac8>
   19380: bc 10        	addi	a5, sp, 104
   19382: be f9        	sd	a5, 240(sp)
   19384: 89 47        	addi	a5, zero, 2
   19386: be fd        	sd	a5, 248(sp)
   19388: 85 45        	addi	a1, zero, 1
   1938a: 4e 84        	mv	s0, s3
   1938c: 2e 87        	mv	a4, a1
   1938e: 41 04        	addi	s0, s0, 16
   19390: 85 25        	addiw	a1, a1, 1
   19392: ad bc        	j	0x18e0c <_vfiprintf_r+0x27e>
   19394: be 8b        	mv	s7, a5
   19396: 89 47        	addi	a5, zero, 2
   19398: e3 8a 07 d8  	beqz	a5, 0x1912c <_vfiprintf_r+0x59e>
   1939c: c2 66        	ld	a3, 16(sp)
   1939e: 13 03 c1 0e  	addi	t1, sp, 236
   193a2: 9a 84        	mv	s1, t1
   193a4: 93 77 f7 00  	andi	a5, a4, 15
   193a8: b6 97        	add	a5, a5, a3
   193aa: 83 c7 07 00  	lbu	a5, 0(a5)
   193ae: 11 83        	srli	a4, a4, 4
   193b0: fd 14        	addi	s1, s1, -1
   193b2: 23 80 f4 00  	sb	a5, 0(s1)
   193b6: 7d f7        	bnez	a4, 0x193a4 <_vfiprintf_r+0x816>
   193b8: 3b 03 93 40  	subw	t1, t1, s1
   193bc: 1b 0e 03 00  	sext.w	t3, t1
   193c0: e3 5d a3 b7  	bge	t1, s10, 0x18f3a <_vfiprintf_r+0x3ac>
   193c4: 1b 0e 0d 00  	sext.w	t3, s10
   193c8: 8d be        	j	0x18f3a <_vfiprintf_r+0x3ac>
   193ca: 41 45        	addi	a0, zero, 16
   193cc: 63 5d d5 42  	bge	a0, a3, 0x19806 <_vfiprintf_r+0xc78>
   193d0: f9 6e        	lui	t4, 30
   193d2: 93 8e 0e 25  	addi	t4, t4, 592
   193d6: 26 f4        	sd	s1, 40(sp)
   193d8: a2 85        	mv	a1, s0
   193da: 41 4f        	addi	t5, zero, 16
   193dc: 9d 43        	addi	t2, zero, 7
   193de: 7e ec        	sd	t6, 24(sp)
   193e0: 16 f0        	sd	t0, 32(sp)
   193e2: c6 e4        	sd	a7, 72(sp)
   193e4: f2 e8        	sd	t3, 80(sp)
   193e6: 9a ec        	sd	t1, 88(sp)
   193e8: 36 84        	mv	s0, a3
   193ea: f6 84        	mv	s1, t4
   193ec: 11 a8        	j	0x19400 <_vfiprintf_r+0x872>
   193ee: 1b 06 27 00  	addiw	a2, a4, 2
   193f2: c1 05        	addi	a1, a1, 16
   193f4: 2a 87        	mv	a4, a0
   193f6: 41 34        	addiw	s0, s0, -16
   193f8: 63 50 8f 04  	bge	t5, s0, 0x19438 <_vfiprintf_r+0x8aa>
   193fc: 1b 06 17 00  	addiw	a2, a4, 1
   19400: c1 07        	addi	a5, a5, 16
   19402: 84 e1        	sd	s1, 0(a1)
   19404: 23 b4 e5 01  	sd	t5, 8(a1)
   19408: 3e e1        	sd	a5, 128(sp)
   1940a: 1b 05 06 00  	sext.w	a0, a2
   1940e: b2 dc        	sw	a2, 120(sp)
   19410: e3 df a3 fc  	bge	t2, a0, 0x193ee <_vfiprintf_r+0x860>
   19414: a9 cb        	beqz	a5, 0x19466 <_vfiprintf_r+0x8d8>
   19416: 90 18        	addi	a2, sp, 112
   19418: d2 85        	mv	a1, s4
   1941a: 56 85        	mv	a0, s5
   1941c: ef f0 0f ec  	jal	0x18adc <__sprint_r.part.0>
   19420: e3 16 05 8e  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19424: 66 57        	lw	a4, 120(sp)
   19426: 41 4f        	addi	t5, zero, 16
   19428: 41 34        	addiw	s0, s0, -16
   1942a: 8a 67        	ld	a5, 128(sp)
   1942c: ce 85        	mv	a1, s3
   1942e: 1b 06 17 00  	addiw	a2, a4, 1
   19432: 9d 43        	addi	t2, zero, 7
   19434: e3 44 8f fc  	blt	t5, s0, 0x193fc <_vfiprintf_r+0x86e>
   19438: a6 8e        	mv	t4, s1
   1943a: e2 6f        	ld	t6, 24(sp)
   1943c: 82 72        	ld	t0, 32(sp)
   1943e: a6 68        	ld	a7, 72(sp)
   19440: 46 6e        	ld	t3, 80(sp)
   19442: 66 63        	ld	t1, 88(sp)
   19444: a2 74        	ld	s1, 40(sp)
   19446: a2 86        	mv	a3, s0
   19448: 2e 84        	mv	s0, a1
   1944a: b6 97        	add	a5, a5, a3
   1944c: 23 30 d4 01  	sd	t4, 0(s0)
   19450: 14 e4        	sd	a3, 8(s0)
   19452: 3e e1        	sd	a5, 128(sp)
   19454: b2 dc        	sw	a2, 120(sp)
   19456: 1d 47        	addi	a4, zero, 7
   19458: 63 48 c7 10  	blt	a4, a2, 0x19568 <_vfiprintf_r+0x9da>
   1945c: 41 04        	addi	s0, s0, 16
   1945e: 9b 05 16 00  	addiw	a1, a2, 1
   19462: 32 87        	mv	a4, a2
   19464: 91 b2        	j	0x18da8 <_vfiprintf_r+0x21a>
   19466: 01 47        	mv	a4, zero
   19468: 05 46        	addi	a2, zero, 1
   1946a: ce 85        	mv	a1, s3
   1946c: 69 b7        	j	0x193f6 <_vfiprintf_r+0x868>
   1946e: 82 dc        	sw	zero, 120(sp)
   19470: 93 fb 4b 00  	andi	s7, s7, 4
   19474: 63 87 0b 0a  	beqz	s7, 0x19522 <_vfiprintf_r+0x994>
   19478: bb 84 c8 41  	subw	s1, a7, t3
   1947c: 63 53 90 0a  	blez	s1, 0x19522 <_vfiprintf_r+0x994>
   19480: 4e 84        	mv	s0, s3
   19482: 41 47        	addi	a4, zero, 16
   19484: e6 56        	lw	a3, 120(sp)
   19486: 63 59 97 38  	bge	a4, s1, 0x19818 <_vfiprintf_r+0xc8a>
   1948a: f9 6e        	lui	t4, 30
   1948c: 93 8e 0e 25  	addi	t4, t4, 592
   19490: 22 86        	mv	a2, s0
   19492: c1 4b        	addi	s7, zero, 16
   19494: 26 84        	mv	s0, s1
   19496: 1d 4d        	addi	s10, zero, 7
   19498: 46 ec        	sd	a7, 24(sp)
   1949a: 72 f0        	sd	t3, 32(sp)
   1949c: f6 84        	mv	s1, t4
   1949e: 01 a8        	j	0x194ae <_vfiprintf_r+0x920>
   194a0: 1b 87 26 00  	addiw	a4, a3, 2
   194a4: 41 06        	addi	a2, a2, 16
   194a6: ae 86        	mv	a3, a1
   194a8: 41 34        	addiw	s0, s0, -16
   194aa: 63 dc 8b 02  	bge	s7, s0, 0x194e2 <_vfiprintf_r+0x954>
   194ae: 9b 85 16 00  	addiw	a1, a3, 1
   194b2: c1 07        	addi	a5, a5, 16
   194b4: 04 e2        	sd	s1, 0(a2)
   194b6: 23 34 76 01  	sd	s7, 8(a2)
   194ba: 3e e1        	sd	a5, 128(sp)
   194bc: ae dc        	sw	a1, 120(sp)
   194be: e3 51 bd fe  	bge	s10, a1, 0x194a0 <_vfiprintf_r+0x912>
   194c2: a1 cf        	beqz	a5, 0x1951a <_vfiprintf_r+0x98c>
   194c4: 90 18        	addi	a2, sp, 112
   194c6: d2 85        	mv	a1, s4
   194c8: 56 85        	mv	a0, s5
   194ca: ef f0 2f e1  	jal	0x18adc <__sprint_r.part.0>
   194ce: e3 1f 05 82  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   194d2: e6 56        	lw	a3, 120(sp)
   194d4: 41 34        	addiw	s0, s0, -16
   194d6: 8a 67        	ld	a5, 128(sp)
   194d8: 4e 86        	mv	a2, s3
   194da: 1b 87 16 00  	addiw	a4, a3, 1
   194de: e3 c8 8b fc  	blt	s7, s0, 0x194ae <_vfiprintf_r+0x920>
   194e2: e2 68        	ld	a7, 24(sp)
   194e4: 02 7e        	ld	t3, 32(sp)
   194e6: a6 8e        	mv	t4, s1
   194e8: a2 84        	mv	s1, s0
   194ea: 32 84        	mv	s0, a2
   194ec: a6 97        	add	a5, a5, s1
   194ee: 23 30 d4 01  	sd	t4, 0(s0)
   194f2: 04 e4        	sd	s1, 8(s0)
   194f4: 3e e1        	sd	a5, 128(sp)
   194f6: ba dc        	sw	a4, 120(sp)
   194f8: 9d 46        	addi	a3, zero, 7
   194fa: e3 d1 e6 96  	bge	a3, a4, 0x18e5c <_vfiprintf_r+0x2ce>
   194fe: 95 c3        	beqz	a5, 0x19522 <_vfiprintf_r+0x994>
   19500: 90 18        	addi	a2, sp, 112
   19502: d2 85        	mv	a1, s4
   19504: 56 85        	mv	a0, s5
   19506: 72 f0        	sd	t3, 32(sp)
   19508: 46 ec        	sd	a7, 24(sp)
   1950a: ef f0 2f dd  	jal	0x18adc <__sprint_r.part.0>
   1950e: 63 1f 05 fe  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19512: 8a 67        	ld	a5, 128(sp)
   19514: 02 7e        	ld	t3, 32(sp)
   19516: e2 68        	ld	a7, 24(sp)
   19518: 91 b2        	j	0x18e5c <_vfiprintf_r+0x2ce>
   1951a: 05 47        	addi	a4, zero, 1
   1951c: 81 46        	mv	a3, zero
   1951e: 4e 86        	mv	a2, s3
   19520: 61 b7        	j	0x194a8 <_vfiprintf_r+0x91a>
   19522: c6 87        	mv	a5, a7
   19524: 63 d3 c8 01  	bge	a7, t3, 0x1952a <_vfiprintf_r+0x99c>
   19528: f2 87        	mv	a5, t3
   1952a: bb 8c 97 01  	addw	s9, a5, s9
   1952e: 3d ba        	j	0x18e6c <_vfiprintf_r+0x2de>
   19530: e3 89 07 ca  	beqz	a5, 0x191e2 <_vfiprintf_r+0x654>
   19534: 90 18        	addi	a2, sp, 112
   19536: d2 85        	mv	a1, s4
   19538: 56 85        	mv	a0, s5
   1953a: 1a f4        	sd	t1, 40(sp)
   1953c: 72 f0        	sd	t3, 32(sp)
   1953e: 46 ec        	sd	a7, 24(sp)
   19540: ef f0 cf d9  	jal	0x18adc <__sprint_r.part.0>
   19544: 63 14 05 fc  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19548: e6 55        	lw	a1, 120(sp)
   1954a: 8a 67        	ld	a5, 128(sp)
   1954c: 22 73        	ld	t1, 40(sp)
   1954e: 02 7e        	ld	t3, 32(sp)
   19550: e2 68        	ld	a7, 24(sp)
   19552: 4e 84        	mv	s0, s3
   19554: 85 25        	addiw	a1, a1, 1
   19556: d9 b0        	j	0x18e1c <_vfiprintf_r+0x28e>
   19558: 82 dc        	sw	zero, 120(sp)
   1955a: 4e 84        	mv	s0, s3
   1955c: 6f f0 8f f1  	j	0x18c74 <_vfiprintf_r+0xe6>
   19560: 85 45        	addi	a1, zero, 1
   19562: 01 47        	mv	a4, zero
   19564: 4e 84        	mv	s0, s3
   19566: 5d b0        	j	0x18e0c <_vfiprintf_r+0x27e>
   19568: 63 84 07 18  	beqz	a5, 0x196f0 <_vfiprintf_r+0xb62>
   1956c: 90 18        	addi	a2, sp, 112
   1956e: d2 85        	mv	a1, s4
   19570: 56 85        	mv	a0, s5
   19572: 9a e8        	sd	t1, 80(sp)
   19574: f2 e4        	sd	t3, 72(sp)
   19576: 46 f4        	sd	a7, 40(sp)
   19578: 16 f0        	sd	t0, 32(sp)
   1957a: 7e ec        	sd	t6, 24(sp)
   1957c: ef f0 0f d6  	jal	0x18adc <__sprint_r.part.0>
   19580: 63 16 05 f8  	bnez	a0, 0x18d0c <_vfiprintf_r+0x17e>
   19584: 66 57        	lw	a4, 120(sp)
   19586: 8a 67        	ld	a5, 128(sp)
   19588: 46 63        	ld	t1, 80(sp)
   1958a: 26 6e        	ld	t3, 72(sp)
   1958c: a2 78        	ld	a7, 40(sp)
   1958e: 82 72        	ld	t0, 32(sp)
   19590: e2 6f        	ld	t6, 24(sp)
   19592: 4e 84        	mv	s0, s3
   19594: 9b 05 17 00  	addiw	a1, a4, 1
   19598: 6f f0 1f 81  	j	0x18da8 <_vfiprintf_r+0x21a>
   1959c: 01 4e        	mv	t3, zero
   1959e: 01 43        	mv	t1, zero
   195a0: e4 11        	addi	s1, sp, 236
   195a2: 61 ba        	j	0x18f3a <_vfiprintf_r+0x3ac>
   195a4: 13 03 c1 0e  	addi	t1, sp, 236
   195a8: 62 78        	ld	a6, 56(sp)
   195aa: 13 ff 0b 40  	andi	t5, s7, 1024
   195ae: 01 46        	mv	a2, zero
   195b0: 1a 87        	mv	a4, t1
   195b2: a9 46        	addi	a3, zero, 10
   195b4: 25 4e        	addi	t3, zero, 9
   195b6: 93 0f f0 0f  	addi	t6, zero, 255
   195ba: 39 a0        	j	0x195c8 <_vfiprintf_r+0xa3a>
   195bc: b3 57 dc 02  	divu	a5, s8, a3
   195c0: 63 77 8e 03  	bgeu	t3, s8, 0x195ee <_vfiprintf_r+0xa60>
   195c4: 3e 8c        	mv	s8, a5
   195c6: 26 87        	mv	a4, s1
   195c8: 93 04 f7 ff  	addi	s1, a4, -1
   195cc: 05 26        	addiw	a2, a2, 1
   195ce: b3 77 dc 02  	remu	a5, s8, a3
   195d2: 9b 87 07 03  	addiw	a5, a5, 48
   195d6: a3 0f f7 fe  	sb	a5, -1(a4)
   195da: e3 01 0f fe  	beqz	t5, 0x195bc <_vfiprintf_r+0xa2e>
   195de: 83 47 08 00  	lbu	a5, 0(a6)
   195e2: e3 9d c7 fc  	bne	a5, a2, 0x195bc <_vfiprintf_r+0xa2e>
   195e6: e3 8b f7 fd  	beq	a5, t6, 0x195bc <_vfiprintf_r+0xa2e>
   195ea: 63 6f 8e 11  	bltu	t3, s8, 0x19708 <_vfiprintf_r+0xb7a>
   195ee: 3b 03 93 40  	subw	t1, t1, s1
   195f2: 42 fc        	sd	a6, 56(sp)
   195f4: 1b 0e 03 00  	sext.w	t3, t1
   195f8: 63 54 a3 01  	bge	t1, s10, 0x19600 <_vfiprintf_r+0xa72>
   195fc: 1b 0e 0d 00  	sext.w	t3, s10
   19600: 76 8c        	mv	s8, t4
   19602: 25 ba        	j	0x18f3a <_vfiprintf_r+0x3ac>
   19604: 01 4e        	mv	t3, zero
   19606: e4 11        	addi	s1, sp, 236
   19608: 0d ba        	j	0x18f3a <_vfiprintf_r+0x3ac>
   1960a: 41 04        	addi	s0, s0, 16
   1960c: 9b 85 16 00  	addiw	a1, a3, 1
   19610: 36 87        	mv	a4, a3
   19612: 6f f0 3f 80  	j	0x18e14 <_vfiprintf_r+0x286>
   19616: 93 77 03 04  	andi	a5, t1, 64
   1961a: 03 2c 0c 00  	lw	s8, 0(s8)
   1961e: 63 8c 07 16  	beqz	a5, 0x19796 <_vfiprintf_r+0xc08>
   19622: 42 1c        	slli	s8, s8, 48
   19624: 13 5c 0c 03  	srli	s8, s8, 48
   19628: e1 bc        	j	0x19100 <_vfiprintf_r+0x572>
   1962a: 93 f6 07 04  	andi	a3, a5, 64
   1962e: 03 27 0c 00  	lw	a4, 0(s8)
   19632: 63 88 06 16  	beqz	a3, 0x197a2 <_vfiprintf_r+0xc14>
   19636: 42 17        	slli	a4, a4, 48
   19638: 41 93        	srli	a4, a4, 48
   1963a: 89 b4        	j	0x1907c <_vfiprintf_r+0x4ee>
   1963c: 83 c7 0d 00  	lbu	a5, 0(s11)
   19640: 63 0b 07 e4  	beqz	a4, 0x18c96 <_vfiprintf_r+0x108>
   19644: 03 47 07 00  	lbu	a4, 0(a4)
   19648: 63 07 07 e4  	beqz	a4, 0x18c96 <_vfiprintf_r+0x108>
   1964c: 93 eb 0b 40  	ori	s7, s7, 1024
   19650: 81 2b        	sext.w	s7, s7
   19652: 6f f0 4f e4  	j	0x18c96 <_vfiprintf_r+0x108>
   19656: 01 47        	mv	a4, zero
   19658: 85 45        	addi	a1, zero, 1
   1965a: 4e 84        	mv	s0, s3
   1965c: 6f f0 0f fb  	j	0x18e0c <_vfiprintf_r+0x27e>
   19660: 1b 0e 0d 00  	sext.w	t3, s10
   19664: b6 84        	mv	s1, a3
   19666: d1 b8        	j	0x18f3a <_vfiprintf_r+0x3ac>
   19668: de 87        	mv	a5, s7
   1966a: f5 ba        	j	0x19066 <_vfiprintf_r+0x4d8>
   1966c: 75 67        	lui	a4, 29
   1966e: 13 07 07 76  	addi	a4, a4, 1888
   19672: 3a e8        	sd	a4, 16(sp)
   19674: 13 f7 0b 02  	andi	a4, s7, 32
   19678: 93 06 8c 00  	addi	a3, s8, 8
   1967c: 29 eb        	bnez	a4, 0x196ce <_vfiprintf_r+0xb40>
   1967e: 13 f7 0b 01  	andi	a4, s7, 16
   19682: 31 e7        	bnez	a4, 0x196ce <_vfiprintf_r+0xb40>
   19684: 13 f6 0b 04  	andi	a2, s7, 64
   19688: 03 27 0c 00  	lw	a4, 0(s8)
   1968c: 63 0f 06 0e  	beqz	a2, 0x1978a <_vfiprintf_r+0xbfc>
   19690: 42 17        	slli	a4, a4, 48
   19692: 41 93        	srli	a4, a4, 48
   19694: 13 f6 1b 00  	andi	a2, s7, 1
   19698: 19 ca        	beqz	a2, 0x196ae <_vfiprintf_r+0xb20>
   1969a: 11 cb        	beqz	a4, 0x196ae <_vfiprintf_r+0xb20>
   1969c: 93 eb 2b 00  	ori	s7, s7, 2
   196a0: 13 06 00 03  	addi	a2, zero, 48
   196a4: 23 04 c1 06  	sb	a2, 104(sp)
   196a8: a3 04 f1 06  	sb	a5, 105(sp)
   196ac: 81 2b        	sext.w	s7, s7
   196ae: 93 fb fb bf  	andi	s7, s7, -1025
   196b2: 81 2b        	sext.w	s7, s7
   196b4: 36 8c        	mv	s8, a3
   196b6: c5 b4        	j	0x19196 <_vfiprintf_r+0x608>
   196b8: 5e 83        	mv	t1, s7
   196ba: 05 bc        	j	0x190ea <_vfiprintf_r+0x55c>
   196bc: 75 67        	lui	a4, 29
   196be: 13 07 87 77  	addi	a4, a4, 1912
   196c2: 3a e8        	sd	a4, 16(sp)
   196c4: 13 f7 0b 02  	andi	a4, s7, 32
   196c8: 93 06 8c 00  	addi	a3, s8, 8
   196cc: 4d db        	beqz	a4, 0x1967e <_vfiprintf_r+0xaf0>
   196ce: 03 37 0c 00  	ld	a4, 0(s8)
   196d2: c9 b7        	j	0x19694 <_vfiprintf_r+0xb06>
   196d4: 26 85        	mv	a0, s1
   196d6: ef d0 af fa  	jal	0x16e80 <strlen>
   196da: 1b 03 05 00  	sext.w	t1, a0
   196de: 13 4e f3 ff  	not	t3, t1
   196e2: 13 5e fe 43  	srai	t3, t3, 63
   196e6: e2 68        	ld	a7, 24(sp)
   196e8: 33 7e 6e 00  	and	t3, t3, t1
   196ec: 01 4d        	mv	s10, zero
   196ee: b1 b0        	j	0x18f3a <_vfiprintf_r+0x3ac>
   196f0: 03 47 71 06  	lbu	a4, 103(sp)
   196f4: 71 cf        	beqz	a4, 0x197d0 <_vfiprintf_r+0xc42>
   196f6: 93 07 71 06  	addi	a5, sp, 103
   196fa: be f9        	sd	a5, 240(sp)
   196fc: 85 47        	addi	a5, zero, 1
   196fe: be fd        	sd	a5, 248(sp)
   19700: 85 45        	addi	a1, zero, 1
   19702: 4e 84        	mv	s0, s3
   19704: 6f f0 0f ec  	j	0x18dc4 <_vfiprintf_r+0x236>
   19708: c2 77        	ld	a5, 48(sp)
   1970a: 86 65        	ld	a1, 64(sp)
   1970c: 9a e4        	sd	t1, 72(sp)
   1970e: 9d 8c        	sub	s1, s1, a5
   19710: 3e 86        	mv	a2, a5
   19712: 26 85        	mv	a0, s1
   19714: 76 fc        	sd	t4, 56(sp)
   19716: 7a f4        	sd	t5, 40(sp)
   19718: 46 f0        	sd	a7, 32(sp)
   1971a: 42 ec        	sd	a6, 24(sp)
   1971c: ef d0 4f ff  	jal	0x16f10 <strncpy>
   19720: 62 68        	ld	a6, 24(sp)
   19722: a9 47        	addi	a5, zero, 10
   19724: b3 57 fc 02  	divu	a5, s8, a5
   19728: 03 47 18 00  	lbu	a4, 1(a6)
   1972c: 82 78        	ld	a7, 32(sp)
   1972e: 22 7f        	ld	t5, 40(sp)
   19730: 33 37 e0 00  	snez	a4, a4
   19734: e2 7e        	ld	t4, 56(sp)
   19736: 26 63        	ld	t1, 72(sp)
   19738: 3a 98        	add	a6, a6, a4
   1973a: 01 46        	mv	a2, zero
   1973c: a9 46        	addi	a3, zero, 10
   1973e: 25 4e        	addi	t3, zero, 9
   19740: 93 0f f0 0f  	addi	t6, zero, 255
   19744: 41 b5        	j	0x195c4 <_vfiprintf_r+0xa36>
   19746: be 8b        	mv	s7, a5
   19748: 81 47        	mv	a5, zero
   1974a: e3 81 07 9e  	beqz	a5, 0x1912c <_vfiprintf_r+0x59e>
   1974e: b9 b1        	j	0x1939c <_vfiprintf_r+0x80e>
   19750: 93 eb 0b 20  	ori	s7, s7, 512
   19754: 83 c7 1d 00  	lbu	a5, 1(s11)
   19758: 81 2b        	sext.w	s7, s7
   1975a: 85 0d        	addi	s11, s11, 1
   1975c: 6f f0 af d3  	j	0x18c96 <_vfiprintf_r+0x108>
   19760: 93 eb 0b 02  	ori	s7, s7, 32
   19764: 83 c7 1d 00  	lbu	a5, 1(s11)
   19768: 81 2b        	sext.w	s7, s7
   1976a: 85 0d        	addi	s11, s11, 1
   1976c: 6f f0 af d2  	j	0x18c96 <_vfiprintf_r+0x108>
   19770: 99 47        	addi	a5, zero, 6
   19772: 1b 03 0d 00  	sext.w	t1, s10
   19776: 63 eb a7 05  	bltu	a5, s10, 0x197cc <_vfiprintf_r+0xc3e>
   1977a: f5 64        	lui	s1, 29
   1977c: 1a 8e        	mv	t3, t1
   1977e: 93 84 04 79  	addi	s1, s1, 1936
   19782: 6f f0 0f e0  	j	0x18d82 <_vfiprintf_r+0x1f4>
   19786: 01 47        	mv	a4, zero
   19788: 11 b9        	j	0x1939c <_vfiprintf_r+0x80e>
   1978a: 13 f6 0b 20  	andi	a2, s7, 512
   1978e: 25 c6        	beqz	a2, 0x197f6 <_vfiprintf_r+0xc68>
   19790: 13 77 f7 0f  	andi	a4, a4, 255
   19794: 01 b7        	j	0x19694 <_vfiprintf_r+0xb06>
   19796: 93 77 03 20  	andi	a5, t1, 512
   1979a: b1 cb        	beqz	a5, 0x197ee <_vfiprintf_r+0xc60>
   1979c: 13 7c fc 0f  	andi	s8, s8, 255
   197a0: 85 b2        	j	0x19100 <_vfiprintf_r+0x572>
   197a2: 93 f6 07 20  	andi	a3, a5, 512
   197a6: a9 c2        	beqz	a3, 0x197e8 <_vfiprintf_r+0xc5a>
   197a8: 13 77 f7 0f  	andi	a4, a4, 255
   197ac: c1 b8        	j	0x1907c <_vfiprintf_r+0x4ee>
   197ae: 93 f7 0b 20  	andi	a5, s7, 512
   197b2: 85 cb        	beqz	a5, 0x197e2 <_vfiprintf_r+0xc54>
   197b4: 1b 1c 8c 01  	slliw	s8, s8, 24
   197b8: 1b 5c 8c 41  	sraiw	s8, s8, 24
   197bc: e2 87        	mv	a5, s8
   197be: 6f f0 0f ef  	j	0x18eae <_vfiprintf_r+0x320>
   197c2: 85 45        	addi	a1, zero, 1
   197c4: 01 47        	mv	a4, zero
   197c6: 4e 84        	mv	s0, s3
   197c8: 6f f0 cf e4  	j	0x18e14 <_vfiprintf_r+0x286>
   197cc: 19 43        	addi	t1, zero, 6
   197ce: 75 b7        	j	0x1977a <_vfiprintf_r+0xbec>
   197d0: 85 45        	addi	a1, zero, 1
   197d2: 4e 84        	mv	s0, s3
   197d4: 6f f0 6f df  	j	0x18dca <_vfiprintf_r+0x23c>
   197d8: 23 10 97 01  	sh	s9, 0(a4)
   197dc: 21 0c        	addi	s8, s8, 8
   197de: 6f f0 4f c4  	j	0x18c22 <_vfiprintf_r+0x94>
   197e2: e2 87        	mv	a5, s8
   197e4: 6f f0 af ec  	j	0x18eae <_vfiprintf_r+0x320>
   197e8: 02 17        	slli	a4, a4, 32
   197ea: 01 93        	srli	a4, a4, 32
   197ec: 41 b8        	j	0x1907c <_vfiprintf_r+0x4ee>
   197ee: 02 1c        	slli	s8, s8, 32
   197f0: 13 5c 0c 02  	srli	s8, s8, 32
   197f4: 31 b2        	j	0x19100 <_vfiprintf_r+0x572>
   197f6: 02 17        	slli	a4, a4, 32
   197f8: 01 93        	srli	a4, a4, 32
   197fa: 69 bd        	j	0x19694 <_vfiprintf_r+0xb06>
   197fc: 6a 8e        	mv	t3, s10
   197fe: 6a 83        	mv	t1, s10
   19800: 01 4d        	mv	s10, zero
   19802: 6f f0 8f f3  	j	0x18f3a <_vfiprintf_r+0x3ac>
   19806: f9 6e        	lui	t4, 30
   19808: 93 8e 0e 25  	addi	t4, t4, 592
   1980c: 3d b9        	j	0x1944a <_vfiprintf_r+0x8bc>
   1980e: 76 8c        	mv	s8, t4
   19810: 93 07 00 03  	addi	a5, zero, 48
   19814: 6f f0 2f f1  	j	0x18f26 <_vfiprintf_r+0x398>
   19818: f9 6e        	lui	t4, 30
   1981a: 1b 87 16 00  	addiw	a4, a3, 1
   1981e: 93 8e 0e 25  	addi	t4, t4, 592
   19822: e9 b1        	j	0x194ec <_vfiprintf_r+0x95e>
   19824: 79 6f        	lui	t5, 30
   19826: ae 86        	mv	a3, a1
   19828: 13 0f 0f 24  	addi	t5, t5, 576
   1982c: 99 b4        	j	0x19272 <_vfiprintf_r+0x6e4>
   1982e: 7d 5c        	addi	s8, zero, -1
   19830: 6f f0 af ce  	j	0x18d1a <_vfiprintf_r+0x18c>
   19834: 23 20 97 01  	sw	s9, 0(a4)
   19838: 21 0c        	addi	s8, s8, 8
   1983a: 6f f0 8f be  	j	0x18c22 <_vfiprintf_r+0x94>
   1983e: 83 27 0c 00  	lw	a5, 0(s8)
   19842: 21 0c        	addi	s8, s8, 8
   19844: 1b 8d 07 00  	sext.w	s10, a5
   19848: 63 d3 07 00  	bgez	a5, 0x1984e <_vfiprintf_r+0xcc0>
   1984c: 7d 5d        	addi	s10, zero, -1
   1984e: 83 c7 1d 00  	lbu	a5, 1(s11)
   19852: ba 8d        	mv	s11, a4
   19854: 6f f0 2f c4  	j	0x18c96 <_vfiprintf_r+0x108>

0000000000019858 <vfiprintf>:
   19858: aa 87        	mv	a5, a0
   1985a: 03 b5 81 85  	ld	a0, -1960(gp)
   1985e: b2 86        	mv	a3, a2
   19860: 2e 86        	mv	a2, a1
   19862: be 85        	mv	a1, a5
   19864: 6f f0 af b2  	j	0x18b8e <_vfiprintf_r>

0000000000019868 <__sbprintf>:
   19868: 83 d7 05 01  	lhu	a5, 16(a1)
   1986c: 03 ae c5 0a  	lw	t3, 172(a1)
   19870: 03 d3 25 01  	lhu	t1, 18(a1)
   19874: 83 b8 05 03  	ld	a7, 48(a1)
   19878: 03 b8 05 04  	ld	a6, 64(a1)
   1987c: 13 01 01 b3  	addi	sp, sp, -1232
   19880: f5 9b        	andi	a5, a5, -3
   19882: 13 07 00 40  	addi	a4, zero, 1024
   19886: 23 30 81 4c  	sd	s0, 1216(sp)
   1988a: 23 18 f1 00  	sh	a5, 16(sp)
   1988e: 2e 84        	mv	s0, a1
   19890: 1c 19        	addi	a5, sp, 176
   19892: 8a 85        	mv	a1, sp
   19894: 23 3c 91 4a  	sd	s1, 1208(sp)
   19898: 23 38 21 4b  	sd	s2, 1200(sp)
   1989c: 23 34 11 4c  	sd	ra, 1224(sp)
   198a0: 2a 89        	mv	s2, a0
   198a2: 72 d7        	sw	t3, 172(sp)
   198a4: 23 19 61 00  	sh	t1, 18(sp)
   198a8: 46 f8        	sd	a7, 48(sp)
   198aa: c2 e0        	sd	a6, 64(sp)
   198ac: 3e e0        	sd	a5, 0(sp)
   198ae: 3e ec        	sd	a5, 24(sp)
   198b0: 3a c6        	sw	a4, 12(sp)
   198b2: 3a d0        	sw	a4, 32(sp)
   198b4: 02 d4        	sw	zero, 40(sp)
   198b6: ef f0 8f ad  	jal	0x18b8e <_vfiprintf_r>
   198ba: aa 84        	mv	s1, a0
   198bc: 63 59 05 02  	bgez	a0, 0x198ee <__sbprintf+0x86>
   198c0: 83 57 01 01  	lhu	a5, 16(sp)
   198c4: 93 f7 07 04  	andi	a5, a5, 64
   198c8: 99 c7        	beqz	a5, 0x198d6 <__sbprintf+0x6e>
   198ca: 83 57 04 01  	lhu	a5, 16(s0)
   198ce: 93 e7 07 04  	ori	a5, a5, 64
   198d2: 23 18 f4 00  	sh	a5, 16(s0)
   198d6: 83 30 81 4c  	ld	ra, 1224(sp)
   198da: 03 34 01 4c  	ld	s0, 1216(sp)
   198de: 03 39 01 4b  	ld	s2, 1200(sp)
   198e2: 26 85        	mv	a0, s1
   198e4: 83 34 81 4b  	ld	s1, 1208(sp)
   198e8: 13 01 01 4d  	addi	sp, sp, 1232
   198ec: 82 80        	ret
   198ee: 8a 85        	mv	a1, sp
   198f0: 4a 85        	mv	a0, s2
   198f2: ef 80 df d9  	jal	0x1268e <_fflush_r>
   198f6: 69 d5        	beqz	a0, 0x198c0 <__sbprintf+0x58>
   198f8: fd 54        	addi	s1, zero, -1
   198fa: d9 b7        	j	0x198c0 <__sbprintf+0x58>

00000000000198fc <_wctomb_r>:
   198fc: b7 07 02 00  	lui	a5, 32
   19900: 83 b7 87 e7  	ld	a5, -392(a5)
   19904: 82 87        	jr	a5

0000000000019906 <__ascii_wctomb>:
   19906: 9b 07 06 00  	sext.w	a5, a2
   1990a: 91 cd        	beqz	a1, 0x19926 <__ascii_wctomb+0x20>
   1990c: 13 07 f0 0f  	addi	a4, zero, 255
   19910: 63 66 f7 00  	bltu	a4, a5, 0x1991c <__ascii_wctomb+0x16>
   19914: 23 80 c5 00  	sb	a2, 0(a1)
   19918: 05 45        	addi	a0, zero, 1
   1991a: 82 80        	ret
   1991c: 93 07 a0 08  	addi	a5, zero, 138
   19920: 1c c1        	sw	a5, 0(a0)
   19922: 7d 55        	addi	a0, zero, -1
   19924: 82 80        	ret
   19926: 01 45        	mv	a0, zero
   19928: 82 80        	ret

000000000001992a <_write_r>:
   1992a: 01 11        	addi	sp, sp, -32
   1992c: 2e 87        	mv	a4, a1
   1992e: 22 e8        	sd	s0, 16(sp)
   19930: 26 e4        	sd	s1, 8(sp)
   19932: b2 85        	mv	a1, a2
   19934: 2a 84        	mv	s0, a0
   19936: 36 86        	mv	a2, a3
   19938: 3a 85        	mv	a0, a4
   1993a: 06 ec        	sd	ra, 24(sp)
   1993c: 23 ac 01 88  	sw	zero, -1896(gp)
   19940: ef 20 60 1a  	jal	0x1bae6 <_write>
   19944: fd 57        	addi	a5, zero, -1
   19946: 63 07 f5 00  	beq	a0, a5, 0x19954 <_write_r+0x2a>
   1994a: e2 60        	ld	ra, 24(sp)
   1994c: 42 64        	ld	s0, 16(sp)
   1994e: a2 64        	ld	s1, 8(sp)
   19950: 05 61        	addi	sp, sp, 32
   19952: 82 80        	ret
   19954: 83 a7 81 89  	lw	a5, -1896(gp)
   19958: ed db        	beqz	a5, 0x1994a <_write_r+0x20>
   1995a: e2 60        	ld	ra, 24(sp)
   1995c: 1c c0        	sw	a5, 0(s0)
   1995e: 42 64        	ld	s0, 16(sp)
   19960: a2 64        	ld	s1, 8(sp)
   19962: 05 61        	addi	sp, sp, 32
   19964: 82 80        	ret

0000000000019966 <__assert_func>:
   19966: 03 b7 81 85  	ld	a4, -1960(gp)
   1996a: 41 11        	addi	sp, sp, -16
   1996c: b6 87        	mv	a5, a3
   1996e: 32 88        	mv	a6, a2
   19970: 06 e4        	sd	ra, 8(sp)
   19972: 83 38 87 01  	ld	a7, 24(a4)
   19976: aa 86        	mv	a3, a0
   19978: 2e 87        	mv	a4, a1
   1997a: 3e 86        	mv	a2, a5
   1997c: 63 0d 08 00  	beqz	a6, 0x19996 <__assert_func+0x30>
   19980: f9 67        	lui	a5, 30
   19982: 93 87 07 26  	addi	a5, a5, 608
   19986: f9 65        	lui	a1, 30
   19988: 93 85 05 27  	addi	a1, a1, 624
   1998c: 46 85        	mv	a0, a7
   1998e: ef 00 c0 1e  	jal	0x19b7a <fiprintf>
   19992: ef 10 90 39  	jal	0x1b52a <abort>
   19996: 75 68        	lui	a6, 29
   19998: 93 07 08 77  	addi	a5, a6, 1904
   1999c: 13 08 08 77  	addi	a6, a6, 1904
   199a0: dd b7        	j	0x19986 <__assert_func+0x20>

00000000000199a2 <__assert>:
   199a2: 41 11        	addi	sp, sp, -16
   199a4: b2 86        	mv	a3, a2
   199a6: 01 46        	mv	a2, zero
   199a8: 06 e4        	sd	ra, 8(sp)
   199aa: ef f0 df fb  	jal	0x19966 <__assert_func>

00000000000199ae <atexit>:
   199ae: aa 85        	mv	a1, a0
   199b0: 81 46        	mv	a3, zero
   199b2: 01 46        	mv	a2, zero
   199b4: 01 45        	mv	a0, zero
   199b6: 6f 10 50 38  	j	0x1b53a <__register_exitproc>

00000000000199ba <_calloc_r>:
   199ba: b3 b7 c5 02  	mulhu	a5, a1, a2
   199be: 41 11        	addi	sp, sp, -16
   199c0: 06 e4        	sd	ra, 8(sp)
   199c2: 22 e0        	sd	s0, 0(sp)
   199c4: b3 85 c5 02  	mul	a1, a1, a2
   199c8: b5 ef        	bnez	a5, 0x19a44 <_calloc_r+0x8a>
   199ca: ef b0 1f d5  	jal	0x1571a <_malloc_r>
   199ce: 2a 84        	mv	s0, a0
   199d0: 35 cd        	beqz	a0, 0x19a4c <_calloc_r+0x92>
   199d2: 03 36 85 ff  	ld	a2, -8(a0)
   199d6: 13 07 80 04  	addi	a4, zero, 72
   199da: 71 9a        	andi	a2, a2, -4
   199dc: 61 16        	addi	a2, a2, -8
   199de: 63 62 c7 04  	bltu	a4, a2, 0x19a22 <_calloc_r+0x68>
   199e2: 93 06 70 02  	addi	a3, zero, 39
   199e6: aa 87        	mv	a5, a0
   199e8: 63 f2 c6 02  	bgeu	a3, a2, 0x19a0c <_calloc_r+0x52>
   199ec: 23 30 05 00  	sd	zero, 0(a0)
   199f0: 23 34 05 00  	sd	zero, 8(a0)
   199f4: 93 07 70 03  	addi	a5, zero, 55
   199f8: 63 fd c7 02  	bgeu	a5, a2, 0x19a32 <_calloc_r+0x78>
   199fc: 23 38 05 00  	sd	zero, 16(a0)
   19a00: 23 3c 05 00  	sd	zero, 24(a0)
   19a04: 93 07 05 02  	addi	a5, a0, 32
   19a08: 63 08 e6 04  	beq	a2, a4, 0x19a58 <_calloc_r+0x9e>
   19a0c: 23 b0 07 00  	sd	zero, 0(a5)
   19a10: 23 b4 07 00  	sd	zero, 8(a5)
   19a14: 23 b8 07 00  	sd	zero, 16(a5)
   19a18: a2 60        	ld	ra, 8(sp)
   19a1a: 22 85        	mv	a0, s0
   19a1c: 02 64        	ld	s0, 0(sp)
   19a1e: 41 01        	addi	sp, sp, 16
   19a20: 82 80        	ret
   19a22: 81 45        	mv	a1, zero
   19a24: ef 60 df c3  	jal	0x10660 <memset>
   19a28: a2 60        	ld	ra, 8(sp)
   19a2a: 22 85        	mv	a0, s0
   19a2c: 02 64        	ld	s0, 0(sp)
   19a2e: 41 01        	addi	sp, sp, 16
   19a30: 82 80        	ret
   19a32: 93 07 05 01  	addi	a5, a0, 16
   19a36: 23 b0 07 00  	sd	zero, 0(a5)
   19a3a: 23 b4 07 00  	sd	zero, 8(a5)
   19a3e: 23 b8 07 00  	sd	zero, 16(a5)
   19a42: d9 bf        	j	0x19a18 <_calloc_r+0x5e>
   19a44: ef 00 80 05  	jal	0x19a9c <__errno>
   19a48: b1 47        	addi	a5, zero, 12
   19a4a: 1c c1        	sw	a5, 0(a0)
   19a4c: 01 44        	mv	s0, zero
   19a4e: a2 60        	ld	ra, 8(sp)
   19a50: 22 85        	mv	a0, s0
   19a52: 02 64        	ld	s0, 0(sp)
   19a54: 41 01        	addi	sp, sp, 16
   19a56: 82 80        	ret
   19a58: 23 30 05 02  	sd	zero, 32(a0)
   19a5c: 93 07 05 03  	addi	a5, a0, 48
   19a60: 23 34 05 02  	sd	zero, 40(a0)
   19a64: 65 b7        	j	0x19a0c <_calloc_r+0x52>

0000000000019a66 <_close_r>:
   19a66: 01 11        	addi	sp, sp, -32
   19a68: 22 e8        	sd	s0, 16(sp)
   19a6a: 26 e4        	sd	s1, 8(sp)
   19a6c: 2a 84        	mv	s0, a0
   19a6e: 2e 85        	mv	a0, a1
   19a70: 06 ec        	sd	ra, 24(sp)
   19a72: 23 ac 01 88  	sw	zero, -1896(gp)
   19a76: ef 10 90 6f  	jal	0x1b96e <_close>
   19a7a: fd 57        	addi	a5, zero, -1
   19a7c: 63 07 f5 00  	beq	a0, a5, 0x19a8a <_close_r+0x24>
   19a80: e2 60        	ld	ra, 24(sp)
   19a82: 42 64        	ld	s0, 16(sp)
   19a84: a2 64        	ld	s1, 8(sp)
   19a86: 05 61        	addi	sp, sp, 32
   19a88: 82 80        	ret
   19a8a: 83 a7 81 89  	lw	a5, -1896(gp)
   19a8e: ed db        	beqz	a5, 0x19a80 <_close_r+0x1a>
   19a90: e2 60        	ld	ra, 24(sp)
   19a92: 1c c0        	sw	a5, 0(s0)
   19a94: 42 64        	ld	s0, 16(sp)
   19a96: a2 64        	ld	s1, 8(sp)
   19a98: 05 61        	addi	sp, sp, 32
   19a9a: 82 80        	ret

0000000000019a9c <__errno>:
   19a9c: 03 b5 81 85  	ld	a0, -1960(gp)
   19aa0: 82 80        	ret

0000000000019aa2 <_fclose_r>:
   19aa2: 01 11        	addi	sp, sp, -32
   19aa4: 06 ec        	sd	ra, 24(sp)
   19aa6: 22 e8        	sd	s0, 16(sp)
   19aa8: 26 e4        	sd	s1, 8(sp)
   19aaa: 4a e0        	sd	s2, 0(sp)
   19aac: 89 c9        	beqz	a1, 0x19abe <_fclose_r+0x1c>
   19aae: 2e 84        	mv	s0, a1
   19ab0: aa 84        	mv	s1, a0
   19ab2: 19 c1        	beqz	a0, 0x19ab8 <_fclose_r+0x16>
   19ab4: 3c 49        	lw	a5, 80(a0)
   19ab6: a5 cf        	beqz	a5, 0x19b2e <_fclose_r+0x8c>
   19ab8: 83 17 04 01  	lh	a5, 16(s0)
   19abc: 89 eb        	bnez	a5, 0x19ace <_fclose_r+0x2c>
   19abe: e2 60        	ld	ra, 24(sp)
   19ac0: 42 64        	ld	s0, 16(sp)
   19ac2: 01 49        	mv	s2, zero
   19ac4: a2 64        	ld	s1, 8(sp)
   19ac6: 4a 85        	mv	a0, s2
   19ac8: 02 69        	ld	s2, 0(sp)
   19aca: 05 61        	addi	sp, sp, 32
   19acc: 82 80        	ret
   19ace: a2 85        	mv	a1, s0
   19ad0: 26 85        	mv	a0, s1
   19ad2: ef 80 bf a3  	jal	0x1250c <__sflush_r>
   19ad6: 3c 68        	ld	a5, 80(s0)
   19ad8: 2a 89        	mv	s2, a0
   19ada: 91 c7        	beqz	a5, 0x19ae6 <_fclose_r+0x44>
   19adc: 0c 78        	ld	a1, 48(s0)
   19ade: 26 85        	mv	a0, s1
   19ae0: 82 97        	jalr	a5
   19ae2: 63 4c 05 04  	bltz	a0, 0x19b3a <_fclose_r+0x98>
   19ae6: 83 57 04 01  	lhu	a5, 16(s0)
   19aea: 93 f7 07 08  	andi	a5, a5, 128
   19aee: a1 ef        	bnez	a5, 0x19b46 <_fclose_r+0xa4>
   19af0: 2c 6c        	ld	a1, 88(s0)
   19af2: 91 c9        	beqz	a1, 0x19b06 <_fclose_r+0x64>
   19af4: 93 07 44 07  	addi	a5, s0, 116
   19af8: 63 85 f5 00  	beq	a1, a5, 0x19b02 <_fclose_r+0x60>
   19afc: 26 85        	mv	a0, s1
   19afe: ef 80 7f f9  	jal	0x12a94 <_free_r>
   19b02: 23 3c 04 04  	sd	zero, 88(s0)
   19b06: 2c 7c        	ld	a1, 120(s0)
   19b08: 91 c5        	beqz	a1, 0x19b14 <_fclose_r+0x72>
   19b0a: 26 85        	mv	a0, s1
   19b0c: ef 80 9f f8  	jal	0x12a94 <_free_r>
   19b10: 23 3c 04 06  	sd	zero, 120(s0)
   19b14: ef 80 5f e5  	jal	0x12968 <__sfp_lock_acquire>
   19b18: 23 18 04 00  	sh	zero, 16(s0)
   19b1c: ef 80 ff e4  	jal	0x1296a <__sfp_lock_release>
   19b20: e2 60        	ld	ra, 24(sp)
   19b22: 42 64        	ld	s0, 16(sp)
   19b24: a2 64        	ld	s1, 8(sp)
   19b26: 4a 85        	mv	a0, s2
   19b28: 02 69        	ld	s2, 0(sp)
   19b2a: 05 61        	addi	sp, sp, 32
   19b2c: 82 80        	ret
   19b2e: ef 80 3f e3  	jal	0x12960 <__sinit>
   19b32: 83 17 04 01  	lh	a5, 16(s0)
   19b36: c1 d7        	beqz	a5, 0x19abe <_fclose_r+0x1c>
   19b38: 59 bf        	j	0x19ace <_fclose_r+0x2c>
   19b3a: 83 57 04 01  	lhu	a5, 16(s0)
   19b3e: 7d 59        	addi	s2, zero, -1
   19b40: 93 f7 07 08  	andi	a5, a5, 128
   19b44: d5 d7        	beqz	a5, 0x19af0 <_fclose_r+0x4e>
   19b46: 0c 6c        	ld	a1, 24(s0)
   19b48: 26 85        	mv	a0, s1
   19b4a: ef 80 bf f4  	jal	0x12a94 <_free_r>
   19b4e: 4d b7        	j	0x19af0 <_fclose_r+0x4e>

0000000000019b50 <fclose>:
   19b50: aa 85        	mv	a1, a0
   19b52: 03 b5 81 85  	ld	a0, -1960(gp)
   19b56: 6f f0 df f4  	j	0x19aa2 <_fclose_r>

0000000000019b5a <_fiprintf_r>:
   19b5a: 5d 71        	addi	sp, sp, -80
   19b5c: 13 03 81 02  	addi	t1, sp, 40
   19b60: 36 f4        	sd	a3, 40(sp)
   19b62: 9a 86        	mv	a3, t1
   19b64: 06 ec        	sd	ra, 24(sp)
   19b66: 3a f8        	sd	a4, 48(sp)
   19b68: 3e fc        	sd	a5, 56(sp)
   19b6a: c2 e0        	sd	a6, 64(sp)
   19b6c: c6 e4        	sd	a7, 72(sp)
   19b6e: 1a e4        	sd	t1, 8(sp)
   19b70: ef f0 ef 81  	jal	0x18b8e <_vfiprintf_r>
   19b74: e2 60        	ld	ra, 24(sp)
   19b76: 61 61        	addi	sp, sp, 80
   19b78: 82 80        	ret

0000000000019b7a <fiprintf>:
   19b7a: 2a 8e        	mv	t3, a0
   19b7c: 5d 71        	addi	sp, sp, -80
   19b7e: 03 b5 81 85  	ld	a0, -1960(gp)
   19b82: 13 03 01 02  	addi	t1, sp, 32
   19b86: 32 f0        	sd	a2, 32(sp)
   19b88: 36 f4        	sd	a3, 40(sp)
   19b8a: 2e 86        	mv	a2, a1
   19b8c: 9a 86        	mv	a3, t1
   19b8e: f2 85        	mv	a1, t3
   19b90: 06 ec        	sd	ra, 24(sp)
   19b92: 3a f8        	sd	a4, 48(sp)
   19b94: 3e fc        	sd	a5, 56(sp)
   19b96: c2 e0        	sd	a6, 64(sp)
   19b98: c6 e4        	sd	a7, 72(sp)
   19b9a: 1a e4        	sd	t1, 8(sp)
   19b9c: ef e0 3f ff  	jal	0x18b8e <_vfiprintf_r>
   19ba0: e2 60        	ld	ra, 24(sp)
   19ba2: 61 61        	addi	sp, sp, 80
   19ba4: 82 80        	ret

0000000000019ba6 <__fputwc>:
   19ba6: 79 71        	addi	sp, sp, -48
   19ba8: 22 f0        	sd	s0, 32(sp)
   19baa: 26 ec        	sd	s1, 24(sp)
   19bac: 4a e8        	sd	s2, 16(sp)
   19bae: 06 f4        	sd	ra, 40(sp)
   19bb0: 2a 89        	mv	s2, a0
   19bb2: ae 84        	mv	s1, a1
   19bb4: 32 84        	mv	s0, a2
   19bb6: ef b0 3f 97  	jal	0x15528 <__locale_mb_cur_max>
   19bba: 85 47        	addi	a5, zero, 1
   19bbc: 63 1b f5 00  	bne	a0, a5, 0x19bd2 <__fputwc+0x2c>
   19bc0: 9b 87 f4 ff  	addiw	a5, s1, -1
   19bc4: 13 07 e0 0f  	addi	a4, zero, 254
   19bc8: 63 65 f7 00  	bltu	a4, a5, 0x19bd2 <__fputwc+0x2c>
   19bcc: 23 04 91 00  	sb	s1, 8(sp)
   19bd0: 21 a8        	j	0x19be8 <__fputwc+0x42>
   19bd2: 93 06 44 0a  	addi	a3, s0, 164
   19bd6: 26 86        	mv	a2, s1
   19bd8: 2c 00        	addi	a1, sp, 8
   19bda: 4a 85        	mv	a0, s2
   19bdc: ef 10 10 0b  	jal	0x1b48c <_wcrtomb_r>
   19be0: fd 57        	addi	a5, zero, -1
   19be2: 63 0d f5 04  	beq	a0, a5, 0x19c3c <__fputwc+0x96>
   19be6: 0d c5        	beqz	a0, 0x19c10 <__fputwc+0x6a>
   19be8: 5c 44        	lw	a5, 12(s0)
   19bea: 83 45 81 00  	lbu	a1, 8(sp)
   19bee: 1b 87 f7 ff  	addiw	a4, a5, -1
   19bf2: 58 c4        	sw	a4, 12(s0)
   19bf4: 63 58 07 00  	bgez	a4, 0x19c04 <__fputwc+0x5e>
   19bf8: 1c 54        	lw	a5, 40(s0)
   19bfa: 63 43 f7 02  	blt	a4, a5, 0x19c20 <__fputwc+0x7a>
   19bfe: a9 47        	addi	a5, zero, 10
   19c00: 63 80 f5 02  	beq	a1, a5, 0x19c20 <__fputwc+0x7a>
   19c04: 1c 60        	ld	a5, 0(s0)
   19c06: 13 87 17 00  	addi	a4, a5, 1
   19c0a: 18 e0        	sd	a4, 0(s0)
   19c0c: 23 80 b7 00  	sb	a1, 0(a5)
   19c10: a2 70        	ld	ra, 40(sp)
   19c12: 02 74        	ld	s0, 32(sp)
   19c14: 42 69        	ld	s2, 16(sp)
   19c16: 1b 85 04 00  	sext.w	a0, s1
   19c1a: e2 64        	ld	s1, 24(sp)
   19c1c: 45 61        	addi	sp, sp, 48
   19c1e: 82 80        	ret
   19c20: 22 86        	mv	a2, s0
   19c22: 4a 85        	mv	a0, s2
   19c24: ef 10 80 76  	jal	0x1b38c <__swbuf_r>
   19c28: fd 57        	addi	a5, zero, -1
   19c2a: e3 13 f5 fe  	bne	a0, a5, 0x19c10 <__fputwc+0x6a>
   19c2e: 7d 55        	addi	a0, zero, -1
   19c30: a2 70        	ld	ra, 40(sp)
   19c32: 02 74        	ld	s0, 32(sp)
   19c34: e2 64        	ld	s1, 24(sp)
   19c36: 42 69        	ld	s2, 16(sp)
   19c38: 45 61        	addi	sp, sp, 48
   19c3a: 82 80        	ret
   19c3c: 83 57 04 01  	lhu	a5, 16(s0)
   19c40: 7d 55        	addi	a0, zero, -1
   19c42: 93 e7 07 04  	ori	a5, a5, 64
   19c46: 23 18 f4 00  	sh	a5, 16(s0)
   19c4a: dd b7        	j	0x19c30 <__fputwc+0x8a>

0000000000019c4c <_fputwc_r>:
   19c4c: 83 17 06 01  	lh	a5, 16(a2)
   19c50: 13 97 27 03  	slli	a4, a5, 50
   19c54: 63 4c 07 00  	bltz	a4, 0x19c6c <_fputwc_r+0x20>
   19c58: 03 27 c6 0a  	lw	a4, 172(a2)
   19c5c: 89 66        	lui	a3, 2
   19c5e: d5 8f        	or	a5, a5, a3
   19c60: 89 66        	lui	a3, 2
   19c62: 55 8f        	or	a4, a4, a3
   19c64: 23 18 f6 00  	sh	a5, 16(a2)
   19c68: 23 26 e6 0a  	sw	a4, 172(a2)
   19c6c: 6f f0 bf f3  	j	0x19ba6 <__fputwc>

0000000000019c70 <fputwc>:
   19c70: 01 11        	addi	sp, sp, -32
   19c72: 22 e8        	sd	s0, 16(sp)
   19c74: 03 b4 81 85  	ld	s0, -1960(gp)
   19c78: 06 ec        	sd	ra, 24(sp)
   19c7a: 2e 86        	mv	a2, a1
   19c7c: aa 85        	mv	a1, a0
   19c7e: 19 c0        	beqz	s0, 0x19c84 <fputwc+0x14>
   19c80: 3c 48        	lw	a5, 80(s0)
   19c82: 9d c7        	beqz	a5, 0x19cb0 <fputwc+0x40>
   19c84: 83 17 06 01  	lh	a5, 16(a2)
   19c88: 13 97 27 03  	slli	a4, a5, 50
   19c8c: 63 4c 07 00  	bltz	a4, 0x19ca4 <fputwc+0x34>
   19c90: 03 27 c6 0a  	lw	a4, 172(a2)
   19c94: 89 66        	lui	a3, 2
   19c96: d5 8f        	or	a5, a5, a3
   19c98: 89 66        	lui	a3, 2
   19c9a: 55 8f        	or	a4, a4, a3
   19c9c: 23 18 f6 00  	sh	a5, 16(a2)
   19ca0: 23 26 e6 0a  	sw	a4, 172(a2)
   19ca4: 22 85        	mv	a0, s0
   19ca6: 42 64        	ld	s0, 16(sp)
   19ca8: e2 60        	ld	ra, 24(sp)
   19caa: 05 61        	addi	sp, sp, 32
   19cac: 6f f0 bf ef  	j	0x19ba6 <__fputwc>
   19cb0: 2a e0        	sd	a0, 0(sp)
   19cb2: 22 85        	mv	a0, s0
   19cb4: 32 e4        	sd	a2, 8(sp)
   19cb6: ef 80 bf ca  	jal	0x12960 <__sinit>
   19cba: 22 66        	ld	a2, 8(sp)
   19cbc: 82 65        	ld	a1, 0(sp)
   19cbe: d9 b7        	j	0x19c84 <fputwc+0x14>

0000000000019cc0 <_fstat_r>:
   19cc0: 01 11        	addi	sp, sp, -32
   19cc2: 2e 87        	mv	a4, a1
   19cc4: 22 e8        	sd	s0, 16(sp)
   19cc6: 26 e4        	sd	s1, 8(sp)
   19cc8: 2a 84        	mv	s0, a0
   19cca: b2 85        	mv	a1, a2
   19ccc: 3a 85        	mv	a0, a4
   19cce: 06 ec        	sd	ra, 24(sp)
   19cd0: 23 ac 01 88  	sw	zero, -1896(gp)
   19cd4: ef 10 90 4e  	jal	0x1b9bc <_fstat>
   19cd8: fd 57        	addi	a5, zero, -1
   19cda: 63 07 f5 00  	beq	a0, a5, 0x19ce8 <_fstat_r+0x28>
   19cde: e2 60        	ld	ra, 24(sp)
   19ce0: 42 64        	ld	s0, 16(sp)
   19ce2: a2 64        	ld	s1, 8(sp)
   19ce4: 05 61        	addi	sp, sp, 32
   19ce6: 82 80        	ret
   19ce8: 83 a7 81 89  	lw	a5, -1896(gp)
   19cec: ed db        	beqz	a5, 0x19cde <_fstat_r+0x1e>
   19cee: e2 60        	ld	ra, 24(sp)
   19cf0: 1c c0        	sw	a5, 0(s0)
   19cf2: 42 64        	ld	s0, 16(sp)
   19cf4: a2 64        	ld	s1, 8(sp)
   19cf6: 05 61        	addi	sp, sp, 32
   19cf8: 82 80        	ret

0000000000019cfa <__sfvwrite_r>:
   19cfa: 1c 6a        	ld	a5, 16(a2)
   19cfc: 63 8a 07 1a  	beqz	a5, 0x19eb0 <__sfvwrite_r+0x1b6>
   19d00: 03 97 05 01  	lh	a4, 16(a1)
   19d04: 1d 71        	addi	sp, sp, -96
   19d06: a2 e8        	sd	s0, 80(sp)
   19d08: 52 f8        	sd	s4, 48(sp)
   19d0a: 5a f0        	sd	s6, 32(sp)
   19d0c: 86 ec        	sd	ra, 88(sp)
   19d0e: a6 e4        	sd	s1, 72(sp)
   19d10: ca e0        	sd	s2, 64(sp)
   19d12: 4e fc        	sd	s3, 56(sp)
   19d14: 56 f4        	sd	s5, 40(sp)
   19d16: 5e ec        	sd	s7, 24(sp)
   19d18: 62 e8        	sd	s8, 16(sp)
   19d1a: 66 e4        	sd	s9, 8(sp)
   19d1c: 93 77 87 00  	andi	a5, a4, 8
   19d20: 32 8b        	mv	s6, a2
   19d22: 2a 8a        	mv	s4, a0
   19d24: 2e 84        	mv	s0, a1
   19d26: b5 c3        	beqz	a5, 0x19d8a <__sfvwrite_r+0x90>
   19d28: 9c 6d        	ld	a5, 24(a1)
   19d2a: a5 c3        	beqz	a5, 0x19d8a <__sfvwrite_r+0x90>
   19d2c: 93 77 27 00  	andi	a5, a4, 2
   19d30: 83 34 0b 00  	ld	s1, 0(s6)
   19d34: a5 cb        	beqz	a5, 0x19da4 <__sfvwrite_r+0xaa>
   19d36: 3c 60        	ld	a5, 64(s0)
   19d38: 0c 78        	ld	a1, 48(s0)
   19d3a: b7 0a 00 80  	lui	s5, 524288
   19d3e: 81 49        	mv	s3, zero
   19d40: 01 49        	mv	s2, zero
   19d42: 93 ca 0a c0  	xori	s5, s5, -1024
   19d46: 4e 86        	mv	a2, s3
   19d48: 52 85        	mv	a0, s4
   19d4a: 63 0a 09 02  	beqz	s2, 0x19d7e <__sfvwrite_r+0x84>
   19d4e: ca 86        	mv	a3, s2
   19d50: 63 f3 2a 01  	bgeu	s5, s2, 0x19d56 <__sfvwrite_r+0x5c>
   19d54: d6 86        	mv	a3, s5
   19d56: 81 26        	sext.w	a3, a3
   19d58: 82 97        	jalr	a5
   19d5a: 63 53 a0 14  	blez	a0, 0x19ea0 <__sfvwrite_r+0x1a6>
   19d5e: 83 37 0b 01  	ld	a5, 16(s6)
   19d62: aa 99        	add	s3, s3, a0
   19d64: 33 09 a9 40  	sub	s2, s2, a0
   19d68: 89 8f        	sub	a5, a5, a0
   19d6a: 23 38 fb 00  	sd	a5, 16(s6)
   19d6e: 63 86 07 10  	beqz	a5, 0x19e7a <__sfvwrite_r+0x180>
   19d72: 3c 60        	ld	a5, 64(s0)
   19d74: 0c 78        	ld	a1, 48(s0)
   19d76: 4e 86        	mv	a2, s3
   19d78: 52 85        	mv	a0, s4
   19d7a: e3 1a 09 fc  	bnez	s2, 0x19d4e <__sfvwrite_r+0x54>
   19d7e: 83 b9 04 00  	ld	s3, 0(s1)
   19d82: 03 b9 84 00  	ld	s2, 8(s1)
   19d86: c1 04        	addi	s1, s1, 16
   19d88: 7d bf        	j	0x19d46 <__sfvwrite_r+0x4c>
   19d8a: a2 85        	mv	a1, s0
   19d8c: 52 85        	mv	a0, s4
   19d8e: ef 80 0f de  	jal	0x1236e <__swsetup_r>
   19d92: 63 1d 05 10  	bnez	a0, 0x19eac <__sfvwrite_r+0x1b2>
   19d96: 03 17 04 01  	lh	a4, 16(s0)
   19d9a: 83 34 0b 00  	ld	s1, 0(s6)
   19d9e: 93 77 27 00  	andi	a5, a4, 2
   19da2: d1 fb        	bnez	a5, 0x19d36 <__sfvwrite_r+0x3c>
   19da4: 93 77 17 00  	andi	a5, a4, 1
   19da8: a5 eb        	bnez	a5, 0x19e18 <__sfvwrite_r+0x11e>
   19daa: 83 2b c4 00  	lw	s7, 12(s0)
   19dae: 1c 60        	ld	a5, 0(s0)
   19db0: b7 0a 00 80  	lui	s5, 524288
   19db4: 01 4c        	mv	s8, zero
   19db6: 01 49        	mv	s2, zero
   19db8: 93 ca fa ff  	not	s5, s5
   19dbc: 3e 85        	mv	a0, a5
   19dbe: 63 07 09 04  	beqz	s2, 0x19e0c <__sfvwrite_r+0x112>
   19dc2: 93 76 07 20  	andi	a3, a4, 512
   19dc6: 63 89 06 12  	beqz	a3, 0x19ef8 <__sfvwrite_r+0x1fe>
   19dca: 63 78 79 19  	bgeu	s2, s7, 0x19f5a <__sfvwrite_r+0x260>
   19dce: ca 8c        	mv	s9, s2
   19dd0: ca 8b        	mv	s7, s2
   19dd2: 5e 86        	mv	a2, s7
   19dd4: e2 85        	mv	a1, s8
   19dd6: ef 00 40 2c  	jal	0x1a09a <memmove>
   19dda: 58 44        	lw	a4, 12(s0)
   19ddc: 1c 60        	ld	a5, 0(s0)
   19dde: ca 89        	mv	s3, s2
   19de0: 3b 07 97 41  	subw	a4, a4, s9
   19de4: de 97        	add	a5, a5, s7
   19de6: 58 c4        	sw	a4, 12(s0)
   19de8: 1c e0        	sd	a5, 0(s0)
   19dea: 01 49        	mv	s2, zero
   19dec: 83 37 0b 01  	ld	a5, 16(s6)
   19df0: 4e 9c        	add	s8, s8, s3
   19df2: b3 87 37 41  	sub	a5, a5, s3
   19df6: 23 38 fb 00  	sd	a5, 16(s6)
   19dfa: c1 c3        	beqz	a5, 0x19e7a <__sfvwrite_r+0x180>
   19dfc: 1c 60        	ld	a5, 0(s0)
   19dfe: 03 17 04 01  	lh	a4, 16(s0)
   19e02: 83 2b c4 00  	lw	s7, 12(s0)
   19e06: 3e 85        	mv	a0, a5
   19e08: e3 1d 09 fa  	bnez	s2, 0x19dc2 <__sfvwrite_r+0xc8>
   19e0c: 03 bc 04 00  	ld	s8, 0(s1)
   19e10: 03 b9 84 00  	ld	s2, 8(s1)
   19e14: c1 04        	addi	s1, s1, 16
   19e16: 5d b7        	j	0x19dbc <__sfvwrite_r+0xc2>
   19e18: 81 49        	mv	s3, zero
   19e1a: 01 45        	mv	a0, zero
   19e1c: 81 4c        	mv	s9, zero
   19e1e: 01 4c        	mv	s8, zero
   19e20: 63 0a 0c 08  	beqz	s8, 0x19eb4 <__sfvwrite_r+0x1ba>
   19e24: 45 c1        	beqz	a0, 0x19ec4 <__sfvwrite_r+0x1ca>
   19e26: ce 8b        	mv	s7, s3
   19e28: 63 73 3c 01  	bgeu	s8, s3, 0x19e2e <__sfvwrite_r+0x134>
   19e2c: e2 8b        	mv	s7, s8
   19e2e: 08 60        	ld	a0, 0(s0)
   19e30: 18 6c        	ld	a4, 24(s0)
   19e32: 5c 44        	lw	a5, 12(s0)
   19e34: 14 50        	lw	a3, 32(s0)
   19e36: 5e 89        	mv	s2, s7
   19e38: 63 76 a7 00  	bgeu	a4, a0, 0x19e44 <__sfvwrite_r+0x14a>
   19e3c: bb 8a d7 00  	addw	s5, a5, a3
   19e40: 63 cd 7a 09  	blt	s5, s7, 0x19eda <__sfvwrite_r+0x1e0>
   19e44: 63 c7 db 18  	blt	s7, a3, 0x19fd2 <__sfvwrite_r+0x2d8>
   19e48: 3c 60        	ld	a5, 64(s0)
   19e4a: 0c 78        	ld	a1, 48(s0)
   19e4c: 66 86        	mv	a2, s9
   19e4e: 52 85        	mv	a0, s4
   19e50: 82 97        	jalr	a5
   19e52: 2a 89        	mv	s2, a0
   19e54: 63 56 a0 04  	blez	a0, 0x19ea0 <__sfvwrite_r+0x1a6>
   19e58: 9b 07 05 00  	sext.w	a5, a0
   19e5c: bb 89 f9 40  	subw	s3, s3, a5
   19e60: 05 45        	addi	a0, zero, 1
   19e62: 63 8a 09 02  	beqz	s3, 0x19e96 <__sfvwrite_r+0x19c>
   19e66: 83 37 0b 01  	ld	a5, 16(s6)
   19e6a: ca 9c        	add	s9, s9, s2
   19e6c: 33 0c 2c 41  	sub	s8, s8, s2
   19e70: b3 87 27 41  	sub	a5, a5, s2
   19e74: 23 38 fb 00  	sd	a5, 16(s6)
   19e78: c5 f7        	bnez	a5, 0x19e20 <__sfvwrite_r+0x126>
   19e7a: 01 45        	mv	a0, zero
   19e7c: e6 60        	ld	ra, 88(sp)
   19e7e: 46 64        	ld	s0, 80(sp)
   19e80: a6 64        	ld	s1, 72(sp)
   19e82: 06 69        	ld	s2, 64(sp)
   19e84: e2 79        	ld	s3, 56(sp)
   19e86: 42 7a        	ld	s4, 48(sp)
   19e88: a2 7a        	ld	s5, 40(sp)
   19e8a: 02 7b        	ld	s6, 32(sp)
   19e8c: e2 6b        	ld	s7, 24(sp)
   19e8e: 42 6c        	ld	s8, 16(sp)
   19e90: a2 6c        	ld	s9, 8(sp)
   19e92: 25 61        	addi	sp, sp, 96
   19e94: 82 80        	ret
   19e96: a2 85        	mv	a1, s0
   19e98: 52 85        	mv	a0, s4
   19e9a: ef 80 4f ff  	jal	0x1268e <_fflush_r>
   19e9e: 61 d5        	beqz	a0, 0x19e66 <__sfvwrite_r+0x16c>
   19ea0: 83 17 04 01  	lh	a5, 16(s0)
   19ea4: 93 e7 07 04  	ori	a5, a5, 64
   19ea8: 23 18 f4 00  	sh	a5, 16(s0)
   19eac: 7d 55        	addi	a0, zero, -1
   19eae: f9 b7        	j	0x19e7c <__sfvwrite_r+0x182>
   19eb0: 01 45        	mv	a0, zero
   19eb2: 82 80        	ret
   19eb4: 03 bc 84 00  	ld	s8, 8(s1)
   19eb8: a6 87        	mv	a5, s1
   19eba: c1 04        	addi	s1, s1, 16
   19ebc: e3 0c 0c fe  	beqz	s8, 0x19eb4 <__sfvwrite_r+0x1ba>
   19ec0: 83 bc 07 00  	ld	s9, 0(a5)
   19ec4: 62 86        	mv	a2, s8
   19ec6: a9 45        	addi	a1, zero, 10
   19ec8: 66 85        	mv	a0, s9
   19eca: ef b0 df e8  	jal	0x15d56 <memchr>
   19ece: 63 04 05 14  	beqz	a0, 0x1a016 <__sfvwrite_r+0x31c>
   19ed2: 05 05        	addi	a0, a0, 1
   19ed4: bb 09 95 41  	subw	s3, a0, s9
   19ed8: b9 b7        	j	0x19e26 <__sfvwrite_r+0x12c>
   19eda: e6 85        	mv	a1, s9
   19edc: 56 86        	mv	a2, s5
   19ede: ef 00 c0 1b  	jal	0x1a09a <memmove>
   19ee2: 1c 60        	ld	a5, 0(s0)
   19ee4: a2 85        	mv	a1, s0
   19ee6: 52 85        	mv	a0, s4
   19ee8: d6 97        	add	a5, a5, s5
   19eea: 1c e0        	sd	a5, 0(s0)
   19eec: ef 80 2f fa  	jal	0x1268e <_fflush_r>
   19ef0: 45 f9        	bnez	a0, 0x19ea0 <__sfvwrite_r+0x1a6>
   19ef2: d6 87        	mv	a5, s5
   19ef4: 56 89        	mv	s2, s5
   19ef6: 9d b7        	j	0x19e5c <__sfvwrite_r+0x162>
   19ef8: 18 6c        	ld	a4, 24(s0)
   19efa: 63 67 f7 02  	bltu	a4, a5, 0x19f28 <__sfvwrite_r+0x22e>
   19efe: 18 50        	lw	a4, 32(s0)
   19f00: 63 64 e9 02  	bltu	s2, a4, 0x19f28 <__sfvwrite_r+0x22e>
   19f04: ca 86        	mv	a3, s2
   19f06: 63 f3 2a 01  	bgeu	s5, s2, 0x19f0c <__sfvwrite_r+0x212>
   19f0a: d6 86        	mv	a3, s5
   19f0c: 3b e7 e6 02  	remw	a4, a3, a4
   19f10: 3c 60        	ld	a5, 64(s0)
   19f12: 0c 78        	ld	a1, 48(s0)
   19f14: 62 86        	mv	a2, s8
   19f16: 52 85        	mv	a0, s4
   19f18: 99 9e        	subw	a3, a3, a4
   19f1a: 82 97        	jalr	a5
   19f1c: aa 89        	mv	s3, a0
   19f1e: e3 51 a0 f8  	blez	a0, 0x19ea0 <__sfvwrite_r+0x1a6>
   19f22: 33 09 39 41  	sub	s2, s2, s3
   19f26: d9 b5        	j	0x19dec <__sfvwrite_r+0xf2>
   19f28: de 89        	mv	s3, s7
   19f2a: 63 73 79 01  	bgeu	s2, s7, 0x19f30 <__sfvwrite_r+0x236>
   19f2e: ca 89        	mv	s3, s2
   19f30: 3e 85        	mv	a0, a5
   19f32: 4e 86        	mv	a2, s3
   19f34: e2 85        	mv	a1, s8
   19f36: ef 00 40 16  	jal	0x1a09a <memmove>
   19f3a: 58 44        	lw	a4, 12(s0)
   19f3c: 1c 60        	ld	a5, 0(s0)
   19f3e: bb 06 37 41  	subw	a3, a4, s3
   19f42: ce 97        	add	a5, a5, s3
   19f44: 54 c4        	sw	a3, 12(s0)
   19f46: 1c e0        	sd	a5, 0(s0)
   19f48: e9 fe        	bnez	a3, 0x19f22 <__sfvwrite_r+0x228>
   19f4a: a2 85        	mv	a1, s0
   19f4c: 52 85        	mv	a0, s4
   19f4e: ef 80 0f f4  	jal	0x1268e <_fflush_r>
   19f52: 39 f5        	bnez	a0, 0x19ea0 <__sfvwrite_r+0x1a6>
   19f54: 33 09 39 41  	sub	s2, s2, s3
   19f58: 51 bd        	j	0x19dec <__sfvwrite_r+0xf2>
   19f5a: 93 76 07 48  	andi	a3, a4, 1152
   19f5e: c1 ca        	beqz	a3, 0x19fee <__sfvwrite_r+0x2f4>
   19f60: 10 50        	lw	a2, 32(s0)
   19f62: 0c 6c        	ld	a1, 24(s0)
   19f64: 9b 16 16 00  	slliw	a3, a2, 1
   19f68: b1 9e        	addw	a3, a3, a2
   19f6a: 8d 8f        	sub	a5, a5, a1
   19f6c: 9b d9 f6 01  	srliw	s3, a3, 31
   19f70: 9b 8b 07 00  	sext.w	s7, a5
   19f74: bb 89 d9 00  	addw	s3, s3, a3
   19f78: 93 86 1b 00  	addi	a3, s7, 1
   19f7c: 9b d9 19 40  	sraiw	s3, s3, 1
   19f80: ca 96        	add	a3, a3, s2
   19f82: 63 f5 d9 00  	bgeu	s3, a3, 0x19f8c <__sfvwrite_r+0x292>
   19f86: 85 27        	addiw	a5, a5, 1
   19f88: bb 89 27 01  	addw	s3, a5, s2
   19f8c: 13 77 07 40  	andi	a4, a4, 1024
   19f90: 2d c3        	beqz	a4, 0x19ff2 <__sfvwrite_r+0x2f8>
   19f92: ce 85        	mv	a1, s3
   19f94: 52 85        	mv	a0, s4
   19f96: ef b0 4f f8  	jal	0x1571a <_malloc_r>
   19f9a: aa 8c        	mv	s9, a0
   19f9c: 41 c1        	beqz	a0, 0x1a01c <__sfvwrite_r+0x322>
   19f9e: 0c 6c        	ld	a1, 24(s0)
   19fa0: 5e 86        	mv	a2, s7
   19fa2: ef 60 8f db  	jal	0x1055a <memcpy>
   19fa6: 83 57 04 01  	lhu	a5, 16(s0)
   19faa: 93 f7 f7 b7  	andi	a5, a5, -1153
   19fae: 93 e7 07 08  	ori	a5, a5, 128
   19fb2: 23 18 f4 00  	sh	a5, 16(s0)
   19fb6: 33 85 7c 01  	add	a0, s9, s7
   19fba: bb 8b 79 41  	subw	s7, s3, s7
   19fbe: 23 3c 94 01  	sd	s9, 24(s0)
   19fc2: 23 26 74 01  	sw	s7, 12(s0)
   19fc6: 08 e0        	sd	a0, 0(s0)
   19fc8: 23 20 34 03  	sw	s3, 32(s0)
   19fcc: ca 8c        	mv	s9, s2
   19fce: ca 8b        	mv	s7, s2
   19fd0: 09 b5        	j	0x19dd2 <__sfvwrite_r+0xd8>
   19fd2: 5e 86        	mv	a2, s7
   19fd4: e6 85        	mv	a1, s9
   19fd6: ef 00 40 0c  	jal	0x1a09a <memmove>
   19fda: 54 44        	lw	a3, 12(s0)
   19fdc: 18 60        	ld	a4, 0(s0)
   19fde: 9b 87 0b 00  	sext.w	a5, s7
   19fe2: bb 86 76 41  	subw	a3, a3, s7
   19fe6: 5e 97        	add	a4, a4, s7
   19fe8: 54 c4        	sw	a3, 12(s0)
   19fea: 18 e0        	sd	a4, 0(s0)
   19fec: 85 bd        	j	0x19e5c <__sfvwrite_r+0x162>
   19fee: de 8c        	mv	s9, s7
   19ff0: cd b3        	j	0x19dd2 <__sfvwrite_r+0xd8>
   19ff2: 4e 86        	mv	a2, s3
   19ff4: 52 85        	mv	a0, s4
   19ff6: ef 00 80 1b  	jal	0x1a1ae <_realloc_r>
   19ffa: aa 8c        	mv	s9, a0
   19ffc: 4d fd        	bnez	a0, 0x19fb6 <__sfvwrite_r+0x2bc>
   19ffe: 0c 6c        	ld	a1, 24(s0)
   1a000: 52 85        	mv	a0, s4
   1a002: ef 80 3f a9  	jal	0x12a94 <_free_r>
   1a006: 83 17 04 01  	lh	a5, 16(s0)
   1a00a: 31 47        	addi	a4, zero, 12
   1a00c: 23 20 ea 00  	sw	a4, 0(s4)
   1a010: 93 f7 f7 f7  	andi	a5, a5, -129
   1a014: 41 bd        	j	0x19ea4 <__sfvwrite_r+0x1aa>
   1a016: 9b 09 1c 00  	addiw	s3, s8, 1
   1a01a: 31 b5        	j	0x19e26 <__sfvwrite_r+0x12c>
   1a01c: 31 47        	addi	a4, zero, 12
   1a01e: 83 17 04 01  	lh	a5, 16(s0)
   1a022: 23 20 ea 00  	sw	a4, 0(s4)
   1a026: bd bd        	j	0x19ea4 <__sfvwrite_r+0x1aa>

000000000001a028 <_isatty_r>:
   1a028: 01 11        	addi	sp, sp, -32
   1a02a: 22 e8        	sd	s0, 16(sp)
   1a02c: 26 e4        	sd	s1, 8(sp)
   1a02e: 2a 84        	mv	s0, a0
   1a030: 2e 85        	mv	a0, a1
   1a032: 06 ec        	sd	ra, 24(sp)
   1a034: 23 ac 01 88  	sw	zero, -1896(gp)
   1a038: ef 10 90 1c  	jal	0x1ba00 <_isatty>
   1a03c: fd 57        	addi	a5, zero, -1
   1a03e: 63 07 f5 00  	beq	a0, a5, 0x1a04c <_isatty_r+0x24>
   1a042: e2 60        	ld	ra, 24(sp)
   1a044: 42 64        	ld	s0, 16(sp)
   1a046: a2 64        	ld	s1, 8(sp)
   1a048: 05 61        	addi	sp, sp, 32
   1a04a: 82 80        	ret
   1a04c: 83 a7 81 89  	lw	a5, -1896(gp)
   1a050: ed db        	beqz	a5, 0x1a042 <_isatty_r+0x1a>
   1a052: e2 60        	ld	ra, 24(sp)
   1a054: 1c c0        	sw	a5, 0(s0)
   1a056: 42 64        	ld	s0, 16(sp)
   1a058: a2 64        	ld	s1, 8(sp)
   1a05a: 05 61        	addi	sp, sp, 32
   1a05c: 82 80        	ret

000000000001a05e <_lseek_r>:
   1a05e: 01 11        	addi	sp, sp, -32
   1a060: 2e 87        	mv	a4, a1
   1a062: 22 e8        	sd	s0, 16(sp)
   1a064: 26 e4        	sd	s1, 8(sp)
   1a066: b2 85        	mv	a1, a2
   1a068: 2a 84        	mv	s0, a0
   1a06a: 36 86        	mv	a2, a3
   1a06c: 3a 85        	mv	a0, a4
   1a06e: 06 ec        	sd	ra, 24(sp)
   1a070: 23 ac 01 88  	sw	zero, -1896(gp)
   1a074: ef 10 70 1c  	jal	0x1ba3a <_lseek>
   1a078: fd 57        	addi	a5, zero, -1
   1a07a: 63 07 f5 00  	beq	a0, a5, 0x1a088 <_lseek_r+0x2a>
   1a07e: e2 60        	ld	ra, 24(sp)
   1a080: 42 64        	ld	s0, 16(sp)
   1a082: a2 64        	ld	s1, 8(sp)
   1a084: 05 61        	addi	sp, sp, 32
   1a086: 82 80        	ret
   1a088: 83 a7 81 89  	lw	a5, -1896(gp)
   1a08c: ed db        	beqz	a5, 0x1a07e <_lseek_r+0x20>
   1a08e: e2 60        	ld	ra, 24(sp)
   1a090: 1c c0        	sw	a5, 0(s0)
   1a092: 42 64        	ld	s0, 16(sp)
   1a094: a2 64        	ld	s1, 8(sp)
   1a096: 05 61        	addi	sp, sp, 32
   1a098: 82 80        	ret

000000000001a09a <memmove>:
   1a09a: 63 f2 a5 02  	bgeu	a1, a0, 0x1a0be <memmove+0x24>
   1a09e: 33 87 c5 00  	add	a4, a1, a2
   1a0a2: 63 7e e5 00  	bgeu	a0, a4, 0x1a0be <memmove+0x24>
   1a0a6: b3 07 c5 00  	add	a5, a0, a2
   1a0aa: 1d ca        	beqz	a2, 0x1a0e0 <memmove+0x46>
   1a0ac: 83 46 f7 ff  	lbu	a3, -1(a4)
   1a0b0: fd 17        	addi	a5, a5, -1
   1a0b2: 7d 17        	addi	a4, a4, -1
   1a0b4: 23 80 d7 00  	sb	a3, 0(a5)
   1a0b8: e3 1a f5 fe  	bne	a0, a5, 0x1a0ac <memmove+0x12>
   1a0bc: 82 80        	ret
   1a0be: fd 47        	addi	a5, zero, 31
   1a0c0: 63 e1 c7 02  	bltu	a5, a2, 0x1a0e2 <memmove+0x48>
   1a0c4: aa 87        	mv	a5, a0
   1a0c6: 93 06 f6 ff  	addi	a3, a2, -1
   1a0ca: 4d c2        	beqz	a2, 0x1a16c <memmove+0xd2>
   1a0cc: 85 06        	addi	a3, a3, 1
   1a0ce: be 96        	add	a3, a3, a5
   1a0d0: 03 c7 05 00  	lbu	a4, 0(a1)
   1a0d4: 85 07        	addi	a5, a5, 1
   1a0d6: 85 05        	addi	a1, a1, 1
   1a0d8: a3 8f e7 fe  	sb	a4, -1(a5)
   1a0dc: e3 9a d7 fe  	bne	a5, a3, 0x1a0d0 <memmove+0x36>
   1a0e0: 82 80        	ret
   1a0e2: b3 67 b5 00  	or	a5, a0, a1
   1a0e6: 9d 8b        	andi	a5, a5, 7
   1a0e8: b5 ef        	bnez	a5, 0x1a164 <memmove+0xca>
   1a0ea: 93 08 06 fe  	addi	a7, a2, -32
   1a0ee: 93 f8 08 fe  	andi	a7, a7, -32
   1a0f2: 93 88 08 02  	addi	a7, a7, 32
   1a0f6: b3 07 15 01  	add	a5, a0, a7
   1a0fa: ae 86        	mv	a3, a1
   1a0fc: 2a 87        	mv	a4, a0
   1a0fe: 03 b8 06 00  	ld	a6, 0(a3)
   1a102: 93 86 06 02  	addi	a3, a3, 32
   1a106: 13 07 07 02  	addi	a4, a4, 32
   1a10a: 23 30 07 ff  	sd	a6, -32(a4)
   1a10e: 03 b8 86 fe  	ld	a6, -24(a3)
   1a112: 23 34 07 ff  	sd	a6, -24(a4)
   1a116: 03 b8 06 ff  	ld	a6, -16(a3)
   1a11a: 23 38 07 ff  	sd	a6, -16(a4)
   1a11e: 03 b8 86 ff  	ld	a6, -8(a3)
   1a122: 23 3c 07 ff  	sd	a6, -8(a4)
   1a126: e3 1c f7 fc  	bne	a4, a5, 0x1a0fe <memmove+0x64>
   1a12a: 13 77 86 01  	andi	a4, a2, 24
   1a12e: c6 95        	add	a1, a1, a7
   1a130: 13 78 f6 01  	andi	a6, a2, 31
   1a134: 0d cf        	beqz	a4, 0x1a16e <memmove+0xd4>
   1a136: ae 86        	mv	a3, a1
   1a138: 3e 87        	mv	a4, a5
   1a13a: b3 8e 07 01  	add	t4, a5, a6
   1a13e: 1d 4e        	addi	t3, zero, 7
   1a140: 03 b3 06 00  	ld	t1, 0(a3)
   1a144: 21 07        	addi	a4, a4, 8
   1a146: b3 88 ee 40  	sub	a7, t4, a4
   1a14a: 23 3c 67 fe  	sd	t1, -8(a4)
   1a14e: a1 06        	addi	a3, a3, 8
   1a150: e3 68 1e ff  	bltu	t3, a7, 0x1a140 <memmove+0xa6>
   1a154: 13 07 88 ff  	addi	a4, a6, -8
   1a158: 61 9b        	andi	a4, a4, -8
   1a15a: 21 07        	addi	a4, a4, 8
   1a15c: 1d 8a        	andi	a2, a2, 7
   1a15e: ba 97        	add	a5, a5, a4
   1a160: ba 95        	add	a1, a1, a4
   1a162: 95 b7        	j	0x1a0c6 <memmove+0x2c>
   1a164: 93 06 f6 ff  	addi	a3, a2, -1
   1a168: aa 87        	mv	a5, a0
   1a16a: 8d b7        	j	0x1a0cc <memmove+0x32>
   1a16c: 82 80        	ret
   1a16e: 42 86        	mv	a2, a6
   1a170: 99 bf        	j	0x1a0c6 <memmove+0x2c>

000000000001a172 <_read_r>:
   1a172: 01 11        	addi	sp, sp, -32
   1a174: 2e 87        	mv	a4, a1
   1a176: 22 e8        	sd	s0, 16(sp)
   1a178: 26 e4        	sd	s1, 8(sp)
   1a17a: b2 85        	mv	a1, a2
   1a17c: 2a 84        	mv	s0, a0
   1a17e: 36 86        	mv	a2, a3
   1a180: 3a 85        	mv	a0, a4
   1a182: 06 ec        	sd	ra, 24(sp)
   1a184: 23 ac 01 88  	sw	zero, -1896(gp)
   1a188: ef 10 f0 0d  	jal	0x1ba66 <_read>
   1a18c: fd 57        	addi	a5, zero, -1
   1a18e: 63 07 f5 00  	beq	a0, a5, 0x1a19c <_read_r+0x2a>
   1a192: e2 60        	ld	ra, 24(sp)
   1a194: 42 64        	ld	s0, 16(sp)
   1a196: a2 64        	ld	s1, 8(sp)
   1a198: 05 61        	addi	sp, sp, 32
   1a19a: 82 80        	ret
   1a19c: 83 a7 81 89  	lw	a5, -1896(gp)
   1a1a0: ed db        	beqz	a5, 0x1a192 <_read_r+0x20>
   1a1a2: e2 60        	ld	ra, 24(sp)
   1a1a4: 1c c0        	sw	a5, 0(s0)
   1a1a6: 42 64        	ld	s0, 16(sp)
   1a1a8: a2 64        	ld	s1, 8(sp)
   1a1aa: 05 61        	addi	sp, sp, 32
   1a1ac: 82 80        	ret

000000000001a1ae <_realloc_r>:
   1a1ae: 5d 71        	addi	sp, sp, -80
   1a1b0: 4a f8        	sd	s2, 48(sp)
   1a1b2: 86 e4        	sd	ra, 72(sp)
   1a1b4: a2 e0        	sd	s0, 64(sp)
   1a1b6: 26 fc        	sd	s1, 56(sp)
   1a1b8: 4e f4        	sd	s3, 40(sp)
   1a1ba: 52 f0        	sd	s4, 32(sp)
   1a1bc: 56 ec        	sd	s5, 24(sp)
   1a1be: 5a e8        	sd	s6, 16(sp)
   1a1c0: 5e e4        	sd	s7, 8(sp)
   1a1c2: 62 e0        	sd	s8, 0(sp)
   1a1c4: 32 89        	mv	s2, a2
   1a1c6: 63 89 05 14  	beqz	a1, 0x1a318 <_realloc_r+0x16a>
   1a1ca: 2e 84        	mv	s0, a1
   1a1cc: aa 89        	mv	s3, a0
   1a1ce: ef b0 ff c1  	jal	0x15dec <__malloc_lock>
   1a1d2: 83 37 84 ff  	ld	a5, -8(s0)
   1a1d6: 93 04 79 01  	addi	s1, s2, 23
   1a1da: 13 07 e0 02  	addi	a4, zero, 46
   1a1de: 93 0a 04 ff  	addi	s5, s0, -16
   1a1e2: 13 fa c7 ff  	andi	s4, a5, -4
   1a1e6: 63 7e 97 0a  	bgeu	a4, s1, 0x1a2a2 <_realloc_r+0xf4>
   1a1ea: 37 07 00 80  	lui	a4, 524288
   1a1ee: c1 98        	andi	s1, s1, -16
   1a1f0: 13 47 f7 ff  	not	a4, a4
   1a1f4: 63 6b 97 0a  	bltu	a4, s1, 0x1a2aa <_realloc_r+0xfc>
   1a1f8: 63 e9 24 0b  	bltu	s1, s2, 0x1a2aa <_realloc_r+0xfc>
   1a1fc: 63 51 9a 0c  	bge	s4, s1, 0x1a2be <_realloc_r+0x110>
   1a200: 37 07 02 00  	lui	a4, 32
   1a204: 13 0c 07 f4  	addi	s8, a4, -192
   1a208: 03 36 0c 01  	ld	a2, 16(s8)
   1a20c: b3 86 4a 01  	add	a3, s5, s4
   1a210: 98 66        	ld	a4, 8(a3)
   1a212: 63 02 d6 14  	beq	a2, a3, 0x1a356 <_realloc_r+0x1a8>
   1a216: 13 76 e7 ff  	andi	a2, a4, -2
   1a21a: 36 96        	add	a2, a2, a3
   1a21c: 10 66        	ld	a2, 8(a2)
   1a21e: 05 8a        	andi	a2, a2, 1
   1a220: 75 e2        	bnez	a2, 0x1a304 <_realloc_r+0x156>
   1a222: 71 9b        	andi	a4, a4, -4
   1a224: 33 06 ea 00  	add	a2, s4, a4
   1a228: 63 56 96 08  	bge	a2, s1, 0x1a2b4 <_realloc_r+0x106>
   1a22c: 85 8b        	andi	a5, a5, 1
   1a22e: 8d e3        	bnez	a5, 0x1a250 <_realloc_r+0xa2>
   1a230: 83 3b 04 ff  	ld	s7, -16(s0)
   1a234: b3 8b 7a 41  	sub	s7, s5, s7
   1a238: 83 b7 8b 00  	ld	a5, 8(s7)
   1a23c: f1 9b        	andi	a5, a5, -4
   1a23e: 3e 97        	add	a4, a4, a5
   1a240: 33 0b 47 01  	add	s6, a4, s4
   1a244: 63 51 9b 24  	bge	s6, s1, 0x1a486 <_realloc_r+0x2d8>
   1a248: 33 0b fa 00  	add	s6, s4, a5
   1a24c: 63 56 9b 1c  	bge	s6, s1, 0x1a418 <_realloc_r+0x26a>
   1a250: ca 85        	mv	a1, s2
   1a252: 4e 85        	mv	a0, s3
   1a254: ef b0 6f cc  	jal	0x1571a <_malloc_r>
   1a258: 2a 89        	mv	s2, a0
   1a25a: 63 05 05 2e  	beqz	a0, 0x1a544 <_realloc_r+0x396>
   1a25e: 83 37 84 ff  	ld	a5, -8(s0)
   1a262: 13 07 05 ff  	addi	a4, a0, -16
   1a266: f9 9b        	andi	a5, a5, -2
   1a268: d6 97        	add	a5, a5, s5
   1a26a: 63 82 e7 1a  	beq	a5, a4, 0x1a40e <_realloc_r+0x260>
   1a26e: 13 06 8a ff  	addi	a2, s4, -8
   1a272: 93 07 80 04  	addi	a5, zero, 72
   1a276: 63 e4 c7 20  	bltu	a5, a2, 0x1a47e <_realloc_r+0x2d0>
   1a27a: 13 07 70 02  	addi	a4, zero, 39
   1a27e: 63 6b c7 16  	bltu	a4, a2, 0x1a3f4 <_realloc_r+0x246>
   1a282: aa 87        	mv	a5, a0
   1a284: 22 87        	mv	a4, s0
   1a286: 14 63        	ld	a3, 0(a4)
   1a288: 94 e3        	sd	a3, 0(a5)
   1a28a: 14 67        	ld	a3, 8(a4)
   1a28c: 94 e7        	sd	a3, 8(a5)
   1a28e: 18 6b        	ld	a4, 16(a4)
   1a290: 98 eb        	sd	a4, 16(a5)
   1a292: 4e 85        	mv	a0, s3
   1a294: a2 85        	mv	a1, s0
   1a296: ef 80 ef ff  	jal	0x12a94 <_free_r>
   1a29a: 4e 85        	mv	a0, s3
   1a29c: ef b0 3f b5  	jal	0x15dee <__malloc_unlock>
   1a2a0: a9 a0        	j	0x1a2ea <_realloc_r+0x13c>
   1a2a2: 93 04 00 02  	addi	s1, zero, 32
   1a2a6: e3 fb 24 f5  	bgeu	s1, s2, 0x1a1fc <_realloc_r+0x4e>
   1a2aa: b1 47        	addi	a5, zero, 12
   1a2ac: 23 a0 f9 00  	sw	a5, 0(s3)
   1a2b0: 01 49        	mv	s2, zero
   1a2b2: 25 a8        	j	0x1a2ea <_realloc_r+0x13c>
   1a2b4: 9c 6e        	ld	a5, 24(a3)
   1a2b6: 98 6a        	ld	a4, 16(a3)
   1a2b8: 32 8a        	mv	s4, a2
   1a2ba: 1c ef        	sd	a5, 24(a4)
   1a2bc: 98 eb        	sd	a4, 16(a5)
   1a2be: 83 b7 8a 00  	ld	a5, 8(s5)
   1a2c2: b3 06 9a 40  	sub	a3, s4, s1
   1a2c6: 7d 46        	addi	a2, zero, 31
   1a2c8: 85 8b        	andi	a5, a5, 1
   1a2ca: 33 87 4a 01  	add	a4, s5, s4
   1a2ce: 63 63 d6 06  	bltu	a2, a3, 0x1a334 <_realloc_r+0x186>
   1a2d2: b3 e7 47 01  	or	a5, a5, s4
   1a2d6: 23 b4 fa 00  	sd	a5, 8(s5)
   1a2da: 1c 67        	ld	a5, 8(a4)
   1a2dc: 93 e7 17 00  	ori	a5, a5, 1
   1a2e0: 1c e7        	sd	a5, 8(a4)
   1a2e2: 4e 85        	mv	a0, s3
   1a2e4: ef b0 bf b0  	jal	0x15dee <__malloc_unlock>
   1a2e8: 22 89        	mv	s2, s0
   1a2ea: a6 60        	ld	ra, 72(sp)
   1a2ec: 06 64        	ld	s0, 64(sp)
   1a2ee: e2 74        	ld	s1, 56(sp)
   1a2f0: a2 79        	ld	s3, 40(sp)
   1a2f2: 02 7a        	ld	s4, 32(sp)
   1a2f4: e2 6a        	ld	s5, 24(sp)
   1a2f6: 42 6b        	ld	s6, 16(sp)
   1a2f8: a2 6b        	ld	s7, 8(sp)
   1a2fa: 02 6c        	ld	s8, 0(sp)
   1a2fc: 4a 85        	mv	a0, s2
   1a2fe: 42 79        	ld	s2, 48(sp)
   1a300: 61 61        	addi	sp, sp, 80
   1a302: 82 80        	ret
   1a304: 85 8b        	andi	a5, a5, 1
   1a306: a9 f7        	bnez	a5, 0x1a250 <_realloc_r+0xa2>
   1a308: 83 3b 04 ff  	ld	s7, -16(s0)
   1a30c: b3 8b 7a 41  	sub	s7, s5, s7
   1a310: 83 b7 8b 00  	ld	a5, 8(s7)
   1a314: f1 9b        	andi	a5, a5, -4
   1a316: 0d bf        	j	0x1a248 <_realloc_r+0x9a>
   1a318: 06 64        	ld	s0, 64(sp)
   1a31a: a6 60        	ld	ra, 72(sp)
   1a31c: e2 74        	ld	s1, 56(sp)
   1a31e: 42 79        	ld	s2, 48(sp)
   1a320: a2 79        	ld	s3, 40(sp)
   1a322: 02 7a        	ld	s4, 32(sp)
   1a324: e2 6a        	ld	s5, 24(sp)
   1a326: 42 6b        	ld	s6, 16(sp)
   1a328: a2 6b        	ld	s7, 8(sp)
   1a32a: 02 6c        	ld	s8, 0(sp)
   1a32c: b2 85        	mv	a1, a2
   1a32e: 61 61        	addi	sp, sp, 80
   1a330: 6f b0 af be  	j	0x1571a <_malloc_r>
   1a334: c5 8f        	or	a5, a5, s1
   1a336: 23 b4 fa 00  	sd	a5, 8(s5)
   1a33a: b3 85 9a 00  	add	a1, s5, s1
   1a33e: 93 e6 16 00  	ori	a3, a3, 1
   1a342: 94 e5        	sd	a3, 8(a1)
   1a344: 1c 67        	ld	a5, 8(a4)
   1a346: c1 05        	addi	a1, a1, 16
   1a348: 4e 85        	mv	a0, s3
   1a34a: 93 e7 17 00  	ori	a5, a5, 1
   1a34e: 1c e7        	sd	a5, 8(a4)
   1a350: ef 80 4f f4  	jal	0x12a94 <_free_r>
   1a354: 79 b7        	j	0x1a2e2 <_realloc_r+0x134>
   1a356: 71 9b        	andi	a4, a4, -4
   1a358: b3 06 47 01  	add	a3, a4, s4
   1a35c: 13 86 04 02  	addi	a2, s1, 32
   1a360: 63 da c6 1a  	bge	a3, a2, 0x1a514 <_realloc_r+0x366>
   1a364: 85 8b        	andi	a5, a5, 1
   1a366: e3 95 07 ee  	bnez	a5, 0x1a250 <_realloc_r+0xa2>
   1a36a: 83 3b 04 ff  	ld	s7, -16(s0)
   1a36e: b3 8b 7a 41  	sub	s7, s5, s7
   1a372: 83 b7 8b 00  	ld	a5, 8(s7)
   1a376: f1 9b        	andi	a5, a5, -4
   1a378: 3e 97        	add	a4, a4, a5
   1a37a: 33 0b 47 01  	add	s6, a4, s4
   1a37e: e3 45 cb ec  	blt	s6, a2, 0x1a248 <_realloc_r+0x9a>
   1a382: 83 b7 8b 01  	ld	a5, 24(s7)
   1a386: 03 b7 0b 01  	ld	a4, 16(s7)
   1a38a: 13 06 8a ff  	addi	a2, s4, -8
   1a38e: 93 06 80 04  	addi	a3, zero, 72
   1a392: 1c ef        	sd	a5, 24(a4)
   1a394: 98 eb        	sd	a4, 16(a5)
   1a396: 13 89 0b 01  	addi	s2, s7, 16
   1a39a: 63 e3 c6 1c  	bltu	a3, a2, 0x1a560 <_realloc_r+0x3b2>
   1a39e: 13 07 70 02  	addi	a4, zero, 39
   1a3a2: ca 87        	mv	a5, s2
   1a3a4: 63 7f c7 00  	bgeu	a4, a2, 0x1a3c2 <_realloc_r+0x214>
   1a3a8: 18 60        	ld	a4, 0(s0)
   1a3aa: 93 07 70 03  	addi	a5, zero, 55
   1a3ae: 23 b8 eb 00  	sd	a4, 16(s7)
   1a3b2: 18 64        	ld	a4, 8(s0)
   1a3b4: 23 bc eb 00  	sd	a4, 24(s7)
   1a3b8: 63 e9 c7 1a  	bltu	a5, a2, 0x1a56a <_realloc_r+0x3bc>
   1a3bc: 41 04        	addi	s0, s0, 16
   1a3be: 93 87 0b 02  	addi	a5, s7, 32
   1a3c2: 18 60        	ld	a4, 0(s0)
   1a3c4: 98 e3        	sd	a4, 0(a5)
   1a3c6: 18 64        	ld	a4, 8(s0)
   1a3c8: 98 e7        	sd	a4, 8(a5)
   1a3ca: 18 68        	ld	a4, 16(s0)
   1a3cc: 98 eb        	sd	a4, 16(a5)
   1a3ce: 33 87 9b 00  	add	a4, s7, s1
   1a3d2: b3 07 9b 40  	sub	a5, s6, s1
   1a3d6: 23 38 ec 00  	sd	a4, 16(s8)
   1a3da: 93 e7 17 00  	ori	a5, a5, 1
   1a3de: 1c e7        	sd	a5, 8(a4)
   1a3e0: 83 b7 8b 00  	ld	a5, 8(s7)
   1a3e4: 4e 85        	mv	a0, s3
   1a3e6: 85 8b        	andi	a5, a5, 1
   1a3e8: c5 8f        	or	a5, a5, s1
   1a3ea: 23 b4 fb 00  	sd	a5, 8(s7)
   1a3ee: ef b0 1f a0  	jal	0x15dee <__malloc_unlock>
   1a3f2: e5 bd        	j	0x1a2ea <_realloc_r+0x13c>
   1a3f4: 14 60        	ld	a3, 0(s0)
   1a3f6: 13 07 70 03  	addi	a4, zero, 55
   1a3fa: 14 e1        	sd	a3, 0(a0)
   1a3fc: 14 64        	ld	a3, 8(s0)
   1a3fe: 14 e5        	sd	a3, 8(a0)
   1a400: 63 6f c7 0e  	bltu	a4, a2, 0x1a4fe <_realloc_r+0x350>
   1a404: 13 07 04 01  	addi	a4, s0, 16
   1a408: 93 07 05 01  	addi	a5, a0, 16
   1a40c: ad bd        	j	0x1a286 <_realloc_r+0xd8>
   1a40e: 83 37 85 ff  	ld	a5, -8(a0)
   1a412: f1 9b        	andi	a5, a5, -4
   1a414: 3e 9a        	add	s4, s4, a5
   1a416: 65 b5        	j	0x1a2be <_realloc_r+0x110>
   1a418: 83 b7 8b 01  	ld	a5, 24(s7)
   1a41c: 03 b7 0b 01  	ld	a4, 16(s7)
   1a420: 13 06 8a ff  	addi	a2, s4, -8
   1a424: 93 06 80 04  	addi	a3, zero, 72
   1a428: 1c ef        	sd	a5, 24(a4)
   1a42a: 98 eb        	sd	a4, 16(a5)
   1a42c: 13 89 0b 01  	addi	s2, s7, 16
   1a430: 63 e2 c6 0c  	bltu	a3, a2, 0x1a4f4 <_realloc_r+0x346>
   1a434: 13 07 70 02  	addi	a4, zero, 39
   1a438: ca 87        	mv	a5, s2
   1a43a: 63 78 c7 02  	bgeu	a4, a2, 0x1a46a <_realloc_r+0x2bc>
   1a43e: 18 60        	ld	a4, 0(s0)
   1a440: 93 07 70 03  	addi	a5, zero, 55
   1a444: 23 b8 eb 00  	sd	a4, 16(s7)
   1a448: 18 64        	ld	a4, 8(s0)
   1a44a: 23 bc eb 00  	sd	a4, 24(s7)
   1a44e: 63 f7 c7 0e  	bgeu	a5, a2, 0x1a53c <_realloc_r+0x38e>
   1a452: 1c 68        	ld	a5, 16(s0)
   1a454: 23 b0 fb 02  	sd	a5, 32(s7)
   1a458: 1c 6c        	ld	a5, 24(s0)
   1a45a: 23 b4 fb 02  	sd	a5, 40(s7)
   1a45e: 63 0f d6 06  	beq	a2, a3, 0x1a4dc <_realloc_r+0x32e>
   1a462: 13 04 04 02  	addi	s0, s0, 32
   1a466: 93 87 0b 03  	addi	a5, s7, 48
   1a46a: 18 60        	ld	a4, 0(s0)
   1a46c: 98 e3        	sd	a4, 0(a5)
   1a46e: 18 64        	ld	a4, 8(s0)
   1a470: 98 e7        	sd	a4, 8(a5)
   1a472: 18 68        	ld	a4, 16(s0)
   1a474: 98 eb        	sd	a4, 16(a5)
   1a476: 4a 84        	mv	s0, s2
   1a478: 5a 8a        	mv	s4, s6
   1a47a: de 8a        	mv	s5, s7
   1a47c: 89 b5        	j	0x1a2be <_realloc_r+0x110>
   1a47e: a2 85        	mv	a1, s0
   1a480: ef f0 bf c1  	jal	0x1a09a <memmove>
   1a484: 39 b5        	j	0x1a292 <_realloc_r+0xe4>
   1a486: 9c 6e        	ld	a5, 24(a3)
   1a488: 98 6a        	ld	a4, 16(a3)
   1a48a: 13 06 8a ff  	addi	a2, s4, -8
   1a48e: 93 06 80 04  	addi	a3, zero, 72
   1a492: 1c ef        	sd	a5, 24(a4)
   1a494: 98 eb        	sd	a4, 16(a5)
   1a496: 03 b7 0b 01  	ld	a4, 16(s7)
   1a49a: 83 b7 8b 01  	ld	a5, 24(s7)
   1a49e: 13 89 0b 01  	addi	s2, s7, 16
   1a4a2: 1c ef        	sd	a5, 24(a4)
   1a4a4: 98 eb        	sd	a4, 16(a5)
   1a4a6: 63 e7 c6 04  	bltu	a3, a2, 0x1a4f4 <_realloc_r+0x346>
   1a4aa: 13 07 70 02  	addi	a4, zero, 39
   1a4ae: ca 87        	mv	a5, s2
   1a4b0: e3 7d c7 fa  	bgeu	a4, a2, 0x1a46a <_realloc_r+0x2bc>
   1a4b4: 18 60        	ld	a4, 0(s0)
   1a4b6: 93 07 70 03  	addi	a5, zero, 55
   1a4ba: 23 b8 eb 00  	sd	a4, 16(s7)
   1a4be: 18 64        	ld	a4, 8(s0)
   1a4c0: 23 bc eb 00  	sd	a4, 24(s7)
   1a4c4: 63 fc c7 06  	bgeu	a5, a2, 0x1a53c <_realloc_r+0x38e>
   1a4c8: 18 68        	ld	a4, 16(s0)
   1a4ca: 93 07 80 04  	addi	a5, zero, 72
   1a4ce: 23 b0 eb 02  	sd	a4, 32(s7)
   1a4d2: 18 6c        	ld	a4, 24(s0)
   1a4d4: 23 b4 eb 02  	sd	a4, 40(s7)
   1a4d8: e3 15 f6 f8  	bne	a2, a5, 0x1a462 <_realloc_r+0x2b4>
   1a4dc: 18 70        	ld	a4, 32(s0)
   1a4de: 93 87 0b 04  	addi	a5, s7, 64
   1a4e2: 13 04 04 03  	addi	s0, s0, 48
   1a4e6: 23 b8 eb 02  	sd	a4, 48(s7)
   1a4ea: 03 37 84 ff  	ld	a4, -8(s0)
   1a4ee: 23 bc eb 02  	sd	a4, 56(s7)
   1a4f2: a5 bf        	j	0x1a46a <_realloc_r+0x2bc>
   1a4f4: a2 85        	mv	a1, s0
   1a4f6: 4a 85        	mv	a0, s2
   1a4f8: ef f0 3f ba  	jal	0x1a09a <memmove>
   1a4fc: ad bf        	j	0x1a476 <_realloc_r+0x2c8>
   1a4fe: 18 68        	ld	a4, 16(s0)
   1a500: 18 e9        	sd	a4, 16(a0)
   1a502: 18 6c        	ld	a4, 24(s0)
   1a504: 18 ed        	sd	a4, 24(a0)
   1a506: 63 04 f6 04  	beq	a2, a5, 0x1a54e <_realloc_r+0x3a0>
   1a50a: 13 07 04 02  	addi	a4, s0, 32
   1a50e: 93 07 05 02  	addi	a5, a0, 32
   1a512: 95 bb        	j	0x1a286 <_realloc_r+0xd8>
   1a514: a6 9a        	add	s5, s5, s1
   1a516: b3 87 96 40  	sub	a5, a3, s1
   1a51a: 23 38 5c 01  	sd	s5, 16(s8)
   1a51e: 93 e7 17 00  	ori	a5, a5, 1
   1a522: 23 b4 fa 00  	sd	a5, 8(s5)
   1a526: 83 37 84 ff  	ld	a5, -8(s0)
   1a52a: 4e 85        	mv	a0, s3
   1a52c: 22 89        	mv	s2, s0
   1a52e: 85 8b        	andi	a5, a5, 1
   1a530: c5 8f        	or	a5, a5, s1
   1a532: 23 3c f4 fe  	sd	a5, -8(s0)
   1a536: ef b0 9f 8b  	jal	0x15dee <__malloc_unlock>
   1a53a: 45 bb        	j	0x1a2ea <_realloc_r+0x13c>
   1a53c: 41 04        	addi	s0, s0, 16
   1a53e: 93 87 0b 02  	addi	a5, s7, 32
   1a542: 25 b7        	j	0x1a46a <_realloc_r+0x2bc>
   1a544: 4e 85        	mv	a0, s3
   1a546: ef b0 9f 8a  	jal	0x15dee <__malloc_unlock>
   1a54a: 01 49        	mv	s2, zero
   1a54c: 79 bb        	j	0x1a2ea <_realloc_r+0x13c>
   1a54e: 14 70        	ld	a3, 32(s0)
   1a550: 13 07 04 03  	addi	a4, s0, 48
   1a554: 93 07 05 03  	addi	a5, a0, 48
   1a558: 14 f1        	sd	a3, 32(a0)
   1a55a: 14 74        	ld	a3, 40(s0)
   1a55c: 14 f5        	sd	a3, 40(a0)
   1a55e: 25 b3        	j	0x1a286 <_realloc_r+0xd8>
   1a560: a2 85        	mv	a1, s0
   1a562: 4a 85        	mv	a0, s2
   1a564: ef f0 7f b3  	jal	0x1a09a <memmove>
   1a568: 9d b5        	j	0x1a3ce <_realloc_r+0x220>
   1a56a: 1c 68        	ld	a5, 16(s0)
   1a56c: 23 b0 fb 02  	sd	a5, 32(s7)
   1a570: 1c 6c        	ld	a5, 24(s0)
   1a572: 23 b4 fb 02  	sd	a5, 40(s7)
   1a576: 63 07 d6 00  	beq	a2, a3, 0x1a584 <_realloc_r+0x3d6>
   1a57a: 13 04 04 02  	addi	s0, s0, 32
   1a57e: 93 87 0b 03  	addi	a5, s7, 48
   1a582: 81 b5        	j	0x1a3c2 <_realloc_r+0x214>
   1a584: 18 70        	ld	a4, 32(s0)
   1a586: 93 87 0b 04  	addi	a5, s7, 64
   1a58a: 13 04 04 03  	addi	s0, s0, 48
   1a58e: 23 b8 eb 02  	sd	a4, 48(s7)
   1a592: 03 37 84 ff  	ld	a4, -8(s0)
   1a596: 23 bc eb 02  	sd	a4, 56(s7)
   1a59a: 25 b5        	j	0x1a3c2 <_realloc_r+0x214>

000000000001a59c <cleanup_glue>:
   1a59c: 79 71        	addi	sp, sp, -48
   1a59e: 4a e8        	sd	s2, 16(sp)
   1a5a0: 03 b9 05 00  	ld	s2, 0(a1)
   1a5a4: 22 f0        	sd	s0, 32(sp)
   1a5a6: 26 ec        	sd	s1, 24(sp)
   1a5a8: 06 f4        	sd	ra, 40(sp)
   1a5aa: 4e e4        	sd	s3, 8(sp)
   1a5ac: 52 e0        	sd	s4, 0(sp)
   1a5ae: 2e 84        	mv	s0, a1
   1a5b0: aa 84        	mv	s1, a0
   1a5b2: 63 0b 09 02  	beqz	s2, 0x1a5e8 <cleanup_glue+0x4c>
   1a5b6: 83 39 09 00  	ld	s3, 0(s2)
   1a5ba: 63 83 09 02  	beqz	s3, 0x1a5e0 <cleanup_glue+0x44>
   1a5be: 03 ba 09 00  	ld	s4, 0(s3)
   1a5c2: 63 0b 0a 00  	beqz	s4, 0x1a5d8 <cleanup_glue+0x3c>
   1a5c6: 83 35 0a 00  	ld	a1, 0(s4)
   1a5ca: 99 c1        	beqz	a1, 0x1a5d0 <cleanup_glue+0x34>
   1a5cc: ef f0 1f fd  	jal	0x1a59c <cleanup_glue>
   1a5d0: d2 85        	mv	a1, s4
   1a5d2: 26 85        	mv	a0, s1
   1a5d4: ef 80 0f cc  	jal	0x12a94 <_free_r>
   1a5d8: ce 85        	mv	a1, s3
   1a5da: 26 85        	mv	a0, s1
   1a5dc: ef 80 8f cb  	jal	0x12a94 <_free_r>
   1a5e0: ca 85        	mv	a1, s2
   1a5e2: 26 85        	mv	a0, s1
   1a5e4: ef 80 0f cb  	jal	0x12a94 <_free_r>
   1a5e8: a2 85        	mv	a1, s0
   1a5ea: 02 74        	ld	s0, 32(sp)
   1a5ec: a2 70        	ld	ra, 40(sp)
   1a5ee: 42 69        	ld	s2, 16(sp)
   1a5f0: a2 69        	ld	s3, 8(sp)
   1a5f2: 02 6a        	ld	s4, 0(sp)
   1a5f4: 26 85        	mv	a0, s1
   1a5f6: e2 64        	ld	s1, 24(sp)
   1a5f8: 45 61        	addi	sp, sp, 48
   1a5fa: 6f 80 af c9  	j	0x12a94 <_free_r>

000000000001a5fe <_reclaim_reent>:
   1a5fe: 83 b7 81 85  	ld	a5, -1960(gp)
   1a602: 63 85 a7 0a  	beq	a5, a0, 0x1a6ac <_reclaim_reent+0xae>
   1a606: 2c 7d        	ld	a1, 120(a0)
   1a608: 79 71        	addi	sp, sp, -48
   1a60a: 26 ec        	sd	s1, 24(sp)
   1a60c: 06 f4        	sd	ra, 40(sp)
   1a60e: 22 f0        	sd	s0, 32(sp)
   1a610: 4a e8        	sd	s2, 16(sp)
   1a612: 4e e4        	sd	s3, 8(sp)
   1a614: aa 84        	mv	s1, a0
   1a616: 8d c5        	beqz	a1, 0x1a640 <_reclaim_reent+0x42>
   1a618: 01 49        	mv	s2, zero
   1a61a: 93 09 00 20  	addi	s3, zero, 512
   1a61e: b3 87 25 01  	add	a5, a1, s2
   1a622: 80 63        	ld	s0, 0(a5)
   1a624: 01 c8        	beqz	s0, 0x1a634 <_reclaim_reent+0x36>
   1a626: a2 85        	mv	a1, s0
   1a628: 00 60        	ld	s0, 0(s0)
   1a62a: 26 85        	mv	a0, s1
   1a62c: ef 80 8f c6  	jal	0x12a94 <_free_r>
   1a630: 7d f8        	bnez	s0, 0x1a626 <_reclaim_reent+0x28>
   1a632: ac 7c        	ld	a1, 120(s1)
   1a634: 21 09        	addi	s2, s2, 8
   1a636: e3 14 39 ff  	bne	s2, s3, 0x1a61e <_reclaim_reent+0x20>
   1a63a: 26 85        	mv	a0, s1
   1a63c: ef 80 8f c5  	jal	0x12a94 <_free_r>
   1a640: ac 70        	ld	a1, 96(s1)
   1a642: 81 c5        	beqz	a1, 0x1a64a <_reclaim_reent+0x4c>
   1a644: 26 85        	mv	a0, s1
   1a646: ef 80 ef c4  	jal	0x12a94 <_free_r>
   1a64a: 03 b4 84 1f  	ld	s0, 504(s1)
   1a64e: 01 cc        	beqz	s0, 0x1a666 <_reclaim_reent+0x68>
   1a650: 13 89 04 20  	addi	s2, s1, 512
   1a654: 63 09 24 01  	beq	s0, s2, 0x1a666 <_reclaim_reent+0x68>
   1a658: a2 85        	mv	a1, s0
   1a65a: 00 60        	ld	s0, 0(s0)
   1a65c: 26 85        	mv	a0, s1
   1a65e: ef 80 6f c3  	jal	0x12a94 <_free_r>
   1a662: e3 1b 89 fe  	bne	s2, s0, 0x1a658 <_reclaim_reent+0x5a>
   1a666: cc 64        	ld	a1, 136(s1)
   1a668: 81 c5        	beqz	a1, 0x1a670 <_reclaim_reent+0x72>
   1a66a: 26 85        	mv	a0, s1
   1a66c: ef 80 8f c2  	jal	0x12a94 <_free_r>
   1a670: bc 48        	lw	a5, 80(s1)
   1a672: 95 c7        	beqz	a5, 0x1a69e <_reclaim_reent+0xa0>
   1a674: bc 6c        	ld	a5, 88(s1)
   1a676: 26 85        	mv	a0, s1
   1a678: 82 97        	jalr	a5
   1a67a: 03 b4 04 52  	ld	s0, 1312(s1)
   1a67e: 05 c0        	beqz	s0, 0x1a69e <_reclaim_reent+0xa0>
   1a680: 0c 60        	ld	a1, 0(s0)
   1a682: 81 c5        	beqz	a1, 0x1a68a <_reclaim_reent+0x8c>
   1a684: 26 85        	mv	a0, s1
   1a686: ef f0 7f f1  	jal	0x1a59c <cleanup_glue>
   1a68a: a2 85        	mv	a1, s0
   1a68c: 02 74        	ld	s0, 32(sp)
   1a68e: a2 70        	ld	ra, 40(sp)
   1a690: 42 69        	ld	s2, 16(sp)
   1a692: a2 69        	ld	s3, 8(sp)
   1a694: 26 85        	mv	a0, s1
   1a696: e2 64        	ld	s1, 24(sp)
   1a698: 45 61        	addi	sp, sp, 48
   1a69a: 6f 80 af bf  	j	0x12a94 <_free_r>
   1a69e: a2 70        	ld	ra, 40(sp)
   1a6a0: 02 74        	ld	s0, 32(sp)
   1a6a2: e2 64        	ld	s1, 24(sp)
   1a6a4: 42 69        	ld	s2, 16(sp)
   1a6a6: a2 69        	ld	s3, 8(sp)
   1a6a8: 45 61        	addi	sp, sp, 48
   1a6aa: 82 80        	ret
   1a6ac: 82 80        	ret

000000000001a6ae <__ssprint_r>:
   1a6ae: 1c 6a        	ld	a5, 16(a2)
   1a6b0: 5d 71        	addi	sp, sp, -80
   1a6b2: 4e f4        	sd	s3, 40(sp)
   1a6b4: 56 ec        	sd	s5, 24(sp)
   1a6b6: 86 e4        	sd	ra, 72(sp)
   1a6b8: a2 e0        	sd	s0, 64(sp)
   1a6ba: 26 fc        	sd	s1, 56(sp)
   1a6bc: 4a f8        	sd	s2, 48(sp)
   1a6be: 52 f0        	sd	s4, 32(sp)
   1a6c0: 5a e8        	sd	s6, 16(sp)
   1a6c2: 5e e4        	sd	s7, 8(sp)
   1a6c4: 62 e0        	sd	s8, 0(sp)
   1a6c6: 83 39 06 00  	ld	s3, 0(a2)
   1a6ca: b2 8a        	mv	s5, a2
   1a6cc: 63 8d 07 10  	beqz	a5, 0x1a7e6 <__ssprint_r+0x138>
   1a6d0: d8 45        	lw	a4, 12(a1)
   1a6d2: 9c 61        	ld	a5, 0(a1)
   1a6d4: 2a 8b        	mv	s6, a0
   1a6d6: ae 84        	mv	s1, a1
   1a6d8: 01 4a        	mv	s4, zero
   1a6da: 01 44        	mv	s0, zero
   1a6dc: 3a 89        	mv	s2, a4
   1a6de: 3e 85        	mv	a0, a5
   1a6e0: 29 c0        	beqz	s0, 0x1a722 <__ssprint_r+0x74>
   1a6e2: 63 76 e4 04  	bgeu	s0, a4, 0x1a72e <__ssprint_r+0x80>
   1a6e6: 1b 07 04 00  	sext.w	a4, s0
   1a6ea: 3a 89        	mv	s2, a4
   1a6ec: ba 8b        	mv	s7, a4
   1a6ee: 3e 85        	mv	a0, a5
   1a6f0: d2 85        	mv	a1, s4
   1a6f2: 4a 86        	mv	a2, s2
   1a6f4: ef f0 7f 9a  	jal	0x1a09a <memmove>
   1a6f8: d4 44        	lw	a3, 12(s1)
   1a6fa: 9c 60        	ld	a5, 0(s1)
   1a6fc: 03 b7 0a 01  	ld	a4, 16(s5)
   1a700: bb 86 26 41  	subw	a3, a3, s2
   1a704: ca 97        	add	a5, a5, s2
   1a706: d4 c4        	sw	a3, 12(s1)
   1a708: 9c e0        	sd	a5, 0(s1)
   1a70a: 33 07 77 41  	sub	a4, a4, s7
   1a70e: 23 b8 ea 00  	sd	a4, 16(s5)
   1a712: 5e 9a        	add	s4, s4, s7
   1a714: 33 04 74 41  	sub	s0, s0, s7
   1a718: 79 c7        	beqz	a4, 0x1a7e6 <__ssprint_r+0x138>
   1a71a: d8 44        	lw	a4, 12(s1)
   1a71c: 3e 85        	mv	a0, a5
   1a71e: 3a 89        	mv	s2, a4
   1a720: 69 f0        	bnez	s0, 0x1a6e2 <__ssprint_r+0x34>
   1a722: 03 ba 09 00  	ld	s4, 0(s3)
   1a726: 03 b4 89 00  	ld	s0, 8(s3)
   1a72a: c1 09        	addi	s3, s3, 16
   1a72c: 45 bf        	j	0x1a6dc <__ssprint_r+0x2e>
   1a72e: 03 97 04 01  	lh	a4, 16(s1)
   1a732: 9b 0b 04 00  	sext.w	s7, s0
   1a736: 93 76 07 48  	andi	a3, a4, 1152
   1a73a: dd da        	beqz	a3, 0x1a6f0 <__ssprint_r+0x42>
   1a73c: 90 50        	lw	a2, 32(s1)
   1a73e: 8c 6c        	ld	a1, 24(s1)
   1a740: 9b 16 16 00  	slliw	a3, a2, 1
   1a744: b1 9e        	addw	a3, a3, a2
   1a746: 8d 8f        	sub	a5, a5, a1
   1a748: 1b d9 f6 01  	srliw	s2, a3, 31
   1a74c: 9b 8b 07 00  	sext.w	s7, a5
   1a750: 3b 09 d9 00  	addw	s2, s2, a3
   1a754: 93 86 1b 00  	addi	a3, s7, 1
   1a758: 1b 59 19 40  	sraiw	s2, s2, 1
   1a75c: a2 96        	add	a3, a3, s0
   1a75e: 63 75 d9 00  	bgeu	s2, a3, 0x1a768 <__ssprint_r+0xba>
   1a762: 85 27        	addiw	a5, a5, 1
   1a764: 3b 89 87 00  	addw	s2, a5, s0
   1a768: 13 77 07 40  	andi	a4, a4, 1024
   1a76c: 31 c7        	beqz	a4, 0x1a7b8 <__ssprint_r+0x10a>
   1a76e: ca 85        	mv	a1, s2
   1a770: 5a 85        	mv	a0, s6
   1a772: ef a0 9f fa  	jal	0x1571a <_malloc_r>
   1a776: 2a 8c        	mv	s8, a0
   1a778: 31 c9        	beqz	a0, 0x1a7cc <__ssprint_r+0x11e>
   1a77a: 8c 6c        	ld	a1, 24(s1)
   1a77c: 5e 86        	mv	a2, s7
   1a77e: ef 50 df dd  	jal	0x1055a <memcpy>
   1a782: 83 d7 04 01  	lhu	a5, 16(s1)
   1a786: 93 f7 f7 b7  	andi	a5, a5, -1153
   1a78a: 93 e7 07 08  	ori	a5, a5, 128
   1a78e: 23 98 f4 00  	sh	a5, 16(s1)
   1a792: 33 05 7c 01  	add	a0, s8, s7
   1a796: bb 0b 79 41  	subw	s7, s2, s7
   1a79a: 1b 07 04 00  	sext.w	a4, s0
   1a79e: 23 a0 24 03  	sw	s2, 32(s1)
   1a7a2: 23 a6 74 01  	sw	s7, 12(s1)
   1a7a6: 23 bc 84 01  	sd	s8, 24(s1)
   1a7aa: 88 e0        	sd	a0, 0(s1)
   1a7ac: 3a 89        	mv	s2, a4
   1a7ae: ba 8b        	mv	s7, a4
   1a7b0: aa 87        	mv	a5, a0
   1a7b2: e3 7f e4 f2  	bgeu	s0, a4, 0x1a6f0 <__ssprint_r+0x42>
   1a7b6: 15 bf        	j	0x1a6ea <__ssprint_r+0x3c>
   1a7b8: 4a 86        	mv	a2, s2
   1a7ba: 5a 85        	mv	a0, s6
   1a7bc: ef f0 3f 9f  	jal	0x1a1ae <_realloc_r>
   1a7c0: 2a 8c        	mv	s8, a0
   1a7c2: 61 f9        	bnez	a0, 0x1a792 <__ssprint_r+0xe4>
   1a7c4: 8c 6c        	ld	a1, 24(s1)
   1a7c6: 5a 85        	mv	a0, s6
   1a7c8: ef 80 cf ac  	jal	0x12a94 <_free_r>
   1a7cc: 83 d7 04 01  	lhu	a5, 16(s1)
   1a7d0: 31 47        	addi	a4, zero, 12
   1a7d2: 23 20 eb 00  	sw	a4, 0(s6)
   1a7d6: 93 e7 07 04  	ori	a5, a5, 64
   1a7da: 23 98 f4 00  	sh	a5, 16(s1)
   1a7de: 23 b8 0a 00  	sd	zero, 16(s5)
   1a7e2: 7d 55        	addi	a0, zero, -1
   1a7e4: 11 a0        	j	0x1a7e8 <__ssprint_r+0x13a>
   1a7e6: 01 45        	mv	a0, zero
   1a7e8: a6 60        	ld	ra, 72(sp)
   1a7ea: 06 64        	ld	s0, 64(sp)
   1a7ec: 23 a4 0a 00  	sw	zero, 8(s5)
   1a7f0: e2 74        	ld	s1, 56(sp)
   1a7f2: 42 79        	ld	s2, 48(sp)
   1a7f4: a2 79        	ld	s3, 40(sp)
   1a7f6: 02 7a        	ld	s4, 32(sp)
   1a7f8: e2 6a        	ld	s5, 24(sp)
   1a7fa: 42 6b        	ld	s6, 16(sp)
   1a7fc: a2 6b        	ld	s7, 8(sp)
   1a7fe: 02 6c        	ld	s8, 0(sp)
   1a800: 61 61        	addi	sp, sp, 80
   1a802: 82 80        	ret

000000000001a804 <_svfiprintf_r>:
   1a804: 03 d7 05 01  	lhu	a4, 16(a1)
   1a808: 05 71        	addi	sp, sp, -480
   1a80a: 52 fb        	sd	s4, 432(sp)
   1a80c: 5a f3        	sd	s6, 416(sp)
   1a80e: 62 eb        	sd	s8, 400(sp)
   1a810: ee fe        	sd	s11, 376(sp)
   1a812: 86 ef        	sd	ra, 472(sp)
   1a814: a2 eb        	sd	s0, 464(sp)
   1a816: a6 e7        	sd	s1, 456(sp)
   1a818: ca e3        	sd	s2, 448(sp)
   1a81a: 4e ff        	sd	s3, 440(sp)
   1a81c: 56 f7        	sd	s5, 424(sp)
   1a81e: 5e ef        	sd	s7, 408(sp)
   1a820: 66 e7        	sd	s9, 392(sp)
   1a822: 6a e3        	sd	s10, 384(sp)
   1a824: 13 77 07 08  	andi	a4, a4, 128
   1a828: 2e 8a        	mv	s4, a1
   1a82a: 2a 8b        	mv	s6, a0
   1a82c: b2 8d        	mv	s11, a2
   1a82e: 36 8c        	mv	s8, a3
   1a830: 01 c7        	beqz	a4, 0x1a838 <_svfiprintf_r+0x34>
   1a832: 98 6d        	ld	a4, 24(a1)
   1a834: e3 04 07 20  	beqz	a4, 0x1b23c <_svfiprintf_r+0xa38>
   1a838: f9 67        	lui	a5, 30
   1a83a: 93 0a 01 0f  	addi	s5, sp, 240
   1a83e: 81 4c        	mv	s9, zero
   1a840: 93 87 47 3a  	addi	a5, a5, 932
   1a844: f9 6b        	lui	s7, 30
   1a846: d6 f8        	sd	s5, 112(sp)
   1a848: 02 e1        	sd	zero, 128(sp)
   1a84a: 82 dc        	sw	zero, 120(sp)
   1a84c: 81 44        	mv	s1, zero
   1a84e: 56 84        	mv	s0, s5
   1a850: 02 e8        	sd	zero, 16(sp)
   1a852: 02 fc        	sd	zero, 56(sp)
   1a854: 02 f8        	sd	zero, 48(sp)
   1a856: 82 e0        	sd	zero, 64(sp)
   1a858: 3e e4        	sd	a5, 8(sp)
   1a85a: c1 49        	addi	s3, zero, 16
   1a85c: 93 8b 0b 51  	addi	s7, s7, 1296
   1a860: 66 8d        	mv	s10, s9
   1a862: e2 86        	mv	a3, s8
   1a864: 83 c7 0d 00  	lbu	a5, 0(s11)
   1a868: 63 8a 07 16  	beqz	a5, 0x1a9dc <_svfiprintf_r+0x1d8>
   1a86c: 6e 89        	mv	s2, s11
   1a86e: 13 07 50 02  	addi	a4, zero, 37
   1a872: 63 8b e7 1c  	beq	a5, a4, 0x1aa48 <_svfiprintf_r+0x244>
   1a876: 83 47 19 00  	lbu	a5, 1(s2)
   1a87a: 05 09        	addi	s2, s2, 1
   1a87c: fd fb        	bnez	a5, 0x1a872 <_svfiprintf_r+0x6e>
   1a87e: 3b 0c b9 41  	subw	s8, s2, s11
   1a882: 63 0d 0c 14  	beqz	s8, 0x1a9dc <_svfiprintf_r+0x1d8>
   1a886: e6 57        	lw	a5, 120(sp)
   1a888: e2 94        	add	s1, s1, s8
   1a88a: 23 30 b4 01  	sd	s11, 0(s0)
   1a88e: 1b 87 17 00  	addiw	a4, a5, 1
   1a892: 23 34 84 01  	sd	s8, 8(s0)
   1a896: 26 e1        	sd	s1, 128(sp)
   1a898: ba dc        	sw	a4, 120(sp)
   1a89a: 9d 47        	addi	a5, zero, 7
   1a89c: 41 04        	addi	s0, s0, 16
   1a89e: 63 cc e7 1a  	blt	a5, a4, 0x1aa56 <_svfiprintf_r+0x252>
   1a8a2: 83 47 09 00  	lbu	a5, 0(s2)
   1a8a6: 3b 0d 8d 01  	addw	s10, s10, s8
   1a8aa: 63 89 07 12  	beqz	a5, 0x1a9dc <_svfiprintf_r+0x1d8>
   1a8ae: 83 47 19 00  	lbu	a5, 1(s2)
   1a8b2: 93 0d 19 00  	addi	s11, s2, 1
   1a8b6: a3 03 01 06  	sb	zero, 103(sp)
   1a8ba: 7d 5c        	addi	s8, zero, -1
   1a8bc: 81 48        	mv	a7, zero
   1a8be: 81 4c        	mv	s9, zero
   1a8c0: 13 09 a0 05  	addi	s2, zero, 90
   1a8c4: 85 0d        	addi	s11, s11, 1
   1a8c6: 81 27        	sext.w	a5, a5
   1a8c8: 1b 87 07 fe  	addiw	a4, a5, -32
   1a8cc: 1b 06 07 00  	sext.w	a2, a4
   1a8d0: 63 65 c9 04  	bltu	s2, a2, 0x1a91a <_svfiprintf_r+0x116>
   1a8d4: 13 16 07 02  	slli	a2, a4, 32
   1a8d8: 13 57 e6 01  	srli	a4, a2, 30
   1a8dc: 22 66        	ld	a2, 8(sp)
   1a8de: 32 97        	add	a4, a4, a2
   1a8e0: 18 43        	lw	a4, 0(a4)
   1a8e2: 02 87        	jr	a4
   1a8e4: 81 48        	mv	a7, zero
   1a8e6: a5 45        	addi	a1, zero, 9
   1a8e8: 03 c6 0d 00  	lbu	a2, 0(s11)
   1a8ec: 1b 97 28 00  	slliw	a4, a7, 2
   1a8f0: bb 08 17 01  	addw	a7, a4, a7
   1a8f4: 9b 87 07 fd  	addiw	a5, a5, -48
   1a8f8: 9b 98 18 00  	slliw	a7, a7, 1
   1a8fc: 1b 07 06 fd  	addiw	a4, a2, -48
   1a900: bb 88 17 01  	addw	a7, a5, a7
   1a904: 85 0d        	addi	s11, s11, 1
   1a906: 9b 07 06 00  	sext.w	a5, a2
   1a90a: e3 ff e5 fc  	bgeu	a1, a4, 0x1a8e8 <_svfiprintf_r+0xe4>
   1a90e: 1b 87 07 fe  	addiw	a4, a5, -32
   1a912: 1b 06 07 00  	sext.w	a2, a4
   1a916: e3 7f c9 fa  	bgeu	s2, a2, 0x1a8d4 <_svfiprintf_r+0xd0>
   1a91a: e9 c3        	beqz	a5, 0x1a9dc <_svfiprintf_r+0x1d8>
   1a91c: 23 04 f1 08  	sb	a5, 136(sp)
   1a920: a3 03 01 06  	sb	zero, 103(sp)
   1a924: 05 4e        	addi	t3, zero, 1
   1a926: 05 43        	addi	t1, zero, 1
   1a928: 13 09 81 08  	addi	s2, sp, 136
   1a92c: 01 4c        	mv	s8, zero
   1a92e: 93 f2 2c 00  	andi	t0, s9, 2
   1a932: 63 83 02 00  	beqz	t0, 0x1a938 <_svfiprintf_r+0x134>
   1a936: 09 2e        	addiw	t3, t3, 2
   1a938: 93 ff 4c 08  	andi	t6, s9, 132
   1a93c: e6 57        	lw	a5, 120(sp)
   1a93e: 63 96 0f 00  	bnez	t6, 0x1a94a <_svfiprintf_r+0x146>
   1a942: 3b 87 c8 41  	subw	a4, a7, t3
   1a946: 63 46 e0 5e  	bgtz	a4, 0x1af32 <_svfiprintf_r+0x72e>
   1a94a: 03 47 71 06  	lbu	a4, 103(sp)
   1a94e: 11 cf        	beqz	a4, 0x1a96a <_svfiprintf_r+0x166>
   1a950: 13 06 71 06  	addi	a2, sp, 103
   1a954: 10 e0        	sd	a2, 0(s0)
   1a956: 85 27        	addiw	a5, a5, 1
   1a958: 85 04        	addi	s1, s1, 1
   1a95a: 05 46        	addi	a2, zero, 1
   1a95c: 10 e4        	sd	a2, 8(s0)
   1a95e: 26 e1        	sd	s1, 128(sp)
   1a960: be dc        	sw	a5, 120(sp)
   1a962: 1d 47        	addi	a4, zero, 7
   1a964: 41 04        	addi	s0, s0, 16
   1a966: 63 4a f7 12  	blt	a4, a5, 0x1aa9a <_svfiprintf_r+0x296>
   1a96a: 63 8e 02 00  	beqz	t0, 0x1a986 <_svfiprintf_r+0x182>
   1a96e: b0 10        	addi	a2, sp, 104
   1a970: 10 e0        	sd	a2, 0(s0)
   1a972: 85 27        	addiw	a5, a5, 1
   1a974: 89 04        	addi	s1, s1, 2
   1a976: 09 46        	addi	a2, zero, 2
   1a978: 10 e4        	sd	a2, 8(s0)
   1a97a: 26 e1        	sd	s1, 128(sp)
   1a97c: be dc        	sw	a5, 120(sp)
   1a97e: 1d 47        	addi	a4, zero, 7
   1a980: 41 04        	addi	s0, s0, 16
   1a982: 63 43 f7 66  	blt	a4, a5, 0x1afe8 <_svfiprintf_r+0x7e4>
   1a986: 13 07 00 08  	addi	a4, zero, 128
   1a98a: 63 80 ef 3e  	beq	t6, a4, 0x1ad6a <_svfiprintf_r+0x566>
   1a98e: 3b 0c 6c 40  	subw	s8, s8, t1
   1a992: 63 46 80 49  	bgtz	s8, 0x1ae1e <_svfiprintf_r+0x61a>
   1a996: 1b 87 17 00  	addiw	a4, a5, 1
   1a99a: 9a 94        	add	s1, s1, t1
   1a99c: 23 30 24 01  	sd	s2, 0(s0)
   1a9a0: 23 34 64 00  	sd	t1, 8(s0)
   1a9a4: 26 e1        	sd	s1, 128(sp)
   1a9a6: ba dc        	sw	a4, 120(sp)
   1a9a8: 9d 47        	addi	a5, zero, 7
   1a9aa: 41 04        	addi	s0, s0, 16
   1a9ac: 63 cd e7 50  	blt	a5, a4, 0x1aec6 <_svfiprintf_r+0x6c2>
   1a9b0: 93 fc 4c 00  	andi	s9, s9, 4
   1a9b4: 63 86 0c 00  	beqz	s9, 0x1a9c0 <_svfiprintf_r+0x1bc>
   1a9b8: 3b 89 c8 41  	subw	s2, a7, t3
   1a9bc: 63 4b 20 65  	bgtz	s2, 0x1b012 <_svfiprintf_r+0x80e>
   1a9c0: c6 87        	mv	a5, a7
   1a9c2: 63 d3 c8 01  	bge	a7, t3, 0x1a9c8 <_svfiprintf_r+0x1c4>
   1a9c6: f2 87        	mv	a5, t3
   1a9c8: 3b 8d a7 01  	addw	s10, a5, s10
   1a9cc: 63 9d 04 50  	bnez	s1, 0x1aee6 <_svfiprintf_r+0x6e2>
   1a9d0: 83 c7 0d 00  	lbu	a5, 0(s11)
   1a9d4: 82 dc        	sw	zero, 120(sp)
   1a9d6: 56 84        	mv	s0, s5
   1a9d8: e3 9a 07 e8  	bnez	a5, 0x1a86c <_svfiprintf_r+0x68>
   1a9dc: ea 8c        	mv	s9, s10
   1a9de: c1 c8        	beqz	s1, 0x1aa6e <_svfiprintf_r+0x26a>
   1a9e0: 90 18        	addi	a2, sp, 112
   1a9e2: d2 85        	mv	a1, s4
   1a9e4: 5a 85        	mv	a0, s6
   1a9e6: ef f0 9f cc  	jal	0x1a6ae <__ssprint_r>
   1a9ea: 51 a0        	j	0x1aa6e <_svfiprintf_r+0x26a>
   1a9ec: 93 ec 0c 01  	ori	s9, s9, 16
   1a9f0: 83 c7 0d 00  	lbu	a5, 0(s11)
   1a9f4: 81 2c        	sext.w	s9, s9
   1a9f6: f9 b5        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1a9f8: 93 ec 0c 01  	ori	s9, s9, 16
   1a9fc: 81 2c        	sext.w	s9, s9
   1a9fe: 93 f7 0c 02  	andi	a5, s9, 32
   1aa02: 93 8e 86 00  	addi	t4, a3, 8
   1aa06: e1 c3        	beqz	a5, 0x1aac6 <_svfiprintf_r+0x2c2>
   1aa08: 98 62        	ld	a4, 0(a3)
   1aa0a: ba 87        	mv	a5, a4
   1aa0c: 63 4c 07 0c  	bltz	a4, 0x1aae4 <_svfiprintf_r+0x2e0>
   1aa10: 7d 57        	addi	a4, zero, -1
   1aa12: 63 05 ec 0e  	beq	s8, a4, 0x1aafc <_svfiprintf_r+0x2f8>
   1aa16: 93 fc fc f7  	andi	s9, s9, -129
   1aa1a: 81 2c        	sext.w	s9, s9
   1aa1c: e5 e3        	bnez	a5, 0x1aafc <_svfiprintf_r+0x2f8>
   1aa1e: e3 1b 0c 12  	bnez	s8, 0x1b354 <_svfiprintf_r+0xb50>
   1aa22: 01 4e        	mv	t3, zero
   1aa24: 01 43        	mv	t1, zero
   1aa26: f6 86        	mv	a3, t4
   1aa28: 13 09 c1 0e  	addi	s2, sp, 236
   1aa2c: d5 a8        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1aa2e: 83 a8 06 00  	lw	a7, 0(a3)
   1aa32: a1 06        	addi	a3, a3, 8
   1aa34: 63 dc 08 2a  	bgez	a7, 0x1acec <_svfiprintf_r+0x4e8>
   1aa38: bb 08 10 41  	negw	a7, a7
   1aa3c: 93 ec 4c 00  	ori	s9, s9, 4
   1aa40: 83 c7 0d 00  	lbu	a5, 0(s11)
   1aa44: 81 2c        	sext.w	s9, s9
   1aa46: bd bd        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1aa48: 3b 0c b9 41  	subw	s8, s2, s11
   1aa4c: e3 1d 0c e2  	bnez	s8, 0x1a886 <_svfiprintf_r+0x82>
   1aa50: 83 47 09 00  	lbu	a5, 0(s2)
   1aa54: 99 bd        	j	0x1a8aa <_svfiprintf_r+0xa6>
   1aa56: 90 18        	addi	a2, sp, 112
   1aa58: d2 85        	mv	a1, s4
   1aa5a: 5a 85        	mv	a0, s6
   1aa5c: 36 ec        	sd	a3, 24(sp)
   1aa5e: ef f0 1f c5  	jal	0x1a6ae <__ssprint_r>
   1aa62: 09 e5        	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1aa64: 8a 64        	ld	s1, 128(sp)
   1aa66: e2 66        	ld	a3, 24(sp)
   1aa68: 56 84        	mv	s0, s5
   1aa6a: 25 bd        	j	0x1a8a2 <_svfiprintf_r+0x9e>
   1aa6c: ea 8c        	mv	s9, s10
   1aa6e: 83 57 0a 01  	lhu	a5, 16(s4)
   1aa72: 93 f7 07 04  	andi	a5, a5, 64
   1aa76: e3 9c 07 0e  	bnez	a5, 0x1b36e <_svfiprintf_r+0xb6a>
   1aa7a: fe 60        	ld	ra, 472(sp)
   1aa7c: 5e 64        	ld	s0, 464(sp)
   1aa7e: be 64        	ld	s1, 456(sp)
   1aa80: 1e 69        	ld	s2, 448(sp)
   1aa82: fa 79        	ld	s3, 440(sp)
   1aa84: 5a 7a        	ld	s4, 432(sp)
   1aa86: ba 7a        	ld	s5, 424(sp)
   1aa88: 1a 7b        	ld	s6, 416(sp)
   1aa8a: fa 6b        	ld	s7, 408(sp)
   1aa8c: 5a 6c        	ld	s8, 400(sp)
   1aa8e: 1a 6d        	ld	s10, 384(sp)
   1aa90: f6 7d        	ld	s11, 376(sp)
   1aa92: 66 85        	mv	a0, s9
   1aa94: ba 6c        	ld	s9, 392(sp)
   1aa96: 3d 61        	addi	sp, sp, 480
   1aa98: 82 80        	ret
   1aa9a: 90 18        	addi	a2, sp, 112
   1aa9c: d2 85        	mv	a1, s4
   1aa9e: 5a 85        	mv	a0, s6
   1aaa0: b6 ec        	sd	a3, 88(sp)
   1aaa2: 9a e8        	sd	t1, 80(sp)
   1aaa4: f2 e4        	sd	t3, 72(sp)
   1aaa6: 46 f4        	sd	a7, 40(sp)
   1aaa8: 7e f0        	sd	t6, 32(sp)
   1aaaa: 16 ec        	sd	t0, 24(sp)
   1aaac: ef f0 3f c0  	jal	0x1a6ae <__ssprint_r>
   1aab0: 55 fd        	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1aab2: 8a 64        	ld	s1, 128(sp)
   1aab4: e6 57        	lw	a5, 120(sp)
   1aab6: e6 66        	ld	a3, 88(sp)
   1aab8: 46 63        	ld	t1, 80(sp)
   1aaba: 26 6e        	ld	t3, 72(sp)
   1aabc: a2 78        	ld	a7, 40(sp)
   1aabe: 82 7f        	ld	t6, 32(sp)
   1aac0: e2 62        	ld	t0, 24(sp)
   1aac2: 56 84        	mv	s0, s5
   1aac4: 5d b5        	j	0x1a96a <_svfiprintf_r+0x166>
   1aac6: 93 f7 0c 01  	andi	a5, s9, 16
   1aaca: 9d ff        	bnez	a5, 0x1aa08 <_svfiprintf_r+0x204>
   1aacc: 13 f7 0c 04  	andi	a4, s9, 64
   1aad0: 9c 42        	lw	a5, 0(a3)
   1aad2: e3 0d 07 02  	beqz	a4, 0x1b30c <_svfiprintf_r+0xb08>
   1aad6: 9b 97 07 01  	slliw	a5, a5, 16
   1aada: 9b d7 07 41  	sraiw	a5, a5, 16
   1aade: 3e 87        	mv	a4, a5
   1aae0: e3 58 07 f2  	bgez	a4, 0x1aa10 <_svfiprintf_r+0x20c>
   1aae4: 13 07 d0 02  	addi	a4, zero, 45
   1aae8: a3 03 e1 06  	sb	a4, 103(sp)
   1aaec: 7d 57        	addi	a4, zero, -1
   1aaee: b3 07 f0 40  	neg	a5, a5
   1aaf2: 63 05 ec 00  	beq	s8, a4, 0x1aafc <_svfiprintf_r+0x2f8>
   1aaf6: 93 fc fc f7  	andi	s9, s9, -129
   1aafa: 81 2c        	sext.w	s9, s9
   1aafc: 25 47        	addi	a4, zero, 9
   1aafe: 63 65 f7 60  	bltu	a4, a5, 0x1b108 <_svfiprintf_r+0x904>
   1ab02: 9b 87 07 03  	addiw	a5, a5, 48
   1ab06: 93 f7 f7 0f  	andi	a5, a5, 255
   1ab0a: f6 86        	mv	a3, t4
   1ab0c: a3 05 f1 0e  	sb	a5, 235(sp)
   1ab10: 1b 0e 0c 00  	sext.w	t3, s8
   1ab14: 63 43 80 01  	bgtz	s8, 0x1ab1a <_svfiprintf_r+0x316>
   1ab18: 05 4e        	addi	t3, zero, 1
   1ab1a: 05 43        	addi	t1, zero, 1
   1ab1c: 13 09 b1 0e  	addi	s2, sp, 235
   1ab20: 83 47 71 06  	lbu	a5, 103(sp)
   1ab24: e3 85 07 e0  	beqz	a5, 0x1a92e <_svfiprintf_r+0x12a>
   1ab28: 05 2e        	addiw	t3, t3, 1
   1ab2a: 11 b5        	j	0x1a92e <_svfiprintf_r+0x12a>
   1ab2c: 03 b9 06 00  	ld	s2, 0(a3)
   1ab30: a3 03 01 06  	sb	zero, 103(sp)
   1ab34: a1 06        	addi	a3, a3, 8
   1ab36: 63 0b 09 78  	beqz	s2, 0x1b2cc <_svfiprintf_r+0xac8>
   1ab3a: fd 57        	addi	a5, zero, -1
   1ab3c: 36 f0        	sd	a3, 32(sp)
   1ab3e: 46 ec        	sd	a7, 24(sp)
   1ab40: 63 0b fc 6c  	beq	s8, a5, 0x1b216 <_svfiprintf_r+0xa12>
   1ab44: 62 86        	mv	a2, s8
   1ab46: 81 45        	mv	a1, zero
   1ab48: 4a 85        	mv	a0, s2
   1ab4a: ef b0 cf a0  	jal	0x15d56 <memchr>
   1ab4e: e2 68        	ld	a7, 24(sp)
   1ab50: 82 76        	ld	a3, 32(sp)
   1ab52: 63 0c 05 7e  	beqz	a0, 0x1b34a <_svfiprintf_r+0xb46>
   1ab56: 3b 03 25 41  	subw	t1, a0, s2
   1ab5a: 13 4e f3 ff  	not	t3, t1
   1ab5e: 13 5e fe 43  	srai	t3, t3, 63
   1ab62: 33 7e 6e 00  	and	t3, t3, t1
   1ab66: 01 4c        	mv	s8, zero
   1ab68: 65 bf        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1ab6a: 9c 42        	lw	a5, 0(a3)
   1ab6c: a3 03 01 06  	sb	zero, 103(sp)
   1ab70: a1 06        	addi	a3, a3, 8
   1ab72: 23 04 f1 08  	sb	a5, 136(sp)
   1ab76: 05 4e        	addi	t3, zero, 1
   1ab78: 05 43        	addi	t1, zero, 1
   1ab7a: 13 09 81 08  	addi	s2, sp, 136
   1ab7e: 7d b3        	j	0x1a92c <_svfiprintf_r+0x128>
   1ab80: 93 ec 0c 02  	ori	s9, s9, 32
   1ab84: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ab88: 81 2c        	sext.w	s9, s9
   1ab8a: 2d bb        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1ab8c: e1 77        	lui	a5, 1048568
   1ab8e: 93 c7 07 83  	xori	a5, a5, -2000
   1ab92: 23 14 f1 06  	sh	a5, 104(sp)
   1ab96: f5 67        	lui	a5, 29
   1ab98: 93 ec 2c 00  	ori	s9, s9, 2
   1ab9c: 93 87 87 77  	addi	a5, a5, 1912
   1aba0: 98 62        	ld	a4, 0(a3)
   1aba2: 81 2c        	sext.w	s9, s9
   1aba4: a1 06        	addi	a3, a3, 8
   1aba6: 3e e8        	sd	a5, 16(sp)
   1aba8: a3 03 01 06  	sb	zero, 103(sp)
   1abac: fd 57        	addi	a5, zero, -1
   1abae: 63 0b fc 34  	beq	s8, a5, 0x1af04 <_svfiprintf_r+0x700>
   1abb2: 93 f7 fc f7  	andi	a5, s9, -129
   1abb6: 81 27        	sext.w	a5, a5
   1abb8: 63 12 07 34  	bnez	a4, 0x1aefc <_svfiprintf_r+0x6f8>
   1abbc: 66 83        	mv	t1, s9
   1abbe: 09 47        	addi	a4, zero, 2
   1abc0: be 8c        	mv	s9, a5
   1abc2: 63 1f 0c 04  	bnez	s8, 0x1ac20 <_svfiprintf_r+0x41c>
   1abc6: 63 1c 07 52  	bnez	a4, 0x1b0fe <_svfiprintf_r+0x8fa>
   1abca: 13 73 13 00  	andi	t1, t1, 1
   1abce: 63 0b 03 58  	beqz	t1, 0x1b164 <_svfiprintf_r+0x960>
   1abd2: 93 07 00 03  	addi	a5, zero, 48
   1abd6: a3 05 f1 0e  	sb	a5, 235(sp)
   1abda: 1a 8e        	mv	t3, t1
   1abdc: 13 09 b1 0e  	addi	s2, sp, 235
   1abe0: 81 b7        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1abe2: 93 e7 0c 01  	ori	a5, s9, 16
   1abe6: 81 27        	sext.w	a5, a5
   1abe8: 13 f7 07 02  	andi	a4, a5, 32
   1abec: 13 86 86 00  	addi	a2, a3, 8
   1abf0: 09 e7        	bnez	a4, 0x1abfa <_svfiprintf_r+0x3f6>
   1abf2: 13 f7 07 01  	andi	a4, a5, 16
   1abf6: 63 00 07 5a  	beqz	a4, 0x1b196 <_svfiprintf_r+0x992>
   1abfa: 98 62        	ld	a4, 0(a3)
   1abfc: 93 fc f7 bf  	andi	s9, a5, -1025
   1ac00: a3 03 01 06  	sb	zero, 103(sp)
   1ac04: fd 56        	addi	a3, zero, -1
   1ac06: 81 2c        	sext.w	s9, s9
   1ac08: 63 02 dc 4a  	beq	s8, a3, 0x1b0ac <_svfiprintf_r+0x8a8>
   1ac0c: 93 f7 f7 b7  	andi	a5, a5, -1153
   1ac10: 81 27        	sext.w	a5, a5
   1ac12: 63 17 07 68  	bnez	a4, 0x1b2a0 <_svfiprintf_r+0xa9c>
   1ac16: 66 83        	mv	t1, s9
   1ac18: b2 86        	mv	a3, a2
   1ac1a: be 8c        	mv	s9, a5
   1ac1c: e3 05 0c fa  	beqz	s8, 0x1abc6 <_svfiprintf_r+0x3c2>
   1ac20: 85 47        	addi	a5, zero, 1
   1ac22: 63 0a f7 72  	beq	a4, a5, 0x1b356 <_svfiprintf_r+0xb52>
   1ac26: 89 47        	addi	a5, zero, 2
   1ac28: 63 0e f7 6a  	beq	a4, a5, 0x1b2e4 <_svfiprintf_r+0xae0>
   1ac2c: 01 47        	mv	a4, zero
   1ac2e: 41 a1        	j	0x1b0ae <_svfiprintf_r+0x8aa>
   1ac30: 13 e3 0c 01  	ori	t1, s9, 16
   1ac34: 01 23        	sext.w	t1, t1
   1ac36: 93 77 03 02  	andi	a5, t1, 32
   1ac3a: 93 8e 86 00  	addi	t4, a3, 8
   1ac3e: 89 e7        	bnez	a5, 0x1ac48 <_svfiprintf_r+0x444>
   1ac40: 93 77 03 01  	andi	a5, t1, 16
   1ac44: 63 81 07 54  	beqz	a5, 0x1b186 <_svfiprintf_r+0x982>
   1ac48: 9c 62        	ld	a5, 0(a3)
   1ac4a: a3 03 01 06  	sb	zero, 103(sp)
   1ac4e: 7d 57        	addi	a4, zero, -1
   1ac50: 9a 8c        	mv	s9, t1
   1ac52: e3 05 ec ea  	beq	s8, a4, 0x1aafc <_svfiprintf_r+0x2f8>
   1ac56: 93 7c f3 f7  	andi	s9, t1, -129
   1ac5a: 81 2c        	sext.w	s9, s9
   1ac5c: e3 90 07 ea  	bnez	a5, 0x1aafc <_svfiprintf_r+0x2f8>
   1ac60: f6 86        	mv	a3, t4
   1ac62: 05 47        	addi	a4, zero, 1
   1ac64: b9 bf        	j	0x1abc2 <_svfiprintf_r+0x3be>
   1ac66: 93 ec 0c 08  	ori	s9, s9, 128
   1ac6a: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ac6e: 81 2c        	sext.w	s9, s9
   1ac70: 91 b9        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1ac72: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ac76: 93 05 a0 02  	addi	a1, zero, 42
   1ac7a: 13 87 1d 00  	addi	a4, s11, 1
   1ac7e: 63 8b b7 6e  	beq	a5, a1, 0x1b374 <_svfiprintf_r+0xb70>
   1ac82: 1b 86 07 fd  	addiw	a2, a5, -48
   1ac86: 25 45        	addi	a0, zero, 9
   1ac88: ba 8d        	mv	s11, a4
   1ac8a: 01 4c        	mv	s8, zero
   1ac8c: a5 45        	addi	a1, zero, 9
   1ac8e: e3 6d c5 c2  	bltu	a0, a2, 0x1a8c8 <_svfiprintf_r+0xc4>
   1ac92: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ac96: 1b 17 2c 00  	slliw	a4, s8, 2
   1ac9a: 3b 0c 87 01  	addw	s8, a4, s8
   1ac9e: 1b 1c 1c 00  	slliw	s8, s8, 1
   1aca2: 3b 0c cc 00  	addw	s8, s8, a2
   1aca6: 1b 86 07 fd  	addiw	a2, a5, -48
   1acaa: 85 0d        	addi	s11, s11, 1
   1acac: e3 f3 c5 fe  	bgeu	a1, a2, 0x1ac92 <_svfiprintf_r+0x48e>
   1acb0: 21 b9        	j	0x1a8c8 <_svfiprintf_r+0xc4>
   1acb2: 13 07 b0 02  	addi	a4, zero, 43
   1acb6: 83 c7 0d 00  	lbu	a5, 0(s11)
   1acba: a3 03 e1 06  	sb	a4, 103(sp)
   1acbe: 19 b1        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1acc0: 5a 85        	mv	a0, s6
   1acc2: 36 f0        	sd	a3, 32(sp)
   1acc4: 46 ec        	sd	a7, 24(sp)
   1acc6: ef a0 3f 80  	jal	0x154c8 <_localeconv_r>
   1acca: 1c 65        	ld	a5, 8(a0)
   1accc: 3e 85        	mv	a0, a5
   1acce: be e0        	sd	a5, 64(sp)
   1acd0: ef c0 0f 9b  	jal	0x16e80 <strlen>
   1acd4: aa 87        	mv	a5, a0
   1acd6: 5a 85        	mv	a0, s6
   1acd8: 3e f8        	sd	a5, 48(sp)
   1acda: ef a0 ef fe  	jal	0x154c8 <_localeconv_r>
   1acde: 18 69        	ld	a4, 16(a0)
   1ace0: c2 77        	ld	a5, 48(sp)
   1ace2: e2 68        	ld	a7, 24(sp)
   1ace4: 3a fc        	sd	a4, 56(sp)
   1ace6: 82 76        	ld	a3, 32(sp)
   1ace8: 63 92 07 48  	bnez	a5, 0x1b16c <_svfiprintf_r+0x968>
   1acec: 83 c7 0d 00  	lbu	a5, 0(s11)
   1acf0: d1 be        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1acf2: 93 ec 1c 00  	ori	s9, s9, 1
   1acf6: 83 c7 0d 00  	lbu	a5, 0(s11)
   1acfa: 81 2c        	sext.w	s9, s9
   1acfc: e1 b6        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1acfe: 03 47 71 06  	lbu	a4, 103(sp)
   1ad02: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ad06: e3 1f 07 ba  	bnez	a4, 0x1a8c4 <_svfiprintf_r+0xc0>
   1ad0a: 13 07 00 02  	addi	a4, zero, 32
   1ad0e: a3 03 e1 06  	sb	a4, 103(sp)
   1ad12: 4d be        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1ad14: 93 f7 0c 02  	andi	a5, s9, 32
   1ad18: 98 62        	ld	a4, 0(a3)
   1ad1a: 85 e3        	bnez	a5, 0x1ad3a <_svfiprintf_r+0x536>
   1ad1c: 93 f7 0c 01  	andi	a5, s9, 16
   1ad20: 89 ef        	bnez	a5, 0x1ad3a <_svfiprintf_r+0x536>
   1ad22: 93 f7 0c 04  	andi	a5, s9, 64
   1ad26: 63 9f 07 5e  	bnez	a5, 0x1b324 <_svfiprintf_r+0xb20>
   1ad2a: 93 fc 0c 20  	andi	s9, s9, 512
   1ad2e: 63 88 0c 62  	beqz	s9, 0x1b35e <_svfiprintf_r+0xb5a>
   1ad32: 23 00 a7 01  	sb	s10, 0(a4)
   1ad36: a1 06        	addi	a3, a3, 8
   1ad38: 35 b6        	j	0x1a864 <_svfiprintf_r+0x60>
   1ad3a: 23 30 a7 01  	sd	s10, 0(a4)
   1ad3e: a1 06        	addi	a3, a3, 8
   1ad40: 15 b6        	j	0x1a864 <_svfiprintf_r+0x60>
   1ad42: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ad46: 13 07 c0 06  	addi	a4, zero, 108
   1ad4a: 63 89 e7 56  	beq	a5, a4, 0x1b2bc <_svfiprintf_r+0xab8>
   1ad4e: 93 ec 0c 01  	ori	s9, s9, 16
   1ad52: 81 2c        	sext.w	s9, s9
   1ad54: 85 be        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1ad56: 83 c7 0d 00  	lbu	a5, 0(s11)
   1ad5a: 13 07 80 06  	addi	a4, zero, 104
   1ad5e: 63 87 e7 54  	beq	a5, a4, 0x1b2ac <_svfiprintf_r+0xaa8>
   1ad62: 93 ec 0c 04  	ori	s9, s9, 64
   1ad66: 81 2c        	sext.w	s9, s9
   1ad68: b1 be        	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1ad6a: 3b 87 c8 41  	subw	a4, a7, t3
   1ad6e: e3 50 e0 c2  	blez	a4, 0x1a98e <_svfiprintf_r+0x18a>
   1ad72: 79 6f        	lui	t5, 30
   1ad74: 41 46        	addi	a2, zero, 16
   1ad76: 13 0f 0f 51  	addi	t5, t5, 1296
   1ad7a: 63 51 e6 06  	bge	a2, a4, 0x1addc <_svfiprintf_r+0x5d8>
   1ad7e: 26 86        	mv	a2, s1
   1ad80: b6 e4        	sd	a3, 72(sp)
   1ad82: c1 4e        	addi	t4, zero, 16
   1ad84: a2 86        	mv	a3, s0
   1ad86: 9d 4f        	addi	t6, zero, 7
   1ad88: 46 ec        	sd	a7, 24(sp)
   1ad8a: 72 f0        	sd	t3, 32(sp)
   1ad8c: 1a f4        	sd	t1, 40(sp)
   1ad8e: 3a 84        	mv	s0, a4
   1ad90: fa 84        	mv	s1, t5
   1ad92: 21 a0        	j	0x1ad9a <_svfiprintf_r+0x596>
   1ad94: 41 34        	addiw	s0, s0, -16
   1ad96: 63 db 8e 02  	bge	t4, s0, 0x1adcc <_svfiprintf_r+0x5c8>
   1ad9a: 85 27        	addiw	a5, a5, 1
   1ad9c: 41 06        	addi	a2, a2, 16
   1ad9e: 84 e2        	sd	s1, 0(a3)
   1ada0: 23 b4 d6 01  	sd	t4, 8(a3)
   1ada4: 32 e1        	sd	a2, 128(sp)
   1ada6: be dc        	sw	a5, 120(sp)
   1ada8: c1 06        	addi	a3, a3, 16
   1adaa: e3 d5 ff fe  	bge	t6, a5, 0x1ad94 <_svfiprintf_r+0x590>
   1adae: 90 18        	addi	a2, sp, 112
   1adb0: d2 85        	mv	a1, s4
   1adb2: 5a 85        	mv	a0, s6
   1adb4: ef f0 bf 8f  	jal	0x1a6ae <__ssprint_r>
   1adb8: e3 1a 05 ca  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1adbc: c1 4e        	addi	t4, zero, 16
   1adbe: 41 34        	addiw	s0, s0, -16
   1adc0: 0a 66        	ld	a2, 128(sp)
   1adc2: e6 57        	lw	a5, 120(sp)
   1adc4: d6 86        	mv	a3, s5
   1adc6: 9d 4f        	addi	t6, zero, 7
   1adc8: e3 c9 8e fc  	blt	t4, s0, 0x1ad9a <_svfiprintf_r+0x596>
   1adcc: 22 87        	mv	a4, s0
   1adce: e2 68        	ld	a7, 24(sp)
   1add0: 36 84        	mv	s0, a3
   1add2: 02 7e        	ld	t3, 32(sp)
   1add4: 22 73        	ld	t1, 40(sp)
   1add6: a6 66        	ld	a3, 72(sp)
   1add8: 26 8f        	mv	t5, s1
   1adda: b2 84        	mv	s1, a2
   1addc: ba 94        	add	s1, s1, a4
   1adde: 85 27        	addiw	a5, a5, 1
   1ade0: 18 e4        	sd	a4, 8(s0)
   1ade2: 23 30 e4 01  	sd	t5, 0(s0)
   1ade6: 26 e1        	sd	s1, 128(sp)
   1ade8: be dc        	sw	a5, 120(sp)
   1adea: 1d 47        	addi	a4, zero, 7
   1adec: 41 04        	addi	s0, s0, 16
   1adee: e3 50 f7 ba  	bge	a4, a5, 0x1a98e <_svfiprintf_r+0x18a>
   1adf2: 90 18        	addi	a2, sp, 112
   1adf4: d2 85        	mv	a1, s4
   1adf6: 5a 85        	mv	a0, s6
   1adf8: b6 e4        	sd	a3, 72(sp)
   1adfa: 1a f4        	sd	t1, 40(sp)
   1adfc: 72 f0        	sd	t3, 32(sp)
   1adfe: 46 ec        	sd	a7, 24(sp)
   1ae00: ef f0 ff 8a  	jal	0x1a6ae <__ssprint_r>
   1ae04: e3 14 05 c6  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1ae08: 22 73        	ld	t1, 40(sp)
   1ae0a: 8a 64        	ld	s1, 128(sp)
   1ae0c: e6 57        	lw	a5, 120(sp)
   1ae0e: 3b 0c 6c 40  	subw	s8, s8, t1
   1ae12: a6 66        	ld	a3, 72(sp)
   1ae14: 02 7e        	ld	t3, 32(sp)
   1ae16: e2 68        	ld	a7, 24(sp)
   1ae18: 56 84        	mv	s0, s5
   1ae1a: e3 5e 80 b7  	blez	s8, 0x1a996 <_svfiprintf_r+0x192>
   1ae1e: 79 6f        	lui	t5, 30
   1ae20: 13 0f 0f 51  	addi	t5, t5, 1296
   1ae24: 63 d2 89 07  	bge	s3, s8, 0x1ae88 <_svfiprintf_r+0x684>
   1ae28: 36 86        	mv	a2, a3
   1ae2a: 22 87        	mv	a4, s0
   1ae2c: a6 86        	mv	a3, s1
   1ae2e: 62 84        	mv	s0, s8
   1ae30: 9d 4e        	addi	t4, zero, 7
   1ae32: 46 ec        	sd	a7, 24(sp)
   1ae34: 72 f0        	sd	t3, 32(sp)
   1ae36: 1a f4        	sd	t1, 40(sp)
   1ae38: 5e 8c        	mv	s8, s7
   1ae3a: b2 84        	mv	s1, a2
   1ae3c: 21 a0        	j	0x1ae44 <_svfiprintf_r+0x640>
   1ae3e: 41 34        	addiw	s0, s0, -16
   1ae40: 63 db 89 02  	bge	s3, s0, 0x1ae76 <_svfiprintf_r+0x672>
   1ae44: 85 27        	addiw	a5, a5, 1
   1ae46: c1 06        	addi	a3, a3, 16
   1ae48: 23 30 77 01  	sd	s7, 0(a4)
   1ae4c: 23 34 37 01  	sd	s3, 8(a4)
   1ae50: 36 e1        	sd	a3, 128(sp)
   1ae52: be dc        	sw	a5, 120(sp)
   1ae54: 41 07        	addi	a4, a4, 16
   1ae56: e3 d4 fe fe  	bge	t4, a5, 0x1ae3e <_svfiprintf_r+0x63a>
   1ae5a: 90 18        	addi	a2, sp, 112
   1ae5c: d2 85        	mv	a1, s4
   1ae5e: 5a 85        	mv	a0, s6
   1ae60: ef f0 ff 84  	jal	0x1a6ae <__ssprint_r>
   1ae64: e3 14 05 c0  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1ae68: 41 34        	addiw	s0, s0, -16
   1ae6a: 8a 66        	ld	a3, 128(sp)
   1ae6c: e6 57        	lw	a5, 120(sp)
   1ae6e: 56 87        	mv	a4, s5
   1ae70: 9d 4e        	addi	t4, zero, 7
   1ae72: e3 c9 89 fc  	blt	s3, s0, 0x1ae44 <_svfiprintf_r+0x640>
   1ae76: e2 68        	ld	a7, 24(sp)
   1ae78: 02 7e        	ld	t3, 32(sp)
   1ae7a: 22 73        	ld	t1, 40(sp)
   1ae7c: 26 86        	mv	a2, s1
   1ae7e: 62 8f        	mv	t5, s8
   1ae80: b6 84        	mv	s1, a3
   1ae82: 22 8c        	mv	s8, s0
   1ae84: b2 86        	mv	a3, a2
   1ae86: 3a 84        	mv	s0, a4
   1ae88: 85 27        	addiw	a5, a5, 1
   1ae8a: e2 94        	add	s1, s1, s8
   1ae8c: 23 30 e4 01  	sd	t5, 0(s0)
   1ae90: 23 34 84 01  	sd	s8, 8(s0)
   1ae94: 26 e1        	sd	s1, 128(sp)
   1ae96: be dc        	sw	a5, 120(sp)
   1ae98: 1d 47        	addi	a4, zero, 7
   1ae9a: 41 04        	addi	s0, s0, 16
   1ae9c: e3 5d f7 ae  	bge	a4, a5, 0x1a996 <_svfiprintf_r+0x192>
   1aea0: 90 18        	addi	a2, sp, 112
   1aea2: d2 85        	mv	a1, s4
   1aea4: 5a 85        	mv	a0, s6
   1aea6: b6 e4        	sd	a3, 72(sp)
   1aea8: 1a f4        	sd	t1, 40(sp)
   1aeaa: 72 f0        	sd	t3, 32(sp)
   1aeac: 46 ec        	sd	a7, 24(sp)
   1aeae: ef f0 1f 80  	jal	0x1a6ae <__ssprint_r>
   1aeb2: e3 1d 05 ba  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1aeb6: 8a 64        	ld	s1, 128(sp)
   1aeb8: e6 57        	lw	a5, 120(sp)
   1aeba: a6 66        	ld	a3, 72(sp)
   1aebc: 22 73        	ld	t1, 40(sp)
   1aebe: 02 7e        	ld	t3, 32(sp)
   1aec0: e2 68        	ld	a7, 24(sp)
   1aec2: 56 84        	mv	s0, s5
   1aec4: c9 bc        	j	0x1a996 <_svfiprintf_r+0x192>
   1aec6: 90 18        	addi	a2, sp, 112
   1aec8: d2 85        	mv	a1, s4
   1aeca: 5a 85        	mv	a0, s6
   1aecc: 36 f4        	sd	a3, 40(sp)
   1aece: 72 f0        	sd	t3, 32(sp)
   1aed0: 46 ec        	sd	a7, 24(sp)
   1aed2: ef f0 cf fd  	jal	0x1a6ae <__ssprint_r>
   1aed6: e3 1b 05 b8  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1aeda: 8a 64        	ld	s1, 128(sp)
   1aedc: a2 76        	ld	a3, 40(sp)
   1aede: 02 7e        	ld	t3, 32(sp)
   1aee0: e2 68        	ld	a7, 24(sp)
   1aee2: 56 84        	mv	s0, s5
   1aee4: f1 b4        	j	0x1a9b0 <_svfiprintf_r+0x1ac>
   1aee6: 90 18        	addi	a2, sp, 112
   1aee8: d2 85        	mv	a1, s4
   1aeea: 5a 85        	mv	a0, s6
   1aeec: 36 ec        	sd	a3, 24(sp)
   1aeee: ef f0 0f fc  	jal	0x1a6ae <__ssprint_r>
   1aef2: e3 1d 05 b6  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1aef6: 8a 64        	ld	s1, 128(sp)
   1aef8: e2 66        	ld	a3, 24(sp)
   1aefa: d9 bc        	j	0x1a9d0 <_svfiprintf_r+0x1cc>
   1aefc: be 8c        	mv	s9, a5
   1aefe: 89 47        	addi	a5, zero, 2
   1af00: 63 87 07 1a  	beqz	a5, 0x1b0ae <_svfiprintf_r+0x8aa>
   1af04: 42 66        	ld	a2, 16(sp)
   1af06: 13 03 c1 0e  	addi	t1, sp, 236
   1af0a: 1a 89        	mv	s2, t1
   1af0c: 93 77 f7 00  	andi	a5, a4, 15
   1af10: b2 97        	add	a5, a5, a2
   1af12: 83 c7 07 00  	lbu	a5, 0(a5)
   1af16: 11 83        	srli	a4, a4, 4
   1af18: 7d 19        	addi	s2, s2, -1
   1af1a: 23 00 f9 00  	sb	a5, 0(s2)
   1af1e: 7d f7        	bnez	a4, 0x1af0c <_svfiprintf_r+0x708>
   1af20: 3b 03 23 41  	subw	t1, t1, s2
   1af24: 1b 0e 0c 00  	sext.w	t3, s8
   1af28: e3 5c 6c be  	bge	s8, t1, 0x1ab20 <_svfiprintf_r+0x31c>
   1af2c: 1b 0e 03 00  	sext.w	t3, t1
   1af30: c5 be        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1af32: f9 6e        	lui	t4, 30
   1af34: 41 46        	addi	a2, zero, 16
   1af36: 93 8e 0e 52  	addi	t4, t4, 1312
   1af3a: 63 55 e6 06  	bge	a2, a4, 0x1afa4 <_svfiprintf_r+0x7a0>
   1af3e: 26 86        	mv	a2, s1
   1af40: b6 ec        	sd	a3, 88(sp)
   1af42: 41 4f        	addi	t5, zero, 16
   1af44: a2 86        	mv	a3, s0
   1af46: 9d 43        	addi	t2, zero, 7
   1af48: 16 ec        	sd	t0, 24(sp)
   1af4a: 7e f0        	sd	t6, 32(sp)
   1af4c: 46 f4        	sd	a7, 40(sp)
   1af4e: f2 e4        	sd	t3, 72(sp)
   1af50: 9a e8        	sd	t1, 80(sp)
   1af52: 3a 84        	mv	s0, a4
   1af54: f6 84        	mv	s1, t4
   1af56: 21 a0        	j	0x1af5e <_svfiprintf_r+0x75a>
   1af58: 41 34        	addiw	s0, s0, -16
   1af5a: 63 5b 8f 02  	bge	t5, s0, 0x1af90 <_svfiprintf_r+0x78c>
   1af5e: 85 27        	addiw	a5, a5, 1
   1af60: 41 06        	addi	a2, a2, 16
   1af62: 84 e2        	sd	s1, 0(a3)
   1af64: 23 b4 e6 01  	sd	t5, 8(a3)
   1af68: 32 e1        	sd	a2, 128(sp)
   1af6a: be dc        	sw	a5, 120(sp)
   1af6c: c1 06        	addi	a3, a3, 16
   1af6e: e3 d5 f3 fe  	bge	t2, a5, 0x1af58 <_svfiprintf_r+0x754>
   1af72: 90 18        	addi	a2, sp, 112
   1af74: d2 85        	mv	a1, s4
   1af76: 5a 85        	mv	a0, s6
   1af78: ef f0 6f f3  	jal	0x1a6ae <__ssprint_r>
   1af7c: e3 18 05 ae  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1af80: 41 4f        	addi	t5, zero, 16
   1af82: 41 34        	addiw	s0, s0, -16
   1af84: 0a 66        	ld	a2, 128(sp)
   1af86: e6 57        	lw	a5, 120(sp)
   1af88: d6 86        	mv	a3, s5
   1af8a: 9d 43        	addi	t2, zero, 7
   1af8c: e3 49 8f fc  	blt	t5, s0, 0x1af5e <_svfiprintf_r+0x75a>
   1af90: 22 87        	mv	a4, s0
   1af92: e2 62        	ld	t0, 24(sp)
   1af94: 36 84        	mv	s0, a3
   1af96: 82 7f        	ld	t6, 32(sp)
   1af98: a2 78        	ld	a7, 40(sp)
   1af9a: 26 6e        	ld	t3, 72(sp)
   1af9c: 46 63        	ld	t1, 80(sp)
   1af9e: e6 66        	ld	a3, 88(sp)
   1afa0: a6 8e        	mv	t4, s1
   1afa2: b2 84        	mv	s1, a2
   1afa4: ba 94        	add	s1, s1, a4
   1afa6: 85 27        	addiw	a5, a5, 1
   1afa8: 18 e4        	sd	a4, 8(s0)
   1afaa: 23 30 d4 01  	sd	t4, 0(s0)
   1afae: 26 e1        	sd	s1, 128(sp)
   1afb0: be dc        	sw	a5, 120(sp)
   1afb2: 1d 47        	addi	a4, zero, 7
   1afb4: 41 04        	addi	s0, s0, 16
   1afb6: e3 5a f7 98  	bge	a4, a5, 0x1a94a <_svfiprintf_r+0x146>
   1afba: 90 18        	addi	a2, sp, 112
   1afbc: d2 85        	mv	a1, s4
   1afbe: 5a 85        	mv	a0, s6
   1afc0: b6 ec        	sd	a3, 88(sp)
   1afc2: 9a e8        	sd	t1, 80(sp)
   1afc4: f2 e4        	sd	t3, 72(sp)
   1afc6: 46 f4        	sd	a7, 40(sp)
   1afc8: 7e f0        	sd	t6, 32(sp)
   1afca: 16 ec        	sd	t0, 24(sp)
   1afcc: ef f0 2f ee  	jal	0x1a6ae <__ssprint_r>
   1afd0: e3 1e 05 a8  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1afd4: 8a 64        	ld	s1, 128(sp)
   1afd6: e6 57        	lw	a5, 120(sp)
   1afd8: e6 66        	ld	a3, 88(sp)
   1afda: 46 63        	ld	t1, 80(sp)
   1afdc: 26 6e        	ld	t3, 72(sp)
   1afde: a2 78        	ld	a7, 40(sp)
   1afe0: 82 7f        	ld	t6, 32(sp)
   1afe2: e2 62        	ld	t0, 24(sp)
   1afe4: 56 84        	mv	s0, s5
   1afe6: 95 b2        	j	0x1a94a <_svfiprintf_r+0x146>
   1afe8: 90 18        	addi	a2, sp, 112
   1afea: d2 85        	mv	a1, s4
   1afec: 5a 85        	mv	a0, s6
   1afee: b6 e8        	sd	a3, 80(sp)
   1aff0: 9a e4        	sd	t1, 72(sp)
   1aff2: 72 f4        	sd	t3, 40(sp)
   1aff4: 46 f0        	sd	a7, 32(sp)
   1aff6: 7e ec        	sd	t6, 24(sp)
   1aff8: ef f0 6f eb  	jal	0x1a6ae <__ssprint_r>
   1affc: e3 18 05 a6  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1b000: 8a 64        	ld	s1, 128(sp)
   1b002: e6 57        	lw	a5, 120(sp)
   1b004: c6 66        	ld	a3, 80(sp)
   1b006: 26 63        	ld	t1, 72(sp)
   1b008: 22 7e        	ld	t3, 40(sp)
   1b00a: 82 78        	ld	a7, 32(sp)
   1b00c: e2 6f        	ld	t6, 24(sp)
   1b00e: 56 84        	mv	s0, s5
   1b010: 9d ba        	j	0x1a986 <_svfiprintf_r+0x182>
   1b012: f9 6e        	lui	t4, 30
   1b014: 41 47        	addi	a4, zero, 16
   1b016: e6 57        	lw	a5, 120(sp)
   1b018: 93 8e 0e 52  	addi	t4, t4, 1312
   1b01c: 63 5d 27 05  	bge	a4, s2, 0x1b076 <_svfiprintf_r+0x872>
   1b020: 22 87        	mv	a4, s0
   1b022: 41 4c        	addi	s8, zero, 16
   1b024: 4a 84        	mv	s0, s2
   1b026: 9d 4c        	addi	s9, zero, 7
   1b028: 36 89        	mv	s2, a3
   1b02a: 46 ec        	sd	a7, 24(sp)
   1b02c: a6 86        	mv	a3, s1
   1b02e: 72 f0        	sd	t3, 32(sp)
   1b030: f6 84        	mv	s1, t4
   1b032: 21 a0        	j	0x1b03a <_svfiprintf_r+0x836>
   1b034: 41 34        	addiw	s0, s0, -16
   1b036: 63 59 8c 02  	bge	s8, s0, 0x1b068 <_svfiprintf_r+0x864>
   1b03a: 85 27        	addiw	a5, a5, 1
   1b03c: c1 06        	addi	a3, a3, 16
   1b03e: 04 e3        	sd	s1, 0(a4)
   1b040: 23 34 87 01  	sd	s8, 8(a4)
   1b044: 36 e1        	sd	a3, 128(sp)
   1b046: be dc        	sw	a5, 120(sp)
   1b048: 41 07        	addi	a4, a4, 16
   1b04a: e3 d5 fc fe  	bge	s9, a5, 0x1b034 <_svfiprintf_r+0x830>
   1b04e: 90 18        	addi	a2, sp, 112
   1b050: d2 85        	mv	a1, s4
   1b052: 5a 85        	mv	a0, s6
   1b054: ef f0 af e5  	jal	0x1a6ae <__ssprint_r>
   1b058: e3 1a 05 a0  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1b05c: 41 34        	addiw	s0, s0, -16
   1b05e: 8a 66        	ld	a3, 128(sp)
   1b060: e6 57        	lw	a5, 120(sp)
   1b062: 56 87        	mv	a4, s5
   1b064: e3 4b 8c fc  	blt	s8, s0, 0x1b03a <_svfiprintf_r+0x836>
   1b068: e2 68        	ld	a7, 24(sp)
   1b06a: 02 7e        	ld	t3, 32(sp)
   1b06c: a6 8e        	mv	t4, s1
   1b06e: b6 84        	mv	s1, a3
   1b070: ca 86        	mv	a3, s2
   1b072: 22 89        	mv	s2, s0
   1b074: 3a 84        	mv	s0, a4
   1b076: 1b 87 17 00  	addiw	a4, a5, 1
   1b07a: ca 94        	add	s1, s1, s2
   1b07c: 23 30 d4 01  	sd	t4, 0(s0)
   1b080: 23 34 24 01  	sd	s2, 8(s0)
   1b084: 26 e1        	sd	s1, 128(sp)
   1b086: ba dc        	sw	a4, 120(sp)
   1b088: 9d 47        	addi	a5, zero, 7
   1b08a: e3 db e7 92  	bge	a5, a4, 0x1a9c0 <_svfiprintf_r+0x1bc>
   1b08e: 90 18        	addi	a2, sp, 112
   1b090: d2 85        	mv	a1, s4
   1b092: 5a 85        	mv	a0, s6
   1b094: 36 f4        	sd	a3, 40(sp)
   1b096: 72 f0        	sd	t3, 32(sp)
   1b098: 46 ec        	sd	a7, 24(sp)
   1b09a: ef f0 4f e1  	jal	0x1a6ae <__ssprint_r>
   1b09e: e3 17 05 9c  	bnez	a0, 0x1aa6c <_svfiprintf_r+0x268>
   1b0a2: 8a 64        	ld	s1, 128(sp)
   1b0a4: a2 76        	ld	a3, 40(sp)
   1b0a6: 02 7e        	ld	t3, 32(sp)
   1b0a8: e2 68        	ld	a7, 24(sp)
   1b0aa: 19 ba        	j	0x1a9c0 <_svfiprintf_r+0x1bc>
   1b0ac: b2 86        	mv	a3, a2
   1b0ae: 13 03 c1 0e  	addi	t1, sp, 236
   1b0b2: 1a 89        	mv	s2, t1
   1b0b4: 93 77 77 00  	andi	a5, a4, 7
   1b0b8: 93 87 07 03  	addi	a5, a5, 48
   1b0bc: a3 0f f9 fe  	sb	a5, -1(s2)
   1b0c0: 0d 83        	srli	a4, a4, 3
   1b0c2: 4a 86        	mv	a2, s2
   1b0c4: 7d 19        	addi	s2, s2, -1
   1b0c6: 7d f7        	bnez	a4, 0x1b0b4 <_svfiprintf_r+0x8b0>
   1b0c8: 13 f7 1c 00  	andi	a4, s9, 1
   1b0cc: 05 c3        	beqz	a4, 0x1b0ec <_svfiprintf_r+0x8e8>
   1b0ce: 13 07 00 03  	addi	a4, zero, 48
   1b0d2: 63 8d e7 00  	beq	a5, a4, 0x1b0ec <_svfiprintf_r+0x8e8>
   1b0d6: 79 16        	addi	a2, a2, -2
   1b0d8: 3b 03 c3 40  	subw	t1, t1, a2
   1b0dc: a3 0f e9 fe  	sb	a4, -1(s2)
   1b0e0: 1b 0e 03 00  	sext.w	t3, t1
   1b0e4: 63 41 83 0d  	blt	t1, s8, 0x1b1a6 <_svfiprintf_r+0x9a2>
   1b0e8: 32 89        	mv	s2, a2
   1b0ea: 1d bc        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b0ec: 3b 03 23 41  	subw	t1, t1, s2
   1b0f0: 1b 0e 03 00  	sext.w	t3, t1
   1b0f4: e3 56 83 a3  	bge	t1, s8, 0x1ab20 <_svfiprintf_r+0x31c>
   1b0f8: 1b 0e 0c 00  	sext.w	t3, s8
   1b0fc: 15 b4        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b0fe: 01 4e        	mv	t3, zero
   1b100: 01 43        	mv	t1, zero
   1b102: 13 09 c1 0e  	addi	s2, sp, 236
   1b106: 29 bc        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b108: 13 03 c1 0e  	addi	t1, sp, 236
   1b10c: 62 78        	ld	a6, 56(sp)
   1b10e: 13 ff 0c 40  	andi	t5, s9, 1024
   1b112: 81 45        	mv	a1, zero
   1b114: 1a 86        	mv	a2, t1
   1b116: a9 46        	addi	a3, zero, 10
   1b118: 25 4e        	addi	t3, zero, 9
   1b11a: 93 0f f0 0f  	addi	t6, zero, 255
   1b11e: 39 a0        	j	0x1b12c <_svfiprintf_r+0x928>
   1b120: 33 d7 d7 02  	divu	a4, a5, a3
   1b124: 63 77 fe 02  	bgeu	t3, a5, 0x1b152 <_svfiprintf_r+0x94e>
   1b128: ba 87        	mv	a5, a4
   1b12a: 4a 86        	mv	a2, s2
   1b12c: 13 09 f6 ff  	addi	s2, a2, -1
   1b130: 85 25        	addiw	a1, a1, 1
   1b132: 33 f7 d7 02  	remu	a4, a5, a3
   1b136: 1b 07 07 03  	addiw	a4, a4, 48
   1b13a: a3 0f e6 fe  	sb	a4, -1(a2)
   1b13e: e3 01 0f fe  	beqz	t5, 0x1b120 <_svfiprintf_r+0x91c>
   1b142: 03 47 08 00  	lbu	a4, 0(a6)
   1b146: e3 1d b7 fc  	bne	a4, a1, 0x1b120 <_svfiprintf_r+0x91c>
   1b14a: e3 0b f7 fd  	beq	a4, t6, 0x1b120 <_svfiprintf_r+0x91c>
   1b14e: 63 67 fe 10  	bltu	t3, a5, 0x1b25c <_svfiprintf_r+0xa58>
   1b152: 3b 03 23 41  	subw	t1, t1, s2
   1b156: 42 fc        	sd	a6, 56(sp)
   1b158: 1b 0e 03 00  	sext.w	t3, t1
   1b15c: 63 4c 83 0d  	blt	t1, s8, 0x1b234 <_svfiprintf_r+0xa30>
   1b160: f6 86        	mv	a3, t4
   1b162: 7d ba        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b164: 01 4e        	mv	t3, zero
   1b166: 13 09 c1 0e  	addi	s2, sp, 236
   1b16a: 5d ba        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b16c: 83 c7 0d 00  	lbu	a5, 0(s11)
   1b170: 63 0a 07 f4  	beqz	a4, 0x1a8c4 <_svfiprintf_r+0xc0>
   1b174: 03 47 07 00  	lbu	a4, 0(a4)
   1b178: 63 06 07 f4  	beqz	a4, 0x1a8c4 <_svfiprintf_r+0xc0>
   1b17c: 93 ec 0c 40  	ori	s9, s9, 1024
   1b180: 81 2c        	sext.w	s9, s9
   1b182: 6f f0 2f f4  	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1b186: 13 77 03 04  	andi	a4, t1, 64
   1b18a: 9c 42        	lw	a5, 0(a3)
   1b18c: 63 04 07 16  	beqz	a4, 0x1b2f4 <_svfiprintf_r+0xaf0>
   1b190: c2 17        	slli	a5, a5, 48
   1b192: c1 93        	srli	a5, a5, 48
   1b194: 5d bc        	j	0x1ac4a <_svfiprintf_r+0x446>
   1b196: 93 f5 07 04  	andi	a1, a5, 64
   1b19a: 98 42        	lw	a4, 0(a3)
   1b19c: 63 82 05 16  	beqz	a1, 0x1b300 <_svfiprintf_r+0xafc>
   1b1a0: 42 17        	slli	a4, a4, 48
   1b1a2: 41 93        	srli	a4, a4, 48
   1b1a4: a1 bc        	j	0x1abfc <_svfiprintf_r+0x3f8>
   1b1a6: 1b 0e 0c 00  	sext.w	t3, s8
   1b1aa: 32 89        	mv	s2, a2
   1b1ac: 95 ba        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b1ae: 75 67        	lui	a4, 29
   1b1b0: 13 07 07 76  	addi	a4, a4, 1888
   1b1b4: 3a e8        	sd	a4, 16(sp)
   1b1b6: 13 f7 0c 02  	andi	a4, s9, 32
   1b1ba: 13 86 86 00  	addi	a2, a3, 8
   1b1be: 31 eb        	bnez	a4, 0x1b212 <_svfiprintf_r+0xa0e>
   1b1c0: 13 f7 0c 01  	andi	a4, s9, 16
   1b1c4: 39 e7        	bnez	a4, 0x1b212 <_svfiprintf_r+0xa0e>
   1b1c6: 93 f5 0c 04  	andi	a1, s9, 64
   1b1ca: 98 42        	lw	a4, 0(a3)
   1b1cc: 63 8e 05 10  	beqz	a1, 0x1b2e8 <_svfiprintf_r+0xae4>
   1b1d0: 42 17        	slli	a4, a4, 48
   1b1d2: 41 93        	srli	a4, a4, 48
   1b1d4: 93 f6 1c 00  	andi	a3, s9, 1
   1b1d8: 99 ca        	beqz	a3, 0x1b1ee <_svfiprintf_r+0x9ea>
   1b1da: 11 cb        	beqz	a4, 0x1b1ee <_svfiprintf_r+0x9ea>
   1b1dc: 93 ec 2c 00  	ori	s9, s9, 2
   1b1e0: 93 06 00 03  	addi	a3, zero, 48
   1b1e4: 23 04 d1 06  	sb	a3, 104(sp)
   1b1e8: a3 04 f1 06  	sb	a5, 105(sp)
   1b1ec: 81 2c        	sext.w	s9, s9
   1b1ee: 93 fc fc bf  	andi	s9, s9, -1025
   1b1f2: 81 2c        	sext.w	s9, s9
   1b1f4: b2 86        	mv	a3, a2
   1b1f6: 4d ba        	j	0x1aba8 <_svfiprintf_r+0x3a4>
   1b1f8: e6 87        	mv	a5, s9
   1b1fa: fd b2        	j	0x1abe8 <_svfiprintf_r+0x3e4>
   1b1fc: 66 83        	mv	t1, s9
   1b1fe: 25 bc        	j	0x1ac36 <_svfiprintf_r+0x432>
   1b200: 75 67        	lui	a4, 29
   1b202: 13 07 87 77  	addi	a4, a4, 1912
   1b206: 3a e8        	sd	a4, 16(sp)
   1b208: 13 f7 0c 02  	andi	a4, s9, 32
   1b20c: 13 86 86 00  	addi	a2, a3, 8
   1b210: 45 db        	beqz	a4, 0x1b1c0 <_svfiprintf_r+0x9bc>
   1b212: 98 62        	ld	a4, 0(a3)
   1b214: c1 b7        	j	0x1b1d4 <_svfiprintf_r+0x9d0>
   1b216: 4a 85        	mv	a0, s2
   1b218: ef b0 9f c6  	jal	0x16e80 <strlen>
   1b21c: 1b 03 05 00  	sext.w	t1, a0
   1b220: 13 4e f3 ff  	not	t3, t1
   1b224: 13 5e fe 43  	srai	t3, t3, 63
   1b228: 82 76        	ld	a3, 32(sp)
   1b22a: e2 68        	ld	a7, 24(sp)
   1b22c: 33 7e 6e 00  	and	t3, t3, t1
   1b230: 01 4c        	mv	s8, zero
   1b232: fd b0        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b234: 1b 0e 0c 00  	sext.w	t3, s8
   1b238: f6 86        	mv	a3, t4
   1b23a: dd b0        	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b23c: 93 05 00 04  	addi	a1, zero, 64
   1b240: ef a0 af cd  	jal	0x1571a <_malloc_r>
   1b244: 23 30 aa 00  	sd	a0, 0(s4)
   1b248: 23 3c aa 00  	sd	a0, 24(s4)
   1b24c: 63 0e 05 10  	beqz	a0, 0x1b368 <_svfiprintf_r+0xb64>
   1b250: 13 07 00 04  	addi	a4, zero, 64
   1b254: 23 20 ea 02  	sw	a4, 32(s4)
   1b258: 6f f0 0f de  	j	0x1a838 <_svfiprintf_r+0x34>
   1b25c: be e4        	sd	a5, 72(sp)
   1b25e: c2 77        	ld	a5, 48(sp)
   1b260: 86 65        	ld	a1, 64(sp)
   1b262: 9a e8        	sd	t1, 80(sp)
   1b264: 33 09 f9 40  	sub	s2, s2, a5
   1b268: 3e 86        	mv	a2, a5
   1b26a: 4a 85        	mv	a0, s2
   1b26c: 76 fc        	sd	t4, 56(sp)
   1b26e: 7a f4        	sd	t5, 40(sp)
   1b270: 46 f0        	sd	a7, 32(sp)
   1b272: 42 ec        	sd	a6, 24(sp)
   1b274: ef b0 df c9  	jal	0x16f10 <strncpy>
   1b278: 62 68        	ld	a6, 24(sp)
   1b27a: a6 67        	ld	a5, 72(sp)
   1b27c: 29 47        	addi	a4, zero, 10
   1b27e: 03 46 18 00  	lbu	a2, 1(a6)
   1b282: 33 d7 e7 02  	divu	a4, a5, a4
   1b286: 82 78        	ld	a7, 32(sp)
   1b288: b3 37 c0 00  	snez	a5, a2
   1b28c: 22 7f        	ld	t5, 40(sp)
   1b28e: e2 7e        	ld	t4, 56(sp)
   1b290: 46 63        	ld	t1, 80(sp)
   1b292: 3e 98        	add	a6, a6, a5
   1b294: 81 45        	mv	a1, zero
   1b296: a9 46        	addi	a3, zero, 10
   1b298: 25 4e        	addi	t3, zero, 9
   1b29a: 93 0f f0 0f  	addi	t6, zero, 255
   1b29e: 69 b5        	j	0x1b128 <_svfiprintf_r+0x924>
   1b2a0: be 8c        	mv	s9, a5
   1b2a2: 81 47        	mv	a5, zero
   1b2a4: b2 86        	mv	a3, a2
   1b2a6: e3 84 07 e0  	beqz	a5, 0x1b0ae <_svfiprintf_r+0x8aa>
   1b2aa: a9 b9        	j	0x1af04 <_svfiprintf_r+0x700>
   1b2ac: 93 ec 0c 20  	ori	s9, s9, 512
   1b2b0: 83 c7 1d 00  	lbu	a5, 1(s11)
   1b2b4: 81 2c        	sext.w	s9, s9
   1b2b6: 85 0d        	addi	s11, s11, 1
   1b2b8: 6f f0 cf e0  	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1b2bc: 93 ec 0c 02  	ori	s9, s9, 32
   1b2c0: 83 c7 1d 00  	lbu	a5, 1(s11)
   1b2c4: 81 2c        	sext.w	s9, s9
   1b2c6: 85 0d        	addi	s11, s11, 1
   1b2c8: 6f f0 cf df  	j	0x1a8c4 <_svfiprintf_r+0xc0>
   1b2cc: 99 47        	addi	a5, zero, 6
   1b2ce: 1b 03 0c 00  	sext.w	t1, s8
   1b2d2: 63 f3 87 01  	bgeu	a5, s8, 0x1b2d8 <_svfiprintf_r+0xad4>
   1b2d6: 19 43        	addi	t1, zero, 6
   1b2d8: 75 69        	lui	s2, 29
   1b2da: 1a 8e        	mv	t3, t1
   1b2dc: 13 09 09 79  	addi	s2, s2, 1936
   1b2e0: 6f f0 cf e4  	j	0x1a92c <_svfiprintf_r+0x128>
   1b2e4: 01 47        	mv	a4, zero
   1b2e6: 39 b9        	j	0x1af04 <_svfiprintf_r+0x700>
   1b2e8: 93 f6 0c 20  	andi	a3, s9, 512
   1b2ec: a1 ce        	beqz	a3, 0x1b344 <_svfiprintf_r+0xb40>
   1b2ee: 13 77 f7 0f  	andi	a4, a4, 255
   1b2f2: cd b5        	j	0x1b1d4 <_svfiprintf_r+0x9d0>
   1b2f4: 13 77 03 20  	andi	a4, t1, 512
   1b2f8: 39 c3        	beqz	a4, 0x1b33e <_svfiprintf_r+0xb3a>
   1b2fa: 93 f7 f7 0f  	andi	a5, a5, 255
   1b2fe: b1 b2        	j	0x1ac4a <_svfiprintf_r+0x446>
   1b300: 93 f6 07 20  	andi	a3, a5, 512
   1b304: 95 ca        	beqz	a3, 0x1b338 <_svfiprintf_r+0xb34>
   1b306: 13 77 f7 0f  	andi	a4, a4, 255
   1b30a: cd b8        	j	0x1abfc <_svfiprintf_r+0x3f8>
   1b30c: 13 f7 0c 20  	andi	a4, s9, 512
   1b310: 19 cf        	beqz	a4, 0x1b32e <_svfiprintf_r+0xb2a>
   1b312: 9b 97 87 01  	slliw	a5, a5, 24
   1b316: 9b d7 87 41  	sraiw	a5, a5, 24
   1b31a: 3e 87        	mv	a4, a5
   1b31c: 63 5a 07 ee  	bgez	a4, 0x1aa10 <_svfiprintf_r+0x20c>
   1b320: 6f f0 4f fc  	j	0x1aae4 <_svfiprintf_r+0x2e0>
   1b324: 23 10 a7 01  	sh	s10, 0(a4)
   1b328: a1 06        	addi	a3, a3, 8
   1b32a: 6f f0 af d3  	j	0x1a864 <_svfiprintf_r+0x60>
   1b32e: 3e 87        	mv	a4, a5
   1b330: 63 50 07 ee  	bgez	a4, 0x1aa10 <_svfiprintf_r+0x20c>
   1b334: 6f f0 0f fb  	j	0x1aae4 <_svfiprintf_r+0x2e0>
   1b338: 02 17        	slli	a4, a4, 32
   1b33a: 01 93        	srli	a4, a4, 32
   1b33c: c1 b0        	j	0x1abfc <_svfiprintf_r+0x3f8>
   1b33e: 82 17        	slli	a5, a5, 32
   1b340: 81 93        	srli	a5, a5, 32
   1b342: 21 b2        	j	0x1ac4a <_svfiprintf_r+0x446>
   1b344: 02 17        	slli	a4, a4, 32
   1b346: 01 93        	srli	a4, a4, 32
   1b348: 71 b5        	j	0x1b1d4 <_svfiprintf_r+0x9d0>
   1b34a: 62 8e        	mv	t3, s8
   1b34c: 62 83        	mv	t1, s8
   1b34e: 01 4c        	mv	s8, zero
   1b350: 6f f0 0f fd  	j	0x1ab20 <_svfiprintf_r+0x31c>
   1b354: f6 86        	mv	a3, t4
   1b356: 93 07 00 03  	addi	a5, zero, 48
   1b35a: 6f f0 2f fb  	j	0x1ab0c <_svfiprintf_r+0x308>
   1b35e: 23 20 a7 01  	sw	s10, 0(a4)
   1b362: a1 06        	addi	a3, a3, 8
   1b364: 6f f0 0f d0  	j	0x1a864 <_svfiprintf_r+0x60>
   1b368: b1 47        	addi	a5, zero, 12
   1b36a: 23 20 fb 00  	sw	a5, 0(s6)
   1b36e: fd 5c        	addi	s9, zero, -1
   1b370: 6f f0 af f0  	j	0x1aa7a <_svfiprintf_r+0x276>
   1b374: 9c 42        	lw	a5, 0(a3)
   1b376: a1 06        	addi	a3, a3, 8
   1b378: 1b 8c 07 00  	sext.w	s8, a5
   1b37c: 63 d3 07 00  	bgez	a5, 0x1b382 <_svfiprintf_r+0xb7e>
   1b380: 7d 5c        	addi	s8, zero, -1
   1b382: 83 c7 1d 00  	lbu	a5, 1(s11)
   1b386: ba 8d        	mv	s11, a4
   1b388: 6f f0 cf d3  	j	0x1a8c4 <_svfiprintf_r+0xc0>

000000000001b38c <__swbuf_r>:
   1b38c: 79 71        	addi	sp, sp, -48
   1b38e: 22 f0        	sd	s0, 32(sp)
   1b390: 26 ec        	sd	s1, 24(sp)
   1b392: 4a e8        	sd	s2, 16(sp)
   1b394: 06 f4        	sd	ra, 40(sp)
   1b396: 4e e4        	sd	s3, 8(sp)
   1b398: 2a 89        	mv	s2, a0
   1b39a: ae 84        	mv	s1, a1
   1b39c: 32 84        	mv	s0, a2
   1b39e: 19 c1        	beqz	a0, 0x1b3a4 <__swbuf_r+0x18>
   1b3a0: 3c 49        	lw	a5, 80(a0)
   1b3a2: f9 cb        	beqz	a5, 0x1b478 <__swbuf_r+0xec>
   1b3a4: 1c 54        	lw	a5, 40(s0)
   1b3a6: 03 17 04 01  	lh	a4, 16(s0)
   1b3aa: 5c c4        	sw	a5, 12(s0)
   1b3ac: 93 77 87 00  	andi	a5, a4, 8
   1b3b0: b9 cb        	beqz	a5, 0x1b406 <__swbuf_r+0x7a>
   1b3b2: 1c 6c        	ld	a5, 24(s0)
   1b3b4: a9 cb        	beqz	a5, 0x1b406 <__swbuf_r+0x7a>
   1b3b6: 93 f9 f4 0f  	andi	s3, s1, 255
   1b3ba: 93 16 27 03  	slli	a3, a4, 50
   1b3be: ce 84        	mv	s1, s3
   1b3c0: 63 d2 06 06  	bgez	a3, 0x1b424 <__swbuf_r+0x98>
   1b3c4: 18 60        	ld	a4, 0(s0)
   1b3c6: 14 50        	lw	a3, 32(s0)
   1b3c8: bb 07 f7 40  	subw	a5, a4, a5
   1b3cc: 63 dd d7 06  	bge	a5, a3, 0x1b446 <__swbuf_r+0xba>
   1b3d0: 54 44        	lw	a3, 12(s0)
   1b3d2: 13 06 17 00  	addi	a2, a4, 1
   1b3d6: 10 e0        	sd	a2, 0(s0)
   1b3d8: fd 36        	addiw	a3, a3, -1
   1b3da: 54 c4        	sw	a3, 12(s0)
   1b3dc: 23 00 37 01  	sb	s3, 0(a4)
   1b3e0: 18 50        	lw	a4, 32(s0)
   1b3e2: 85 27        	addiw	a5, a5, 1
   1b3e4: 63 03 f7 08  	beq	a4, a5, 0x1b46a <__swbuf_r+0xde>
   1b3e8: 83 57 04 01  	lhu	a5, 16(s0)
   1b3ec: 85 8b        	andi	a5, a5, 1
   1b3ee: 81 c7        	beqz	a5, 0x1b3f6 <__swbuf_r+0x6a>
   1b3f0: a9 47        	addi	a5, zero, 10
   1b3f2: 63 8c f4 06  	beq	s1, a5, 0x1b46a <__swbuf_r+0xde>
   1b3f6: a2 70        	ld	ra, 40(sp)
   1b3f8: 02 74        	ld	s0, 32(sp)
   1b3fa: 42 69        	ld	s2, 16(sp)
   1b3fc: a2 69        	ld	s3, 8(sp)
   1b3fe: 26 85        	mv	a0, s1
   1b400: e2 64        	ld	s1, 24(sp)
   1b402: 45 61        	addi	sp, sp, 48
   1b404: 82 80        	ret
   1b406: a2 85        	mv	a1, s0
   1b408: 4a 85        	mv	a0, s2
   1b40a: ef 60 5f f6  	jal	0x1236e <__swsetup_r>
   1b40e: 3d e1        	bnez	a0, 0x1b474 <__swbuf_r+0xe8>
   1b410: 03 17 04 01  	lh	a4, 16(s0)
   1b414: 93 f9 f4 0f  	andi	s3, s1, 255
   1b418: 1c 6c        	ld	a5, 24(s0)
   1b41a: 93 16 27 03  	slli	a3, a4, 50
   1b41e: ce 84        	mv	s1, s3
   1b420: e3 c2 06 fa  	bltz	a3, 0x1b3c4 <__swbuf_r+0x38>
   1b424: 83 26 c4 0a  	lw	a3, 172(s0)
   1b428: 09 66        	lui	a2, 2
   1b42a: 51 8f        	or	a4, a4, a2
   1b42c: 79 76        	lui	a2, 1048574
   1b42e: 7d 16        	addi	a2, a2, -1
   1b430: f1 8e        	and	a3, a3, a2
   1b432: 23 18 e4 00  	sh	a4, 16(s0)
   1b436: 18 60        	ld	a4, 0(s0)
   1b438: 23 26 d4 0a  	sw	a3, 172(s0)
   1b43c: 14 50        	lw	a3, 32(s0)
   1b43e: bb 07 f7 40  	subw	a5, a4, a5
   1b442: e3 c7 d7 f8  	blt	a5, a3, 0x1b3d0 <__swbuf_r+0x44>
   1b446: a2 85        	mv	a1, s0
   1b448: 4a 85        	mv	a0, s2
   1b44a: ef 70 4f a4  	jal	0x1268e <_fflush_r>
   1b44e: 1d e1        	bnez	a0, 0x1b474 <__swbuf_r+0xe8>
   1b450: 18 60        	ld	a4, 0(s0)
   1b452: 54 44        	lw	a3, 12(s0)
   1b454: 85 47        	addi	a5, zero, 1
   1b456: 13 06 17 00  	addi	a2, a4, 1
   1b45a: fd 36        	addiw	a3, a3, -1
   1b45c: 10 e0        	sd	a2, 0(s0)
   1b45e: 54 c4        	sw	a3, 12(s0)
   1b460: 23 00 37 01  	sb	s3, 0(a4)
   1b464: 18 50        	lw	a4, 32(s0)
   1b466: e3 11 f7 f8  	bne	a4, a5, 0x1b3e8 <__swbuf_r+0x5c>
   1b46a: a2 85        	mv	a1, s0
   1b46c: 4a 85        	mv	a0, s2
   1b46e: ef 70 0f a2  	jal	0x1268e <_fflush_r>
   1b472: 51 d1        	beqz	a0, 0x1b3f6 <__swbuf_r+0x6a>
   1b474: fd 54        	addi	s1, zero, -1
   1b476: 41 b7        	j	0x1b3f6 <__swbuf_r+0x6a>
   1b478: ef 70 8f ce  	jal	0x12960 <__sinit>
   1b47c: 25 b7        	j	0x1b3a4 <__swbuf_r+0x18>

000000000001b47e <__swbuf>:
   1b47e: aa 87        	mv	a5, a0
   1b480: 03 b5 81 85  	ld	a0, -1960(gp)
   1b484: 2e 86        	mv	a2, a1
   1b486: be 85        	mv	a1, a5
   1b488: 6f f0 5f f0  	j	0x1b38c <__swbuf_r>

000000000001b48c <_wcrtomb_r>:
   1b48c: 79 71        	addi	sp, sp, -48
   1b48e: 22 f0        	sd	s0, 32(sp)
   1b490: 26 ec        	sd	s1, 24(sp)
   1b492: b7 07 02 00  	lui	a5, 32
   1b496: 06 f4        	sd	ra, 40(sp)
   1b498: 83 b7 87 e7  	ld	a5, -392(a5)
   1b49c: 2a 84        	mv	s0, a0
   1b49e: b6 84        	mv	s1, a3
   1b4a0: 91 c9        	beqz	a1, 0x1b4b4 <_wcrtomb_r+0x28>
   1b4a2: 82 97        	jalr	a5
   1b4a4: fd 57        	addi	a5, zero, -1
   1b4a6: 63 0d f5 00  	beq	a0, a5, 0x1b4c0 <_wcrtomb_r+0x34>
   1b4aa: a2 70        	ld	ra, 40(sp)
   1b4ac: 02 74        	ld	s0, 32(sp)
   1b4ae: e2 64        	ld	s1, 24(sp)
   1b4b0: 45 61        	addi	sp, sp, 48
   1b4b2: 82 80        	ret
   1b4b4: 01 46        	mv	a2, zero
   1b4b6: 8a 85        	mv	a1, sp
   1b4b8: 82 97        	jalr	a5
   1b4ba: fd 57        	addi	a5, zero, -1
   1b4bc: e3 17 f5 fe  	bne	a0, a5, 0x1b4aa <_wcrtomb_r+0x1e>
   1b4c0: 23 a0 04 00  	sw	zero, 0(s1)
   1b4c4: 93 07 a0 08  	addi	a5, zero, 138
   1b4c8: a2 70        	ld	ra, 40(sp)
   1b4ca: 1c c0        	sw	a5, 0(s0)
   1b4cc: 02 74        	ld	s0, 32(sp)
   1b4ce: e2 64        	ld	s1, 24(sp)
   1b4d0: 45 61        	addi	sp, sp, 48
   1b4d2: 82 80        	ret

000000000001b4d4 <wcrtomb>:
   1b4d4: 79 71        	addi	sp, sp, -48
   1b4d6: 22 f0        	sd	s0, 32(sp)
   1b4d8: 26 ec        	sd	s1, 24(sp)
   1b4da: 06 f4        	sd	ra, 40(sp)
   1b4dc: 83 b4 81 85  	ld	s1, -1960(gp)
   1b4e0: b7 07 02 00  	lui	a5, 32
   1b4e4: 83 b7 87 e7  	ld	a5, -392(a5)
   1b4e8: 32 84        	mv	s0, a2
   1b4ea: 11 cd        	beqz	a0, 0x1b506 <wcrtomb+0x32>
   1b4ec: 2e 86        	mv	a2, a1
   1b4ee: a2 86        	mv	a3, s0
   1b4f0: aa 85        	mv	a1, a0
   1b4f2: 26 85        	mv	a0, s1
   1b4f4: 82 97        	jalr	a5
   1b4f6: fd 57        	addi	a5, zero, -1
   1b4f8: 63 0f f5 00  	beq	a0, a5, 0x1b516 <wcrtomb+0x42>
   1b4fc: a2 70        	ld	ra, 40(sp)
   1b4fe: 02 74        	ld	s0, 32(sp)
   1b500: e2 64        	ld	s1, 24(sp)
   1b502: 45 61        	addi	sp, sp, 48
   1b504: 82 80        	ret
   1b506: b2 86        	mv	a3, a2
   1b508: 8a 85        	mv	a1, sp
   1b50a: 01 46        	mv	a2, zero
   1b50c: 26 85        	mv	a0, s1
   1b50e: 82 97        	jalr	a5
   1b510: fd 57        	addi	a5, zero, -1
   1b512: e3 15 f5 fe  	bne	a0, a5, 0x1b4fc <wcrtomb+0x28>
   1b516: 23 20 04 00  	sw	zero, 0(s0)
   1b51a: a2 70        	ld	ra, 40(sp)
   1b51c: 02 74        	ld	s0, 32(sp)
   1b51e: 93 07 a0 08  	addi	a5, zero, 138
   1b522: 9c c0        	sw	a5, 0(s1)
   1b524: e2 64        	ld	s1, 24(sp)
   1b526: 45 61        	addi	sp, sp, 48
   1b528: 82 80        	ret

000000000001b52a <abort>:
   1b52a: 41 11        	addi	sp, sp, -16
   1b52c: 19 45        	addi	a0, zero, 6
   1b52e: 06 e4        	sd	ra, 8(sp)
   1b530: ef 00 c0 23  	jal	0x1b76c <raise>
   1b534: 05 45        	addi	a0, zero, 1
   1b536: ef 00 40 46  	jal	0x1b99a <_exit>

000000000001b53a <__register_exitproc>:
   1b53a: 03 b7 81 83  	ld	a4, -1992(gp)
   1b53e: 83 37 87 1f  	ld	a5, 504(a4)
   1b542: b1 c3        	beqz	a5, 0x1b586 <__register_exitproc+0x4c>
   1b544: 98 47        	lw	a4, 8(a5)
   1b546: 7d 48        	addi	a6, zero, 31
   1b548: 63 42 e8 06  	blt	a6, a4, 0x1b5ac <__register_exitproc+0x72>
   1b54c: 05 c5        	beqz	a0, 0x1b574 <__register_exitproc+0x3a>
   1b54e: 13 18 37 00  	slli	a6, a4, 3
   1b552: 3e 98        	add	a6, a6, a5
   1b554: 23 38 c8 10  	sd	a2, 272(a6)
   1b558: 83 a8 07 31  	lw	a7, 784(a5)
   1b55c: 05 46        	addi	a2, zero, 1
   1b55e: 3b 16 e6 00  	sllw	a2, a2, a4
   1b562: b3 e8 c8 00  	or	a7, a7, a2
   1b566: 23 a8 17 31  	sw	a7, 784(a5)
   1b56a: 23 38 d8 20  	sd	a3, 528(a6)
   1b56e: 89 46        	addi	a3, zero, 2
   1b570: 63 00 d5 02  	beq	a0, a3, 0x1b590 <__register_exitproc+0x56>
   1b574: 93 06 27 00  	addi	a3, a4, 2
   1b578: 8e 06        	slli	a3, a3, 3
   1b57a: 05 27        	addiw	a4, a4, 1
   1b57c: 98 c7        	sw	a4, 8(a5)
   1b57e: b6 97        	add	a5, a5, a3
   1b580: 8c e3        	sd	a1, 0(a5)
   1b582: 01 45        	mv	a0, zero
   1b584: 82 80        	ret
   1b586: 93 07 07 20  	addi	a5, a4, 512
   1b58a: 23 3c f7 1e  	sd	a5, 504(a4)
   1b58e: 5d bf        	j	0x1b544 <__register_exitproc+0xa>
   1b590: 83 a6 47 31  	lw	a3, 788(a5)
   1b594: 01 45        	mv	a0, zero
   1b596: d1 8e        	or	a3, a3, a2
   1b598: 23 aa d7 30  	sw	a3, 788(a5)
   1b59c: 93 06 27 00  	addi	a3, a4, 2
   1b5a0: 8e 06        	slli	a3, a3, 3
   1b5a2: 05 27        	addiw	a4, a4, 1
   1b5a4: 98 c7        	sw	a4, 8(a5)
   1b5a6: b6 97        	add	a5, a5, a3
   1b5a8: 8c e3        	sd	a1, 0(a5)
   1b5aa: 82 80        	ret
   1b5ac: 7d 55        	addi	a0, zero, -1
   1b5ae: 82 80        	ret

000000000001b5b0 <_init_signal_r>:
   1b5b0: 83 37 85 51  	ld	a5, 1304(a0)
   1b5b4: 99 c3        	beqz	a5, 0x1b5ba <_init_signal_r+0xa>
   1b5b6: 01 45        	mv	a0, zero
   1b5b8: 82 80        	ret
   1b5ba: 41 11        	addi	sp, sp, -16
   1b5bc: 93 05 00 10  	addi	a1, zero, 256
   1b5c0: 22 e0        	sd	s0, 0(sp)
   1b5c2: 06 e4        	sd	ra, 8(sp)
   1b5c4: 2a 84        	mv	s0, a0
   1b5c6: ef a0 4f 95  	jal	0x1571a <_malloc_r>
   1b5ca: 23 3c a4 50  	sd	a0, 1304(s0)
   1b5ce: 0d c5        	beqz	a0, 0x1b5f8 <_init_signal_r+0x48>
   1b5d0: 93 07 00 02  	addi	a5, zero, 32
   1b5d4: f3 26 20 c2  	csrr	a3, vlenb
   1b5d8: 57 77 80 05  	<unknown>
   1b5dc: 57 3c 00 5e  	<unknown>
   1b5e0: 57 f7 87 05  	<unknown>
   1b5e4: 27 7c 05 02  	<unknown>
   1b5e8: 99 8f        	sub	a5, a5, a4
   1b5ea: 36 95        	add	a0, a0, a3
   1b5ec: f5 fb        	bnez	a5, 0x1b5e0 <_init_signal_r+0x30>
   1b5ee: 01 45        	mv	a0, zero
   1b5f0: a2 60        	ld	ra, 8(sp)
   1b5f2: 02 64        	ld	s0, 0(sp)
   1b5f4: 41 01        	addi	sp, sp, 16
   1b5f6: 82 80        	ret
   1b5f8: 7d 55        	addi	a0, zero, -1
   1b5fa: dd bf        	j	0x1b5f0 <_init_signal_r+0x40>

000000000001b5fc <_signal_r>:
   1b5fc: 79 71        	addi	sp, sp, -48
   1b5fe: 26 ec        	sd	s1, 24(sp)
   1b600: 06 f4        	sd	ra, 40(sp)
   1b602: 22 f0        	sd	s0, 32(sp)
   1b604: fd 47        	addi	a5, zero, 31
   1b606: aa 84        	mv	s1, a0
   1b608: 63 ef b7 00  	bltu	a5, a1, 0x1b626 <_signal_r+0x2a>
   1b60c: 83 37 85 51  	ld	a5, 1304(a0)
   1b610: 2e 84        	mv	s0, a1
   1b612: 91 cf        	beqz	a5, 0x1b62e <_signal_r+0x32>
   1b614: 0e 04        	slli	s0, s0, 3
   1b616: a2 97        	add	a5, a5, s0
   1b618: 88 63        	ld	a0, 0(a5)
   1b61a: 90 e3        	sd	a2, 0(a5)
   1b61c: a2 70        	ld	ra, 40(sp)
   1b61e: 02 74        	ld	s0, 32(sp)
   1b620: e2 64        	ld	s1, 24(sp)
   1b622: 45 61        	addi	sp, sp, 48
   1b624: 82 80        	ret
   1b626: d9 47        	addi	a5, zero, 22
   1b628: 1c c1        	sw	a5, 0(a0)
   1b62a: 7d 55        	addi	a0, zero, -1
   1b62c: c5 bf        	j	0x1b61c <_signal_r+0x20>
   1b62e: 93 05 00 10  	addi	a1, zero, 256
   1b632: 32 e4        	sd	a2, 8(sp)
   1b634: ef a0 6f 8e  	jal	0x1571a <_malloc_r>
   1b638: 57 77 80 05  	<unknown>
   1b63c: aa 87        	mv	a5, a0
   1b63e: 23 bc a4 50  	sd	a0, 1304(s1)
   1b642: aa 86        	mv	a3, a0
   1b644: 22 66        	ld	a2, 8(sp)
   1b646: 13 07 00 02  	addi	a4, zero, 32
   1b64a: 73 25 20 c2  	csrr	a0, vlenb
   1b64e: 57 3c 00 5e  	<unknown>
   1b652: e1 df        	beqz	a5, 0x1b62a <_signal_r+0x2e>
   1b654: d7 75 87 05  	<unknown>
   1b658: 27 fc 06 02  	<unknown>
   1b65c: 0d 8f        	sub	a4, a4, a1
   1b65e: aa 96        	add	a3, a3, a0
   1b660: 75 fb        	bnez	a4, 0x1b654 <_signal_r+0x58>
   1b662: 4d bf        	j	0x1b614 <_signal_r+0x18>

000000000001b664 <_raise_r>:
   1b664: 01 11        	addi	sp, sp, -32
   1b666: 26 e4        	sd	s1, 8(sp)
   1b668: 06 ec        	sd	ra, 24(sp)
   1b66a: 22 e8        	sd	s0, 16(sp)
   1b66c: fd 47        	addi	a5, zero, 31
   1b66e: aa 84        	mv	s1, a0
   1b670: 63 ee b7 04  	bltu	a5, a1, 0x1b6cc <_raise_r+0x68>
   1b674: 83 37 85 51  	ld	a5, 1304(a0)
   1b678: 2e 84        	mv	s0, a1
   1b67a: 95 c7        	beqz	a5, 0x1b6a6 <_raise_r+0x42>
   1b67c: 13 97 35 00  	slli	a4, a1, 3
   1b680: ba 97        	add	a5, a5, a4
   1b682: 98 63        	ld	a4, 0(a5)
   1b684: 0d c3        	beqz	a4, 0x1b6a6 <_raise_r+0x42>
   1b686: 85 46        	addi	a3, zero, 1
   1b688: 63 09 d7 00  	beq	a4, a3, 0x1b69a <_raise_r+0x36>
   1b68c: fd 56        	addi	a3, zero, -1
   1b68e: 63 07 d7 02  	beq	a4, a3, 0x1b6bc <_raise_r+0x58>
   1b692: 23 b0 07 00  	sd	zero, 0(a5)
   1b696: 2e 85        	mv	a0, a1
   1b698: 02 97        	jalr	a4
   1b69a: 01 45        	mv	a0, zero
   1b69c: e2 60        	ld	ra, 24(sp)
   1b69e: 42 64        	ld	s0, 16(sp)
   1b6a0: a2 64        	ld	s1, 8(sp)
   1b6a2: 05 61        	addi	sp, sp, 32
   1b6a4: 82 80        	ret
   1b6a6: 26 85        	mv	a0, s1
   1b6a8: ef 00 40 2c  	jal	0x1b96c <_getpid_r>
   1b6ac: 22 86        	mv	a2, s0
   1b6ae: 42 64        	ld	s0, 16(sp)
   1b6b0: e2 60        	ld	ra, 24(sp)
   1b6b2: aa 85        	mv	a1, a0
   1b6b4: 26 85        	mv	a0, s1
   1b6b6: a2 64        	ld	s1, 8(sp)
   1b6b8: 05 61        	addi	sp, sp, 32
   1b6ba: a5 ac        	j	0x1b932 <_kill_r>
   1b6bc: e2 60        	ld	ra, 24(sp)
   1b6be: 42 64        	ld	s0, 16(sp)
   1b6c0: d9 47        	addi	a5, zero, 22
   1b6c2: 1c c1        	sw	a5, 0(a0)
   1b6c4: a2 64        	ld	s1, 8(sp)
   1b6c6: 05 45        	addi	a0, zero, 1
   1b6c8: 05 61        	addi	sp, sp, 32
   1b6ca: 82 80        	ret
   1b6cc: d9 47        	addi	a5, zero, 22
   1b6ce: 1c c1        	sw	a5, 0(a0)
   1b6d0: 7d 55        	addi	a0, zero, -1
   1b6d2: e9 b7        	j	0x1b69c <_raise_r+0x38>

000000000001b6d4 <__sigtramp_r>:
   1b6d4: fd 47        	addi	a5, zero, 31
   1b6d6: 63 e9 b7 08  	bltu	a5, a1, 0x1b768 <__sigtramp_r+0x94>
   1b6da: 83 37 85 51  	ld	a5, 1304(a0)
   1b6de: 01 11        	addi	sp, sp, -32
   1b6e0: 22 e8        	sd	s0, 16(sp)
   1b6e2: 26 e4        	sd	s1, 8(sp)
   1b6e4: 06 ec        	sd	ra, 24(sp)
   1b6e6: 2e 84        	mv	s0, a1
   1b6e8: aa 84        	mv	s1, a0
   1b6ea: a1 c7        	beqz	a5, 0x1b732 <__sigtramp_r+0x5e>
   1b6ec: 13 17 34 00  	slli	a4, s0, 3
   1b6f0: ba 97        	add	a5, a5, a4
   1b6f2: 98 63        	ld	a4, 0(a5)
   1b6f4: 0d c3        	beqz	a4, 0x1b716 <__sigtramp_r+0x42>
   1b6f6: fd 56        	addi	a3, zero, -1
   1b6f8: 63 0b d7 02  	beq	a4, a3, 0x1b72e <__sigtramp_r+0x5a>
   1b6fc: 85 46        	addi	a3, zero, 1
   1b6fe: 63 02 d7 02  	beq	a4, a3, 0x1b722 <__sigtramp_r+0x4e>
   1b702: 22 85        	mv	a0, s0
   1b704: 23 b0 07 00  	sd	zero, 0(a5)
   1b708: 02 97        	jalr	a4
   1b70a: 01 45        	mv	a0, zero
   1b70c: e2 60        	ld	ra, 24(sp)
   1b70e: 42 64        	ld	s0, 16(sp)
   1b710: a2 64        	ld	s1, 8(sp)
   1b712: 05 61        	addi	sp, sp, 32
   1b714: 82 80        	ret
   1b716: e2 60        	ld	ra, 24(sp)
   1b718: 42 64        	ld	s0, 16(sp)
   1b71a: a2 64        	ld	s1, 8(sp)
   1b71c: 05 45        	addi	a0, zero, 1
   1b71e: 05 61        	addi	sp, sp, 32
   1b720: 82 80        	ret
   1b722: e2 60        	ld	ra, 24(sp)
   1b724: 42 64        	ld	s0, 16(sp)
   1b726: a2 64        	ld	s1, 8(sp)
   1b728: 0d 45        	addi	a0, zero, 3
   1b72a: 05 61        	addi	sp, sp, 32
   1b72c: 82 80        	ret
   1b72e: 09 45        	addi	a0, zero, 2
   1b730: f1 bf        	j	0x1b70c <__sigtramp_r+0x38>
   1b732: 93 05 00 10  	addi	a1, zero, 256
   1b736: ef 90 5f fe  	jal	0x1571a <_malloc_r>
   1b73a: 23 bc a4 50  	sd	a0, 1304(s1)
   1b73e: aa 87        	mv	a5, a0
   1b740: 15 c1        	beqz	a0, 0x1b764 <__sigtramp_r+0x90>
   1b742: aa 86        	mv	a3, a0
   1b744: 13 07 00 02  	addi	a4, zero, 32
   1b748: f3 25 20 c2  	csrr	a1, vlenb
   1b74c: 57 76 80 05  	<unknown>
   1b750: 57 3c 00 5e  	<unknown>
   1b754: 57 76 87 05  	<unknown>
   1b758: 27 fc 06 02  	<unknown>
   1b75c: 11 8f        	sub	a4, a4, a2
   1b75e: ae 96        	add	a3, a3, a1
   1b760: 75 fb        	bnez	a4, 0x1b754 <__sigtramp_r+0x80>
   1b762: 69 b7        	j	0x1b6ec <__sigtramp_r+0x18>
   1b764: 7d 55        	addi	a0, zero, -1
   1b766: 5d b7        	j	0x1b70c <__sigtramp_r+0x38>
   1b768: 7d 55        	addi	a0, zero, -1
   1b76a: 82 80        	ret

000000000001b76c <raise>:
   1b76c: 01 11        	addi	sp, sp, -32
   1b76e: 26 e4        	sd	s1, 8(sp)
   1b770: 06 ec        	sd	ra, 24(sp)
   1b772: 22 e8        	sd	s0, 16(sp)
   1b774: fd 47        	addi	a5, zero, 31
   1b776: 83 b4 81 85  	ld	s1, -1960(gp)
   1b77a: 63 ed a7 04  	bltu	a5, a0, 0x1b7d4 <raise+0x68>
   1b77e: 83 b7 84 51  	ld	a5, 1304(s1)
   1b782: 2a 84        	mv	s0, a0
   1b784: 8d c7        	beqz	a5, 0x1b7ae <raise+0x42>
   1b786: 13 17 35 00  	slli	a4, a0, 3
   1b78a: ba 97        	add	a5, a5, a4
   1b78c: 98 63        	ld	a4, 0(a5)
   1b78e: 05 c3        	beqz	a4, 0x1b7ae <raise+0x42>
   1b790: 85 46        	addi	a3, zero, 1
   1b792: 63 08 d7 00  	beq	a4, a3, 0x1b7a2 <raise+0x36>
   1b796: fd 56        	addi	a3, zero, -1
   1b798: 63 06 d7 02  	beq	a4, a3, 0x1b7c4 <raise+0x58>
   1b79c: 23 b0 07 00  	sd	zero, 0(a5)
   1b7a0: 02 97        	jalr	a4
   1b7a2: 01 45        	mv	a0, zero
   1b7a4: e2 60        	ld	ra, 24(sp)
   1b7a6: 42 64        	ld	s0, 16(sp)
   1b7a8: a2 64        	ld	s1, 8(sp)
   1b7aa: 05 61        	addi	sp, sp, 32
   1b7ac: 82 80        	ret
   1b7ae: 26 85        	mv	a0, s1
   1b7b0: ef 00 c0 1b  	jal	0x1b96c <_getpid_r>
   1b7b4: 22 86        	mv	a2, s0
   1b7b6: 42 64        	ld	s0, 16(sp)
   1b7b8: e2 60        	ld	ra, 24(sp)
   1b7ba: aa 85        	mv	a1, a0
   1b7bc: 26 85        	mv	a0, s1
   1b7be: a2 64        	ld	s1, 8(sp)
   1b7c0: 05 61        	addi	sp, sp, 32
   1b7c2: 85 aa        	j	0x1b932 <_kill_r>
   1b7c4: e2 60        	ld	ra, 24(sp)
   1b7c6: 42 64        	ld	s0, 16(sp)
   1b7c8: d9 47        	addi	a5, zero, 22
   1b7ca: 9c c0        	sw	a5, 0(s1)
   1b7cc: 05 45        	addi	a0, zero, 1
   1b7ce: a2 64        	ld	s1, 8(sp)
   1b7d0: 05 61        	addi	sp, sp, 32
   1b7d2: 82 80        	ret
   1b7d4: d9 47        	addi	a5, zero, 22
   1b7d6: 9c c0        	sw	a5, 0(s1)
   1b7d8: 7d 55        	addi	a0, zero, -1
   1b7da: e9 b7        	j	0x1b7a4 <raise+0x38>

000000000001b7dc <signal>:
   1b7dc: 01 11        	addi	sp, sp, -32
   1b7de: 4a e0        	sd	s2, 0(sp)
   1b7e0: 06 ec        	sd	ra, 24(sp)
   1b7e2: 22 e8        	sd	s0, 16(sp)
   1b7e4: 26 e4        	sd	s1, 8(sp)
   1b7e6: fd 47        	addi	a5, zero, 31
   1b7e8: 03 b9 81 85  	ld	s2, -1960(gp)
   1b7ec: 63 e2 a7 02  	bltu	a5, a0, 0x1b810 <signal+0x34>
   1b7f0: 2a 84        	mv	s0, a0
   1b7f2: 03 35 89 51  	ld	a0, 1304(s2)
   1b7f6: ae 84        	mv	s1, a1
   1b7f8: 0d c1        	beqz	a0, 0x1b81a <signal+0x3e>
   1b7fa: 0e 04        	slli	s0, s0, 3
   1b7fc: b3 07 85 00  	add	a5, a0, s0
   1b800: 88 63        	ld	a0, 0(a5)
   1b802: 84 e3        	sd	s1, 0(a5)
   1b804: e2 60        	ld	ra, 24(sp)
   1b806: 42 64        	ld	s0, 16(sp)
   1b808: a2 64        	ld	s1, 8(sp)
   1b80a: 02 69        	ld	s2, 0(sp)
   1b80c: 05 61        	addi	sp, sp, 32
   1b80e: 82 80        	ret
   1b810: d9 47        	addi	a5, zero, 22
   1b812: 23 20 f9 00  	sw	a5, 0(s2)
   1b816: 7d 55        	addi	a0, zero, -1
   1b818: f5 b7        	j	0x1b804 <signal+0x28>
   1b81a: 93 05 00 10  	addi	a1, zero, 256
   1b81e: 4a 85        	mv	a0, s2
   1b820: ef 90 bf ef  	jal	0x1571a <_malloc_r>
   1b824: d7 77 80 05  	<unknown>
   1b828: 23 3c a9 50  	sd	a0, 1304(s2)
   1b82c: 2a 87        	mv	a4, a0
   1b82e: 93 07 00 02  	addi	a5, zero, 32
   1b832: 73 26 20 c2  	csrr	a2, vlenb
   1b836: 57 3c 00 5e  	<unknown>
   1b83a: 71 dd        	beqz	a0, 0x1b816 <signal+0x3a>
   1b83c: d7 f6 87 05  	<unknown>
   1b840: 27 7c 07 02  	<unknown>
   1b844: 95 8f        	sub	a5, a5, a3
   1b846: 32 97        	add	a4, a4, a2
   1b848: f5 fb        	bnez	a5, 0x1b83c <signal+0x60>
   1b84a: 45 bf        	j	0x1b7fa <signal+0x1e>

000000000001b84c <_init_signal>:
   1b84c: 41 11        	addi	sp, sp, -16
   1b84e: 22 e0        	sd	s0, 0(sp)
   1b850: 03 b4 81 85  	ld	s0, -1960(gp)
   1b854: 06 e4        	sd	ra, 8(sp)
   1b856: 83 37 84 51  	ld	a5, 1304(s0)
   1b85a: 91 c7        	beqz	a5, 0x1b866 <_init_signal+0x1a>
   1b85c: 01 45        	mv	a0, zero
   1b85e: a2 60        	ld	ra, 8(sp)
   1b860: 02 64        	ld	s0, 0(sp)
   1b862: 41 01        	addi	sp, sp, 16
   1b864: 82 80        	ret
   1b866: 93 05 00 10  	addi	a1, zero, 256
   1b86a: 22 85        	mv	a0, s0
   1b86c: ef 90 ff ea  	jal	0x1571a <_malloc_r>
   1b870: 23 3c a4 50  	sd	a0, 1304(s0)
   1b874: 0d c1        	beqz	a0, 0x1b896 <_init_signal+0x4a>
   1b876: 93 07 00 02  	addi	a5, zero, 32
   1b87a: f3 26 20 c2  	csrr	a3, vlenb
   1b87e: 57 77 80 05  	<unknown>
   1b882: 57 3c 00 5e  	<unknown>
   1b886: 57 f7 87 05  	<unknown>
   1b88a: 27 7c 05 02  	<unknown>
   1b88e: 99 8f        	sub	a5, a5, a4
   1b890: 36 95        	add	a0, a0, a3
   1b892: f5 fb        	bnez	a5, 0x1b886 <_init_signal+0x3a>
   1b894: e1 b7        	j	0x1b85c <_init_signal+0x10>
   1b896: 7d 55        	addi	a0, zero, -1
   1b898: d9 b7        	j	0x1b85e <_init_signal+0x12>

000000000001b89a <__sigtramp>:
   1b89a: 01 11        	addi	sp, sp, -32
   1b89c: 26 e4        	sd	s1, 8(sp)
   1b89e: 06 ec        	sd	ra, 24(sp)
   1b8a0: 22 e8        	sd	s0, 16(sp)
   1b8a2: fd 47        	addi	a5, zero, 31
   1b8a4: 83 b4 81 85  	ld	s1, -1960(gp)
   1b8a8: 63 e3 a7 08  	bltu	a5, a0, 0x1b92e <__sigtramp+0x94>
   1b8ac: 83 b7 84 51  	ld	a5, 1304(s1)
   1b8b0: 2a 84        	mv	s0, a0
   1b8b2: a1 c7        	beqz	a5, 0x1b8fa <__sigtramp+0x60>
   1b8b4: 13 17 34 00  	slli	a4, s0, 3
   1b8b8: ba 97        	add	a5, a5, a4
   1b8ba: 98 63        	ld	a4, 0(a5)
   1b8bc: 0d c3        	beqz	a4, 0x1b8de <__sigtramp+0x44>
   1b8be: fd 56        	addi	a3, zero, -1
   1b8c0: 63 0b d7 02  	beq	a4, a3, 0x1b8f6 <__sigtramp+0x5c>
   1b8c4: 85 46        	addi	a3, zero, 1
   1b8c6: 63 02 d7 02  	beq	a4, a3, 0x1b8ea <__sigtramp+0x50>
   1b8ca: 22 85        	mv	a0, s0
   1b8cc: 23 b0 07 00  	sd	zero, 0(a5)
   1b8d0: 02 97        	jalr	a4
   1b8d2: 01 45        	mv	a0, zero
   1b8d4: e2 60        	ld	ra, 24(sp)
   1b8d6: 42 64        	ld	s0, 16(sp)
   1b8d8: a2 64        	ld	s1, 8(sp)
   1b8da: 05 61        	addi	sp, sp, 32
   1b8dc: 82 80        	ret
   1b8de: e2 60        	ld	ra, 24(sp)
   1b8e0: 42 64        	ld	s0, 16(sp)
   1b8e2: a2 64        	ld	s1, 8(sp)
   1b8e4: 05 45        	addi	a0, zero, 1
   1b8e6: 05 61        	addi	sp, sp, 32
   1b8e8: 82 80        	ret
   1b8ea: e2 60        	ld	ra, 24(sp)
   1b8ec: 42 64        	ld	s0, 16(sp)
   1b8ee: a2 64        	ld	s1, 8(sp)
   1b8f0: 0d 45        	addi	a0, zero, 3
   1b8f2: 05 61        	addi	sp, sp, 32
   1b8f4: 82 80        	ret
   1b8f6: 09 45        	addi	a0, zero, 2
   1b8f8: f1 bf        	j	0x1b8d4 <__sigtramp+0x3a>
   1b8fa: 93 05 00 10  	addi	a1, zero, 256
   1b8fe: 26 85        	mv	a0, s1
   1b900: ef 90 bf e1  	jal	0x1571a <_malloc_r>
   1b904: 23 bc a4 50  	sd	a0, 1304(s1)
   1b908: aa 87        	mv	a5, a0
   1b90a: 15 c1        	beqz	a0, 0x1b92e <__sigtramp+0x94>
   1b90c: aa 86        	mv	a3, a0
   1b90e: 13 07 00 02  	addi	a4, zero, 32
   1b912: f3 25 20 c2  	csrr	a1, vlenb
   1b916: 57 76 80 05  	<unknown>
   1b91a: 57 3c 00 5e  	<unknown>
   1b91e: 57 76 87 05  	<unknown>
   1b922: 27 fc 06 02  	<unknown>
   1b926: 11 8f        	sub	a4, a4, a2
   1b928: ae 96        	add	a3, a3, a1
   1b92a: 75 fb        	bnez	a4, 0x1b91e <__sigtramp+0x84>
   1b92c: 61 b7        	j	0x1b8b4 <__sigtramp+0x1a>
   1b92e: 7d 55        	addi	a0, zero, -1
   1b930: 55 b7        	j	0x1b8d4 <__sigtramp+0x3a>

000000000001b932 <_kill_r>:
   1b932: 01 11        	addi	sp, sp, -32
   1b934: 2e 87        	mv	a4, a1
   1b936: 22 e8        	sd	s0, 16(sp)
   1b938: 26 e4        	sd	s1, 8(sp)
   1b93a: 2a 84        	mv	s0, a0
   1b93c: b2 85        	mv	a1, a2
   1b93e: 3a 85        	mv	a0, a4
   1b940: 06 ec        	sd	ra, 24(sp)
   1b942: 23 ac 01 88  	sw	zero, -1896(gp)
   1b946: ef 00 00 0e  	jal	0x1ba26 <_kill>
   1b94a: fd 57        	addi	a5, zero, -1
   1b94c: 63 07 f5 00  	beq	a0, a5, 0x1b95a <_kill_r+0x28>
   1b950: e2 60        	ld	ra, 24(sp)
   1b952: 42 64        	ld	s0, 16(sp)
   1b954: a2 64        	ld	s1, 8(sp)
   1b956: 05 61        	addi	sp, sp, 32
   1b958: 82 80        	ret
   1b95a: 83 a7 81 89  	lw	a5, -1896(gp)
   1b95e: ed db        	beqz	a5, 0x1b950 <_kill_r+0x1e>
   1b960: e2 60        	ld	ra, 24(sp)
   1b962: 1c c0        	sw	a5, 0(s0)
   1b964: 42 64        	ld	s0, 16(sp)
   1b966: a2 64        	ld	s1, 8(sp)
   1b968: 05 61        	addi	sp, sp, 32
   1b96a: 82 80        	ret

000000000001b96c <_getpid_r>:
   1b96c: 41 a8        	j	0x1b9fc <_getpid>

000000000001b96e <_close>:
   1b96e: 41 11        	addi	sp, sp, -16
   1b970: 06 e4        	sd	ra, 8(sp)
   1b972: 22 e0        	sd	s0, 0(sp)
   1b974: 93 08 90 03  	addi	a7, zero, 57
   1b978: 73 00 00 00  	ecall	
   1b97c: 2a 84        	mv	s0, a0
   1b97e: 63 47 05 00  	bltz	a0, 0x1b98c <_close+0x1e>
   1b982: 01 25        	sext.w	a0, a0
   1b984: a2 60        	ld	ra, 8(sp)
   1b986: 02 64        	ld	s0, 0(sp)
   1b988: 41 01        	addi	sp, sp, 16
   1b98a: 82 80        	ret
   1b98c: ef e0 0f 91  	jal	0x19a9c <__errno>
   1b990: 3b 04 80 40  	negw	s0, s0
   1b994: 00 c1        	sw	s0, 0(a0)
   1b996: 7d 55        	addi	a0, zero, -1
   1b998: f5 b7        	j	0x1b984 <_close+0x16>

000000000001b99a <_exit>:
   1b99a: 93 08 d0 05  	addi	a7, zero, 93
   1b99e: 73 00 00 00  	ecall	
   1b9a2: 63 43 05 00  	bltz	a0, 0x1b9a8 <_exit+0xe>
   1b9a6: 01 a0        	j	0x1b9a6 <_exit+0xc>
   1b9a8: 41 11        	addi	sp, sp, -16
   1b9aa: 22 e0        	sd	s0, 0(sp)
   1b9ac: 2a 84        	mv	s0, a0
   1b9ae: 06 e4        	sd	ra, 8(sp)
   1b9b0: 3b 04 80 40  	negw	s0, s0
   1b9b4: ef e0 8f 8e  	jal	0x19a9c <__errno>
   1b9b8: 00 c1        	sw	s0, 0(a0)
   1b9ba: 01 a0        	j	0x1b9ba <_exit+0x20>

000000000001b9bc <_fstat>:
   1b9bc: 35 71        	addi	sp, sp, -160
   1b9be: 26 e5        	sd	s1, 136(sp)
   1b9c0: 06 ed        	sd	ra, 152(sp)
   1b9c2: ae 84        	mv	s1, a1
   1b9c4: 22 e9        	sd	s0, 144(sp)
   1b9c6: 93 08 00 05  	addi	a7, zero, 80
   1b9ca: 8a 85        	mv	a1, sp
   1b9cc: 73 00 00 00  	ecall	
   1b9d0: 2a 84        	mv	s0, a0
   1b9d2: 63 4e 05 00  	bltz	a0, 0x1b9ee <_fstat+0x32>
   1b9d6: 1b 04 05 00  	sext.w	s0, a0
   1b9da: 26 85        	mv	a0, s1
   1b9dc: 8a 85        	mv	a1, sp
   1b9de: ef 00 40 13  	jal	0x1bb12 <_conv_stat>
   1b9e2: ea 60        	ld	ra, 152(sp)
   1b9e4: 22 85        	mv	a0, s0
   1b9e6: 4a 64        	ld	s0, 144(sp)
   1b9e8: aa 64        	ld	s1, 136(sp)
   1b9ea: 0d 61        	addi	sp, sp, 160
   1b9ec: 82 80        	ret
   1b9ee: 3b 04 80 40  	negw	s0, s0
   1b9f2: ef e0 af 8a  	jal	0x19a9c <__errno>
   1b9f6: 00 c1        	sw	s0, 0(a0)
   1b9f8: 7d 54        	addi	s0, zero, -1
   1b9fa: c5 b7        	j	0x1b9da <_fstat+0x1e>

000000000001b9fc <_getpid>:
   1b9fc: 05 45        	addi	a0, zero, 1
   1b9fe: 82 80        	ret

000000000001ba00 <_isatty>:
   1ba00: 19 71        	addi	sp, sp, -128
   1ba02: 2c 00        	addi	a1, sp, 8
   1ba04: 86 fc        	sd	ra, 120(sp)
   1ba06: ef f0 7f fb  	jal	0x1b9bc <_fstat>
   1ba0a: fd 57        	addi	a5, zero, -1
   1ba0c: 63 09 f5 00  	beq	a0, a5, 0x1ba1e <_isatty+0x1e>
   1ba10: 32 45        	lw	a0, 12(sp)
   1ba12: e6 70        	ld	ra, 120(sp)
   1ba14: 1b 55 d5 40  	sraiw	a0, a0, 13
   1ba18: 05 89        	andi	a0, a0, 1
   1ba1a: 09 61        	addi	sp, sp, 128
   1ba1c: 82 80        	ret
   1ba1e: e6 70        	ld	ra, 120(sp)
   1ba20: 01 45        	mv	a0, zero
   1ba22: 09 61        	addi	sp, sp, 128
   1ba24: 82 80        	ret

000000000001ba26 <_kill>:
   1ba26: 41 11        	addi	sp, sp, -16
   1ba28: 06 e4        	sd	ra, 8(sp)
   1ba2a: ef e0 2f 87  	jal	0x19a9c <__errno>
   1ba2e: a2 60        	ld	ra, 8(sp)
   1ba30: d9 47        	addi	a5, zero, 22
   1ba32: 1c c1        	sw	a5, 0(a0)
   1ba34: 7d 55        	addi	a0, zero, -1
   1ba36: 41 01        	addi	sp, sp, 16
   1ba38: 82 80        	ret

000000000001ba3a <_lseek>:
   1ba3a: 41 11        	addi	sp, sp, -16
   1ba3c: 06 e4        	sd	ra, 8(sp)
   1ba3e: 22 e0        	sd	s0, 0(sp)
   1ba40: 93 08 e0 03  	addi	a7, zero, 62
   1ba44: 73 00 00 00  	ecall	
   1ba48: 2a 84        	mv	s0, a0
   1ba4a: 63 47 05 00  	bltz	a0, 0x1ba58 <_lseek+0x1e>
   1ba4e: a2 60        	ld	ra, 8(sp)
   1ba50: 22 85        	mv	a0, s0
   1ba52: 02 64        	ld	s0, 0(sp)
   1ba54: 41 01        	addi	sp, sp, 16
   1ba56: 82 80        	ret
   1ba58: 3b 04 80 40  	negw	s0, s0
   1ba5c: ef e0 0f 84  	jal	0x19a9c <__errno>
   1ba60: 00 c1        	sw	s0, 0(a0)
   1ba62: 7d 54        	addi	s0, zero, -1
   1ba64: ed b7        	j	0x1ba4e <_lseek+0x14>

000000000001ba66 <_read>:
   1ba66: 41 11        	addi	sp, sp, -16
   1ba68: 06 e4        	sd	ra, 8(sp)
   1ba6a: 22 e0        	sd	s0, 0(sp)
   1ba6c: 93 08 f0 03  	addi	a7, zero, 63
   1ba70: 73 00 00 00  	ecall	
   1ba74: 2a 84        	mv	s0, a0
   1ba76: 63 47 05 00  	bltz	a0, 0x1ba84 <_read+0x1e>
   1ba7a: a2 60        	ld	ra, 8(sp)
   1ba7c: 22 85        	mv	a0, s0
   1ba7e: 02 64        	ld	s0, 0(sp)
   1ba80: 41 01        	addi	sp, sp, 16
   1ba82: 82 80        	ret
   1ba84: 3b 04 80 40  	negw	s0, s0
   1ba88: ef e0 4f 81  	jal	0x19a9c <__errno>
   1ba8c: 00 c1        	sw	s0, 0(a0)
   1ba8e: 7d 54        	addi	s0, zero, -1
   1ba90: ed b7        	j	0x1ba7a <_read+0x14>

000000000001ba92 <_sbrk>:
   1ba92: 03 b7 01 8a  	ld	a4, -1888(gp)
   1ba96: 41 11        	addi	sp, sp, -16
   1ba98: 06 e4        	sd	ra, 8(sp)
   1ba9a: aa 87        	mv	a5, a0
   1ba9c: 01 ef        	bnez	a4, 0x1bab4 <_sbrk+0x22>
   1ba9e: 93 08 60 0d  	addi	a7, zero, 214
   1baa2: 01 45        	mv	a0, zero
   1baa4: 73 00 00 00  	ecall	
   1baa8: 7d 56        	addi	a2, zero, -1
   1baaa: 2a 87        	mv	a4, a0
   1baac: 63 05 c5 02  	beq	a0, a2, 0x1bad6 <_sbrk+0x44>
   1bab0: 23 b0 a1 8a  	sd	a0, -1888(gp)
   1bab4: 93 08 60 0d  	addi	a7, zero, 214
   1bab8: 33 85 e7 00  	add	a0, a5, a4
   1babc: 73 00 00 00  	ecall	
   1bac0: 03 b7 01 8a  	ld	a4, -1888(gp)
   1bac4: ba 97        	add	a5, a5, a4
   1bac6: 63 18 f5 00  	bne	a0, a5, 0x1bad6 <_sbrk+0x44>
   1baca: a2 60        	ld	ra, 8(sp)
   1bacc: 23 b0 a1 8a  	sd	a0, -1888(gp)
   1bad0: 3a 85        	mv	a0, a4
   1bad2: 41 01        	addi	sp, sp, 16
   1bad4: 82 80        	ret
   1bad6: ef d0 7f fc  	jal	0x19a9c <__errno>
   1bada: a2 60        	ld	ra, 8(sp)
   1badc: b1 47        	addi	a5, zero, 12
   1bade: 1c c1        	sw	a5, 0(a0)
   1bae0: 7d 55        	addi	a0, zero, -1
   1bae2: 41 01        	addi	sp, sp, 16
   1bae4: 82 80        	ret

000000000001bae6 <_write>:
   1bae6: 41 11        	addi	sp, sp, -16
   1bae8: 06 e4        	sd	ra, 8(sp)
   1baea: 22 e0        	sd	s0, 0(sp)
   1baec: 93 08 00 04  	addi	a7, zero, 64
   1baf0: 73 00 00 00  	ecall	
   1baf4: 2a 84        	mv	s0, a0
   1baf6: 63 47 05 00  	bltz	a0, 0x1bb04 <_write+0x1e>
   1bafa: a2 60        	ld	ra, 8(sp)
   1bafc: 22 85        	mv	a0, s0
   1bafe: 02 64        	ld	s0, 0(sp)
   1bb00: 41 01        	addi	sp, sp, 16
   1bb02: 82 80        	ret
   1bb04: 3b 04 80 40  	negw	s0, s0
   1bb08: ef d0 5f f9  	jal	0x19a9c <__errno>
   1bb0c: 00 c1        	sw	s0, 0(a0)
   1bb0e: 7d 54        	addi	s0, zero, -1
   1bb10: ed b7        	j	0x1bafa <_write+0x14>

000000000001bb12 <_conv_stat>:
   1bb12: 83 b3 05 00  	ld	t2, 0(a1)
   1bb16: 83 b2 85 00  	ld	t0, 8(a1)
   1bb1a: 83 af 05 01  	lw	t6, 16(a1)
   1bb1e: 03 af 45 01  	lw	t5, 20(a1)
   1bb22: 83 ae 85 01  	lw	t4, 24(a1)
   1bb26: 03 ae c5 01  	lw	t3, 28(a1)
   1bb2a: 03 b3 05 02  	ld	t1, 32(a1)
   1bb2e: 83 b8 05 03  	ld	a7, 48(a1)
   1bb32: 03 b8 05 04  	ld	a6, 64(a1)
   1bb36: 90 5d        	lw	a2, 56(a1)
   1bb38: b4 65        	ld	a3, 72(a1)
   1bb3a: b8 6d        	ld	a4, 88(a1)
   1bb3c: bc 75        	ld	a5, 104(a1)
   1bb3e: 23 10 75 00  	sh	t2, 0(a0)
   1bb42: 23 11 55 00  	sh	t0, 2(a0)
   1bb46: 23 22 f5 01  	sw	t6, 4(a0)
   1bb4a: 23 14 e5 01  	sh	t5, 8(a0)
   1bb4e: 23 15 d5 01  	sh	t4, 10(a0)
   1bb52: 23 16 c5 01  	sh	t3, 12(a0)
   1bb56: 23 17 65 00  	sh	t1, 14(a0)
   1bb5a: 23 38 15 01  	sd	a7, 16(a0)
   1bb5e: 23 38 05 05  	sd	a6, 80(a0)
   1bb62: 30 e5        	sd	a2, 72(a0)
   1bb64: 14 ed        	sd	a3, 24(a0)
   1bb66: 18 f5        	sd	a4, 40(a0)
   1bb68: 1c fd        	sd	a5, 56(a0)
   1bb6a: 82 80        	ret

000000000001bb6c <__netf2>:
   1bb6c: f3 27 20 00  	frrm	a5
   1bb70: a1 67        	lui	a5, 8
   1bb72: 13 d8 05 03  	srli	a6, a1, 48
   1bb76: fd 17        	addi	a5, a5, -1
   1bb78: 7d 5e        	addi	t3, zero, -1
   1bb7a: 13 57 0e 01  	srli	a4, t3, 16
   1bb7e: 13 d3 06 03  	srli	t1, a3, 48
   1bb82: 33 78 f8 00  	and	a6, a6, a5
   1bb86: b3 f8 e5 00  	and	a7, a1, a4
   1bb8a: 33 73 f3 00  	and	t1, t1, a5
   1bb8e: 75 8f        	and	a4, a4, a3
   1bb90: fd 91        	srli	a1, a1, 63
   1bb92: fd 92        	srli	a3, a3, 63
   1bb94: 63 1f f8 00  	bne	a6, a5, 0x1bbb2 <__netf2+0x46>
   1bb98: 33 e5 a8 00  	or	a0, a7, a0
   1bb9c: 0d e9        	bnez	a0, 0x1bbce <__netf2+0x62>
   1bb9e: 85 47        	addi	a5, zero, 1
   1bba0: 63 17 03 01  	bne	t1, a6, 0x1bbae <__netf2+0x42>
   1bba4: 59 8e        	or	a2, a2, a4
   1bba6: 19 ea        	bnez	a2, 0x1bbbc <__netf2+0x50>
   1bba8: 95 8d        	sub	a1, a1, a3
   1bbaa: b3 37 b0 00  	snez	a5, a1
   1bbae: 3e 85        	mv	a0, a5
   1bbb0: 05 a8        	j	0x1bbe0 <__netf2+0x74>
   1bbb2: 63 18 f3 02  	bne	t1, a5, 0x1bbe2 <__netf2+0x76>
   1bbb6: 59 8e        	or	a2, a2, a4
   1bbb8: 85 47        	addi	a5, zero, 1
   1bbba: 75 da        	beqz	a2, 0x1bbae <__netf2+0x42>
   1bbbc: fd 57        	addi	a5, zero, -1
   1bbbe: c5 83        	srli	a5, a5, 17
   1bbc0: 05 45        	addi	a0, zero, 1
   1bbc2: 63 ef e7 00  	bltu	a5, a4, 0x1bbe0 <__netf2+0x74>
   1bbc6: 73 60 18 00  	csrsi	fflags, 16
   1bbca: 05 45        	addi	a0, zero, 1
   1bbcc: 82 80        	ret
   1bbce: 13 5e 1e 01  	srli	t3, t3, 17
   1bbd2: e3 7a 1e ff  	bgeu	t3, a7, 0x1bbc6 <__netf2+0x5a>
   1bbd6: 05 45        	addi	a0, zero, 1
   1bbd8: 63 14 03 01  	bne	t1, a6, 0x1bbe0 <__netf2+0x74>
   1bbdc: 59 8e        	or	a2, a2, a4
   1bbde: 79 fe        	bnez	a2, 0x1bbbc <__netf2+0x50>
   1bbe0: 82 80        	ret
   1bbe2: 85 47        	addi	a5, zero, 1
   1bbe4: e3 15 68 fc  	bne	a6, t1, 0x1bbae <__netf2+0x42>
   1bbe8: e3 93 e8 fc  	bne	a7, a4, 0x1bbae <__netf2+0x42>
   1bbec: e3 11 c5 fc  	bne	a0, a2, 0x1bbae <__netf2+0x42>
   1bbf0: 63 89 d5 00  	beq	a1, a3, 0x1bc02 <__netf2+0x96>
   1bbf4: e3 1d 08 fa  	bnez	a6, 0x1bbae <__netf2+0x42>
   1bbf8: b3 e8 a8 00  	or	a7, a7, a0
   1bbfc: b3 37 10 01  	snez	a5, a7
   1bc00: 7d b7        	j	0x1bbae <__netf2+0x42>
   1bc02: 81 47        	mv	a5, zero
   1bc04: 6d b7        	j	0x1bbae <__netf2+0x42>

000000000001bc06 <__gttf2>:
   1bc06: 2a 83        	mv	t1, a0
   1bc08: f3 27 20 00  	frrm	a5
   1bc0c: 21 6e        	lui	t3, 8
   1bc0e: 93 d8 05 03  	srli	a7, a1, 48
   1bc12: 7d 1e        	addi	t3, t3, -1
   1bc14: 7d 57        	addi	a4, zero, -1
   1bc16: 41 83        	srli	a4, a4, 16
   1bc18: 93 d7 06 03  	srli	a5, a3, 48
   1bc1c: b3 f8 c8 01  	and	a7, a7, t3
   1bc20: 33 f8 e5 00  	and	a6, a1, a4
   1bc24: 33 f5 c7 01  	and	a0, a5, t3
   1bc28: 75 8f        	and	a4, a4, a3
   1bc2a: fd 91        	srli	a1, a1, 63
   1bc2c: fd 92        	srli	a3, a3, 63
   1bc2e: 63 92 c8 03  	bne	a7, t3, 0x1bc52 <__gttf2+0x4c>
   1bc32: 33 68 68 00  	or	a6, a6, t1
   1bc36: 63 14 08 08  	bnez	a6, 0x1bcbe <__gttf2+0xb8>
   1bc3a: 63 06 15 01  	beq	a0, a7, 0x1bc46 <__gttf2+0x40>
   1bc3e: 05 45        	addi	a0, zero, 1
   1bc40: b5 cd        	beqz	a1, 0x1bcbc <__gttf2+0xb6>
   1bc42: 7d 55        	addi	a0, zero, -1
   1bc44: 82 80        	ret
   1bc46: 51 8f        	or	a4, a4, a2
   1bc48: 3d eb        	bnez	a4, 0x1bcbe <__gttf2+0xb8>
   1bc4a: e3 9a d5 fe  	bne	a1, a3, 0x1bc3e <__gttf2+0x38>
   1bc4e: 01 45        	mv	a0, zero
   1bc50: 82 80        	ret
   1bc52: 63 14 c5 03  	bne	a0, t3, 0x1bc7a <__gttf2+0x74>
   1bc56: 51 8f        	or	a4, a4, a2
   1bc58: 3d e3        	bnez	a4, 0x1bcbe <__gttf2+0xb8>
   1bc5a: 63 88 08 00  	beqz	a7, 0x1bc6a <__gttf2+0x64>
   1bc5e: e3 90 d5 fe  	bne	a1, a3, 0x1bc3e <__gttf2+0x38>
   1bc62: 7d 55        	addi	a0, zero, -1
   1bc64: a1 cd        	beqz	a1, 0x1bcbc <__gttf2+0xb6>
   1bc66: 2e 85        	mv	a0, a1
   1bc68: 82 80        	ret
   1bc6a: 33 68 68 00  	or	a6, a6, t1
   1bc6e: e3 18 08 fe  	bnez	a6, 0x1bc5e <__gttf2+0x58>
   1bc72: 7d 55        	addi	a0, zero, -1
   1bc74: a1 c6        	beqz	a3, 0x1bcbc <__gttf2+0xb6>
   1bc76: 36 85        	mv	a0, a3
   1bc78: 82 80        	ret
   1bc7a: 63 83 08 02  	beqz	a7, 0x1bca0 <__gttf2+0x9a>
   1bc7e: 61 d1        	beqz	a0, 0x1bc3e <__gttf2+0x38>
   1bc80: e3 9f d5 fa  	bne	a1, a3, 0x1bc3e <__gttf2+0x38>
   1bc84: e3 4d 15 fb  	blt	a0, a7, 0x1bc3e <__gttf2+0x38>
   1bc88: e3 cd a8 fc  	blt	a7, a0, 0x1bc62 <__gttf2+0x5c>
   1bc8c: e3 69 07 fb  	bltu	a4, a6, 0x1bc3e <__gttf2+0x38>
   1bc90: e3 19 e8 fc  	bne	a6, a4, 0x1bc62 <__gttf2+0x5c>
   1bc94: e3 65 66 fa  	bltu	a2, t1, 0x1bc3e <__gttf2+0x38>
   1bc98: 01 45        	mv	a0, zero
   1bc9a: 63 71 c3 02  	bgeu	t1, a2, 0x1bcbc <__gttf2+0xb6>
   1bc9e: d1 b7        	j	0x1bc62 <__gttf2+0x5c>
   1bca0: b3 67 68 00  	or	a5, a6, t1
   1bca4: 19 c1        	beqz	a0, 0x1bcaa <__gttf2+0xa4>
   1bca6: f1 d7        	beqz	a5, 0x1bc72 <__gttf2+0x6c>
   1bca8: 5d bf        	j	0x1bc5e <__gttf2+0x58>
   1bcaa: b3 68 c7 00  	or	a7, a4, a2
   1bcae: 63 86 08 00  	beqz	a7, 0x1bcba <__gttf2+0xb4>
   1bcb2: e1 d3        	beqz	a5, 0x1bc72 <__gttf2+0x6c>
   1bcb4: e3 8c d5 fc  	beq	a1, a3, 0x1bc8c <__gttf2+0x86>
   1bcb8: 59 b7        	j	0x1bc3e <__gttf2+0x38>
   1bcba: d1 f3        	bnez	a5, 0x1bc3e <__gttf2+0x38>
   1bcbc: 82 80        	ret
   1bcbe: 73 60 18 00  	csrsi	fflags, 16
   1bcc2: 79 55        	addi	a0, zero, -2
   1bcc4: 82 80        	ret

000000000001bcc6 <__lttf2>:
   1bcc6: 2a 83        	mv	t1, a0
   1bcc8: f3 27 20 00  	frrm	a5
   1bccc: 21 6e        	lui	t3, 8
   1bcce: 93 d8 05 03  	srli	a7, a1, 48
   1bcd2: 7d 1e        	addi	t3, t3, -1
   1bcd4: 7d 57        	addi	a4, zero, -1
   1bcd6: 41 83        	srli	a4, a4, 16
   1bcd8: 93 d7 06 03  	srli	a5, a3, 48
   1bcdc: b3 f8 c8 01  	and	a7, a7, t3
   1bce0: 33 f8 e5 00  	and	a6, a1, a4
   1bce4: 33 f5 c7 01  	and	a0, a5, t3
   1bce8: 75 8f        	and	a4, a4, a3
   1bcea: fd 91        	srli	a1, a1, 63
   1bcec: fd 92        	srli	a3, a3, 63
   1bcee: 63 92 c8 03  	bne	a7, t3, 0x1bd12 <__lttf2+0x4c>
   1bcf2: 33 68 68 00  	or	a6, a6, t1
   1bcf6: 63 14 08 08  	bnez	a6, 0x1bd7e <__lttf2+0xb8>
   1bcfa: 63 06 15 01  	beq	a0, a7, 0x1bd06 <__lttf2+0x40>
   1bcfe: 05 45        	addi	a0, zero, 1
   1bd00: b5 cd        	beqz	a1, 0x1bd7c <__lttf2+0xb6>
   1bd02: 7d 55        	addi	a0, zero, -1
   1bd04: 82 80        	ret
   1bd06: 51 8f        	or	a4, a4, a2
   1bd08: 3d eb        	bnez	a4, 0x1bd7e <__lttf2+0xb8>
   1bd0a: e3 9a d5 fe  	bne	a1, a3, 0x1bcfe <__lttf2+0x38>
   1bd0e: 01 45        	mv	a0, zero
   1bd10: 82 80        	ret
   1bd12: 63 14 c5 03  	bne	a0, t3, 0x1bd3a <__lttf2+0x74>
   1bd16: 51 8f        	or	a4, a4, a2
   1bd18: 3d e3        	bnez	a4, 0x1bd7e <__lttf2+0xb8>
   1bd1a: 63 88 08 00  	beqz	a7, 0x1bd2a <__lttf2+0x64>
   1bd1e: e3 90 d5 fe  	bne	a1, a3, 0x1bcfe <__lttf2+0x38>
   1bd22: 7d 55        	addi	a0, zero, -1
   1bd24: a1 cd        	beqz	a1, 0x1bd7c <__lttf2+0xb6>
   1bd26: 2e 85        	mv	a0, a1
   1bd28: 82 80        	ret
   1bd2a: 33 68 68 00  	or	a6, a6, t1
   1bd2e: e3 18 08 fe  	bnez	a6, 0x1bd1e <__lttf2+0x58>
   1bd32: 7d 55        	addi	a0, zero, -1
   1bd34: a1 c6        	beqz	a3, 0x1bd7c <__lttf2+0xb6>
   1bd36: 36 85        	mv	a0, a3
   1bd38: 82 80        	ret
   1bd3a: 63 83 08 02  	beqz	a7, 0x1bd60 <__lttf2+0x9a>
   1bd3e: 61 d1        	beqz	a0, 0x1bcfe <__lttf2+0x38>
   1bd40: e3 9f d5 fa  	bne	a1, a3, 0x1bcfe <__lttf2+0x38>
   1bd44: e3 4d 15 fb  	blt	a0, a7, 0x1bcfe <__lttf2+0x38>
   1bd48: e3 cd a8 fc  	blt	a7, a0, 0x1bd22 <__lttf2+0x5c>
   1bd4c: e3 69 07 fb  	bltu	a4, a6, 0x1bcfe <__lttf2+0x38>
   1bd50: e3 19 e8 fc  	bne	a6, a4, 0x1bd22 <__lttf2+0x5c>
   1bd54: e3 65 66 fa  	bltu	a2, t1, 0x1bcfe <__lttf2+0x38>
   1bd58: 01 45        	mv	a0, zero
   1bd5a: 63 71 c3 02  	bgeu	t1, a2, 0x1bd7c <__lttf2+0xb6>
   1bd5e: d1 b7        	j	0x1bd22 <__lttf2+0x5c>
   1bd60: b3 67 68 00  	or	a5, a6, t1
   1bd64: 19 c1        	beqz	a0, 0x1bd6a <__lttf2+0xa4>
   1bd66: f1 d7        	beqz	a5, 0x1bd32 <__lttf2+0x6c>
   1bd68: 5d bf        	j	0x1bd1e <__lttf2+0x58>
   1bd6a: b3 68 c7 00  	or	a7, a4, a2
   1bd6e: 63 86 08 00  	beqz	a7, 0x1bd7a <__lttf2+0xb4>
   1bd72: e1 d3        	beqz	a5, 0x1bd32 <__lttf2+0x6c>
   1bd74: e3 8c d5 fc  	beq	a1, a3, 0x1bd4c <__lttf2+0x86>
   1bd78: 59 b7        	j	0x1bcfe <__lttf2+0x38>
   1bd7a: d1 f3        	bnez	a5, 0x1bcfe <__lttf2+0x38>
   1bd7c: 82 80        	ret
   1bd7e: 73 60 18 00  	csrsi	fflags, 16
   1bd82: 09 45        	addi	a0, zero, 2
   1bd84: 82 80        	ret

000000000001bd86 <__multf3>:
   1bd86: 1d 71        	addi	sp, sp, -96
   1bd88: a2 e8        	sd	s0, 80(sp)
   1bd8a: 52 f8        	sd	s4, 48(sp)
   1bd8c: 62 e8        	sd	s8, 16(sp)
   1bd8e: 86 ec        	sd	ra, 88(sp)
   1bd90: a6 e4        	sd	s1, 72(sp)
   1bd92: ca e0        	sd	s2, 64(sp)
   1bd94: 4e fc        	sd	s3, 56(sp)
   1bd96: 56 f4        	sd	s5, 40(sp)
   1bd98: 5a f0        	sd	s6, 32(sp)
   1bd9a: 5e ec        	sd	s7, 24(sp)
   1bd9c: 66 e4        	sd	s9, 8(sp)
   1bd9e: 2a 84        	mv	s0, a0
   1bda0: 32 8c        	mv	s8, a2
   1bda2: 36 8a        	mv	s4, a3
   1bda4: f3 29 20 00  	frrm	s3
   1bda8: 21 67        	lui	a4, 8
   1bdaa: 93 d7 05 03  	srli	a5, a1, 48
   1bdae: 7d 17        	addi	a4, a4, -1
   1bdb0: 13 99 05 01  	slli	s2, a1, 16
   1bdb4: f9 8f        	and	a5, a5, a4
   1bdb6: 81 29        	sext.w	s3, s3
   1bdb8: 13 59 09 01  	srli	s2, s2, 16
   1bdbc: 93 da f5 03  	srli	s5, a1, 63
   1bdc0: 95 c7        	beqz	a5, 0x1bdec <__multf3+0x66>
   1bdc2: 9b 8c 07 00  	sext.w	s9, a5
   1bdc6: 63 84 e7 08  	beq	a5, a4, 0x1be4e <__multf3+0xc8>
   1bdca: 0e 09        	slli	s2, s2, 3
   1bdcc: 13 57 d5 03  	srli	a4, a0, 61
   1bdd0: 33 67 27 01  	or	a4, a4, s2
   1bdd4: f1 7c        	lui	s9, 1048572
   1bdd6: 05 49        	addi	s2, zero, 1
   1bdd8: 4e 19        	slli	s2, s2, 51
   1bdda: 85 0c        	addi	s9, s9, 1
   1bddc: 33 69 27 01  	or	s2, a4, s2
   1bde0: 13 1b 35 00  	slli	s6, a0, 3
   1bde4: be 9c        	add	s9, s9, a5
   1bde6: 81 4b        	mv	s7, zero
   1bde8: 81 44        	mv	s1, zero
   1bdea: b5 a8        	j	0x1be66 <__multf3+0xe0>
   1bdec: 33 6b a9 00  	or	s6, s2, a0
   1bdf0: 63 0b 0b 0e  	beqz	s6, 0x1bee6 <__multf3+0x160>
   1bdf4: 63 0e 09 02  	beqz	s2, 0x1be30 <__multf3+0xaa>
   1bdf8: 4a 85        	mv	a0, s2
   1bdfa: ef 10 a0 30  	jal	0x1d104 <__clzdi2>
   1bdfe: 1b 07 05 00  	sext.w	a4, a0
   1be02: 93 07 17 ff  	addi	a5, a4, -15
   1be06: 93 06 d0 03  	addi	a3, zero, 61
   1be0a: 1b 86 37 00  	addiw	a2, a5, 3
   1be0e: bb 87 f6 40  	subw	a5, a3, a5
   1be12: 33 19 c9 00  	sll	s2, s2, a2
   1be16: b3 57 f4 00  	srl	a5, s0, a5
   1be1a: b3 e7 27 01  	or	a5, a5, s2
   1be1e: 33 19 c4 00  	sll	s2, s0, a2
   1be22: f1 7c        	lui	s9, 1048572
   1be24: c5 0c        	addi	s9, s9, 17
   1be26: 4a 8b        	mv	s6, s2
   1be28: b3 8c ec 40  	sub	s9, s9, a4
   1be2c: 3e 89        	mv	s2, a5
   1be2e: 65 bf        	j	0x1bde6 <__multf3+0x60>
   1be30: ef 10 40 2d  	jal	0x1d104 <__clzdi2>
   1be34: 1b 07 05 04  	addiw	a4, a0, 64
   1be38: 93 07 17 ff  	addi	a5, a4, -15
   1be3c: 93 06 c0 03  	addi	a3, zero, 60
   1be40: e3 d3 f6 fc  	bge	a3, a5, 0x1be06 <__multf3+0x80>
   1be44: 9b 87 37 fc  	addiw	a5, a5, -61
   1be48: b3 17 f4 00  	sll	a5, s0, a5
   1be4c: d9 bf        	j	0x1be22 <__multf3+0x9c>
   1be4e: 33 6b a9 00  	or	s6, s2, a0
   1be52: 63 0e 0b 08  	beqz	s6, 0x1beee <__multf3+0x168>
   1be56: 85 44        	addi	s1, zero, 1
   1be58: be 14        	slli	s1, s1, 47
   1be5a: b3 34 99 00  	sltu	s1, s2, s1
   1be5e: 9b 94 44 00  	slliw	s1, s1, 4
   1be62: 2a 8b        	mv	s6, a0
   1be64: 8d 4b        	addi	s7, zero, 3
   1be66: a1 65        	lui	a1, 8
   1be68: 13 56 0a 03  	srli	a2, s4, 48
   1be6c: fd 56        	addi	a3, zero, -1
   1be6e: fd 15        	addi	a1, a1, -1
   1be70: 13 d4 06 01  	srli	s0, a3, 16
   1be74: 6d 8e        	and	a2, a2, a1
   1be76: 33 74 8a 00  	and	s0, s4, s0
   1be7a: e2 87        	mv	a5, s8
   1be7c: 13 5a fa 03  	srli	s4, s4, 63
   1be80: 35 ca        	beqz	a2, 0x1bef4 <__multf3+0x16e>
   1be82: 1b 07 06 00  	sext.w	a4, a2
   1be86: 63 05 b6 0c  	beq	a2, a1, 0x1bf50 <__multf3+0x1ca>
   1be8a: 0e 04        	slli	s0, s0, 3
   1be8c: 13 57 dc 03  	srli	a4, s8, 61
   1be90: 41 8f        	or	a4, a4, s0
   1be92: 05 44        	addi	s0, zero, 1
   1be94: 4e 14        	slli	s0, s0, 51
   1be96: 59 8c        	or	s0, s0, a4
   1be98: 71 77        	lui	a4, 1048572
   1be9a: 05 07        	addi	a4, a4, 1
   1be9c: 93 17 3c 00  	slli	a5, s8, 3
   1bea0: 32 97        	add	a4, a4, a2
   1bea2: 81 48        	mv	a7, zero
   1bea4: 93 96 2b 00  	slli	a3, s7, 2
   1bea8: 33 08 97 01  	add	a6, a4, s9
   1beac: b3 e6 16 01  	or	a3, a3, a7
   1beb0: 29 47        	addi	a4, zero, 10
   1beb2: 33 c6 4a 01  	xor	a2, s5, s4
   1beb6: 93 05 18 00  	addi	a1, a6, 1
   1beba: 63 41 d7 2c  	blt	a4, a3, 0x1c17c <__multf3+0x3f6>
   1bebe: 09 47        	addi	a4, zero, 2
   1bec0: 63 4a d7 0a  	blt	a4, a3, 0x1bf74 <__multf3+0x1ee>
   1bec4: fd 16        	addi	a3, a3, -1
   1bec6: 05 47        	addi	a4, zero, 1
   1bec8: 63 65 d7 0c  	bltu	a4, a3, 0x1bf92 <__multf3+0x20c>
   1becc: c6 8b        	mv	s7, a7
   1bece: 09 47        	addi	a4, zero, 2
   1bed0: 63 80 eb 4c  	beq	s7, a4, 0x1c390 <__multf3+0x60a>
   1bed4: 0d 47        	addi	a4, zero, 3
   1bed6: 63 82 eb 4c  	beq	s7, a4, 0x1c39a <__multf3+0x614>
   1beda: 05 47        	addi	a4, zero, 1
   1bedc: 63 92 eb 26  	bne	s7, a4, 0x1c140 <__multf3+0x3ba>
   1bee0: 01 47        	mv	a4, zero
   1bee2: 01 45        	mv	a0, zero
   1bee4: 79 a1        	j	0x1c372 <__multf3+0x5ec>
   1bee6: 01 49        	mv	s2, zero
   1bee8: 81 4c        	mv	s9, zero
   1beea: 85 4b        	addi	s7, zero, 1
   1beec: f5 bd        	j	0x1bde8 <__multf3+0x62>
   1beee: 01 49        	mv	s2, zero
   1bef0: 89 4b        	addi	s7, zero, 2
   1bef2: dd bd        	j	0x1bde8 <__multf3+0x62>
   1bef4: b3 67 8c 00  	or	a5, s8, s0
   1bef8: ad c7        	beqz	a5, 0x1bf62 <__multf3+0x1dc>
   1befa: 1d c8        	beqz	s0, 0x1bf30 <__multf3+0x1aa>
   1befc: 22 85        	mv	a0, s0
   1befe: ef 10 60 20  	jal	0x1d104 <__clzdi2>
   1bf02: 01 25        	sext.w	a0, a0
   1bf04: 93 07 15 ff  	addi	a5, a0, -15
   1bf08: 13 07 d0 03  	addi	a4, zero, 61
   1bf0c: 1b 86 37 00  	addiw	a2, a5, 3
   1bf10: bb 07 f7 40  	subw	a5, a4, a5
   1bf14: 33 14 c4 00  	sll	s0, s0, a2
   1bf18: b3 57 fc 00  	srl	a5, s8, a5
   1bf1c: b3 e6 87 00  	or	a3, a5, s0
   1bf20: 33 14 cc 00  	sll	s0, s8, a2
   1bf24: 71 77        	lui	a4, 1048572
   1bf26: 45 07        	addi	a4, a4, 17
   1bf28: a2 87        	mv	a5, s0
   1bf2a: 09 8f        	sub	a4, a4, a0
   1bf2c: 36 84        	mv	s0, a3
   1bf2e: 95 bf        	j	0x1bea2 <__multf3+0x11c>
   1bf30: 62 85        	mv	a0, s8
   1bf32: ef 10 20 1d  	jal	0x1d104 <__clzdi2>
   1bf36: 1b 05 05 04  	addiw	a0, a0, 64
   1bf3a: 93 07 15 ff  	addi	a5, a0, -15
   1bf3e: 13 07 c0 03  	addi	a4, zero, 60
   1bf42: e3 53 f7 fc  	bge	a4, a5, 0x1bf08 <__multf3+0x182>
   1bf46: 9b 87 37 fc  	addiw	a5, a5, -61
   1bf4a: b3 16 fc 00  	sll	a3, s8, a5
   1bf4e: d9 bf        	j	0x1bf24 <__multf3+0x19e>
   1bf50: 33 66 8c 00  	or	a2, s8, s0
   1bf54: 01 ce        	beqz	a2, 0x1bf6c <__multf3+0x1e6>
   1bf56: c5 82        	srli	a3, a3, 17
   1bf58: 8d 48        	addi	a7, zero, 3
   1bf5a: e3 e5 86 f4  	bltu	a3, s0, 0x1bea4 <__multf3+0x11e>
   1bf5e: c1 44        	addi	s1, zero, 16
   1bf60: 91 b7        	j	0x1bea4 <__multf3+0x11e>
   1bf62: 01 44        	mv	s0, zero
   1bf64: 81 47        	mv	a5, zero
   1bf66: 01 47        	mv	a4, zero
   1bf68: 85 48        	addi	a7, zero, 1
   1bf6a: 2d bf        	j	0x1bea4 <__multf3+0x11e>
   1bf6c: 01 44        	mv	s0, zero
   1bf6e: 81 47        	mv	a5, zero
   1bf70: 89 48        	addi	a7, zero, 2
   1bf72: 0d bf        	j	0x1bea4 <__multf3+0x11e>
   1bf74: 05 47        	addi	a4, zero, 1
   1bf76: b3 16 d7 00  	sll	a3, a4, a3
   1bf7a: 13 f5 06 53  	andi	a0, a3, 1328
   1bf7e: 63 18 05 20  	bnez	a0, 0x1c18e <__multf3+0x408>
   1bf82: 13 f3 06 24  	andi	t1, a3, 576
   1bf86: 63 1f 03 3e  	bnez	t1, 0x1c384 <__multf3+0x5fe>
   1bf8a: 93 f6 86 08  	andi	a3, a3, 136
   1bf8e: 63 9e 06 1e  	bnez	a3, 0x1c18a <__multf3+0x404>
   1bf92: fd 56        	addi	a3, zero, -1
   1bf94: 81 92        	srli	a3, a3, 32
   1bf96: 33 8f 87 00  	add	t5, a5, s0
   1bf9a: 13 55 0b 02  	srli	a0, s6, 32
   1bf9e: 13 d3 07 02  	srli	t1, a5, 32
   1bfa2: b3 0f 69 01  	add	t6, s2, s6
   1bfa6: b3 3e ff 00  	sltu	t4, t5, a5
   1bfaa: 33 7b db 00  	and	s6, s6, a3
   1bfae: f5 8f        	and	a5, a5, a3
   1bfb0: b3 82 67 03  	mul	t0, a5, s6
   1bfb4: 33 be 2f 01  	sltu	t3, t6, s2
   1bfb8: f2 88        	mv	a7, t3
   1bfba: 76 87        	mv	a4, t4
   1bfbc: b3 07 f5 02  	mul	a5, a0, a5
   1bfc0: b3 06 65 02  	mul	a3, a0, t1
   1bfc4: 13 d5 02 02  	srli	a0, t0, 32
   1bfc8: 33 03 63 03  	mul	t1, t1, s6
   1bfcc: 3e 93        	add	t1, t1, a5
   1bfce: 1a 95        	add	a0, a0, t1
   1bfd0: 63 75 f5 00  	bgeu	a0, a5, 0x1bfda <__multf3+0x254>
   1bfd4: 85 47        	addi	a5, zero, 1
   1bfd6: 82 17        	slli	a5, a5, 32
   1bfd8: be 96        	add	a3, a3, a5
   1bfda: fd 57        	addi	a5, zero, -1
   1bfdc: 81 93        	srli	a5, a5, 32
   1bfde: 13 53 05 02  	srli	t1, a0, 32
   1bfe2: 7d 8d        	and	a0, a0, a5
   1bfe4: b3 f2 f2 00  	and	t0, t0, a5
   1bfe8: 02 15        	slli	a0, a0, 32
   1bfea: b3 f3 ff 00  	and	t2, t6, a5
   1bfee: 36 93        	add	t1, t1, a3
   1bff0: b3 77 ff 00  	and	a5, t5, a5
   1bff4: 93 56 0f 02  	srli	a3, t5, 32
   1bff8: 16 95        	add	a0, a0, t0
   1bffa: 93 d2 0f 02  	srli	t0, t6, 32
   1bffe: b3 8a 77 02  	mul	s5, a5, t2
   1c002: b3 87 f2 02  	mul	a5, t0, a5
   1c006: 13 da 0a 02  	srli	s4, s5, 32
   1c00a: b3 82 d2 02  	mul	t0, t0, a3
   1c00e: b3 86 76 02  	mul	a3, a3, t2
   1c012: be 96        	add	a3, a3, a5
   1c014: 36 9a        	add	s4, s4, a3
   1c016: 63 75 fa 00  	bgeu	s4, a5, 0x1c020 <__multf3+0x29a>
   1c01a: 85 47        	addi	a5, zero, 1
   1c01c: 82 17        	slli	a5, a5, 32
   1c01e: be 92        	add	t0, t0, a5
   1c020: fd 57        	addi	a5, zero, -1
   1c022: 81 93        	srli	a5, a5, 32
   1c024: 93 53 0a 02  	srli	t2, s4, 32
   1c028: 33 7a fa 00  	and	s4, s4, a5
   1c02c: b3 fa fa 00  	and	s5, s5, a5
   1c030: 02 1a        	slli	s4, s4, 32
   1c032: 93 56 04 02  	srli	a3, s0, 32
   1c036: 56 9a        	add	s4, s4, s5
   1c038: 93 5a 09 02  	srli	s5, s2, 32
   1c03c: 33 79 f9 00  	and	s2, s2, a5
   1c040: e1 8f        	and	a5, a5, s0
   1c042: 33 0b f9 02  	mul	s6, s2, a5
   1c046: b3 87 fa 02  	mul	a5, s5, a5
   1c04a: 13 54 0b 02  	srli	s0, s6, 32
   1c04e: b3 8a da 02  	mul	s5, s5, a3
   1c052: b3 86 26 03  	mul	a3, a3, s2
   1c056: be 96        	add	a3, a3, a5
   1c058: 36 94        	add	s0, s0, a3
   1c05a: 63 75 f4 00  	bgeu	s0, a5, 0x1c064 <__multf3+0x2de>
   1c05e: 85 47        	addi	a5, zero, 1
   1c060: 82 17        	slli	a5, a5, 32
   1c062: be 9a        	add	s5, s5, a5
   1c064: 7d 59        	addi	s2, zero, -1
   1c066: 13 59 09 02  	srli	s2, s2, 32
   1c06a: 93 56 04 02  	srli	a3, s0, 32
   1c06e: b3 77 24 01  	and	a5, s0, s2
   1c072: 33 07 e0 40  	neg	a4, a4
   1c076: 33 04 6a 00  	add	s0, s4, t1
   1c07a: 33 77 f7 01  	and	a4, a4, t6
   1c07e: b3 08 10 41  	neg	a7, a7
   1c082: b3 3f 44 01  	sltu	t6, s0, s4
   1c086: b3 f8 e8 01  	and	a7, a7, t5
   1c08a: 33 8f f3 01  	add	t5, t2, t6
   1c08e: 16 9f        	add	t5, t5, t0
   1c090: 33 7e de 01  	and	t3, t3, t4
   1c094: b3 3f ff 01  	sltu	t6, t5, t6
   1c098: 7a 97        	add	a4, a4, t5
   1c09a: f2 9f        	add	t6, t6, t3
   1c09c: 33 3f e7 01  	sltu	t5, a4, t5
   1c0a0: 46 97        	add	a4, a4, a7
   1c0a2: 7e 9f        	add	t5, t5, t6
   1c0a4: b3 38 17 01  	sltu	a7, a4, a7
   1c0a8: b3 0e a4 40  	sub	t4, s0, a0
   1c0ac: 33 0e 67 40  	sub	t3, a4, t1
   1c0b0: 82 17        	slli	a5, a5, 32
   1c0b2: 33 7b 2b 01  	and	s6, s6, s2
   1c0b6: fa 98        	add	a7, a7, t5
   1c0b8: 33 3f d4 01  	sltu	t5, s0, t4
   1c0bc: b3 3f c7 01  	sltu	t6, a4, t3
   1c0c0: d6 96        	add	a3, a3, s5
   1c0c2: 33 0e ee 41  	sub	t3, t3, t5
   1c0c6: da 97        	add	a5, a5, s6
   1c0c8: 01 4f        	mv	t5, zero
   1c0ca: 63 76 d4 01  	bgeu	s0, t4, 0x1c0d6 <__multf3+0x350>
   1c0ce: 33 03 e3 40  	sub	t1, t1, a4
   1c0d2: 13 3f 13 00  	seqz	t5, t1
   1c0d6: 33 6f ff 01  	or	t5, t5, t6
   1c0da: b3 8f fe 40  	sub	t6, t4, a5
   1c0de: b6 98        	add	a7, a7, a3
   1c0e0: 33 b7 fe 01  	sltu	a4, t4, t6
   1c0e4: b3 06 de 40  	sub	a3, t3, a3
   1c0e8: 33 3e de 00  	sltu	t3, t3, a3
   1c0ec: 33 87 e7 40  	sub	a4, a5, a4
   1c0f0: 01 43        	mv	t1, zero
   1c0f2: 63 f4 fe 01  	bgeu	t4, t6, 0x1c0fa <__multf3+0x374>
   1c0f6: 13 b3 16 00  	seqz	t1, a3
   1c0fa: ba 96        	add	a3, a3, a4
   1c0fc: 33 b7 f6 00  	sltu	a4, a3, a5
   1c100: 46 97        	add	a4, a4, a7
   1c102: 33 07 e7 41  	sub	a4, a4, t5
   1c106: 33 63 c3 01  	or	t1, t1, t3
   1c10a: 33 07 67 40  	sub	a4, a4, t1
   1c10e: 93 97 df 00  	slli	a5, t6, 13
   1c112: 36 07        	slli	a4, a4, 13
   1c114: 13 d4 36 03  	srli	s0, a3, 51
   1c118: c9 8f        	or	a5, a5, a0
   1c11a: 59 8c        	or	s0, s0, a4
   1c11c: b3 37 f0 00  	snez	a5, a5
   1c120: 13 d5 3f 03  	srli	a0, t6, 51
   1c124: 51 93        	srli	a4, a4, 52
   1c126: c9 8f        	or	a5, a5, a0
   1c128: b6 06        	slli	a3, a3, 13
   1c12a: 05 8b        	andi	a4, a4, 1
   1c12c: d5 8f        	or	a5, a5, a3
   1c12e: 3d c3        	beqz	a4, 0x1c194 <__multf3+0x40e>
   1c130: 13 d7 17 00  	srli	a4, a5, 1
   1c134: 85 8b        	andi	a5, a5, 1
   1c136: 5d 8f        	or	a4, a4, a5
   1c138: 93 17 f4 03  	slli	a5, s0, 63
   1c13c: d9 8f        	or	a5, a5, a4
   1c13e: 05 80        	srli	s0, s0, 1
   1c140: 91 66        	lui	a3, 4
   1c142: fd 16        	addi	a3, a3, -1
   1c144: ae 96        	add	a3, a3, a1
   1c146: 63 50 d0 0e  	blez	a3, 0x1c226 <__multf3+0x4a0>
   1c14a: 13 f7 77 00  	andi	a4, a5, 7
   1c14e: 31 cb        	beqz	a4, 0x1c1a2 <__multf3+0x41c>
   1c150: 09 47        	addi	a4, zero, 2
   1c152: 93 e4 14 00  	ori	s1, s1, 1
   1c156: 63 85 e9 04  	beq	s3, a4, 0x1c1a0 <__multf3+0x41a>
   1c15a: 0d 47        	addi	a4, zero, 3
   1c15c: 63 8e e9 02  	beq	s3, a4, 0x1c198 <__multf3+0x412>
   1c160: 63 91 09 04  	bnez	s3, 0x1c1a2 <__multf3+0x41c>
   1c164: 13 f7 f7 00  	andi	a4, a5, 15
   1c168: 11 45        	addi	a0, zero, 4
   1c16a: 63 0c a7 02  	beq	a4, a0, 0x1c1a2 <__multf3+0x41c>
   1c16e: 13 87 47 00  	addi	a4, a5, 4
   1c172: 33 35 f7 00  	sltu	a0, a4, a5
   1c176: 2a 94        	add	s0, s0, a0
   1c178: ba 87        	mv	a5, a4
   1c17a: 25 a0        	j	0x1c1a2 <__multf3+0x41c>
   1c17c: 3d 47        	addi	a4, zero, 15
   1c17e: 63 8e e6 20  	beq	a3, a4, 0x1c39a <__multf3+0x614>
   1c182: 2d 47        	addi	a4, zero, 11
   1c184: 56 86        	mv	a2, s5
   1c186: 63 94 e6 00  	bne	a3, a4, 0x1c18e <__multf3+0x408>
   1c18a: 52 86        	mv	a2, s4
   1c18c: 81 b3        	j	0x1becc <__multf3+0x146>
   1c18e: 4a 84        	mv	s0, s2
   1c190: da 87        	mv	a5, s6
   1c192: 35 bb        	j	0x1bece <__multf3+0x148>
   1c194: c2 85        	mv	a1, a6
   1c196: 6d b7        	j	0x1c140 <__multf3+0x3ba>
   1c198: 09 e6        	bnez	a2, 0x1c1a2 <__multf3+0x41c>
   1c19a: 13 87 87 00  	addi	a4, a5, 8
   1c19e: d1 bf        	j	0x1c172 <__multf3+0x3ec>
   1c1a0: 6d fe        	bnez	a2, 0x1c19a <__multf3+0x414>
   1c1a2: 13 57 44 03  	srli	a4, s0, 52
   1c1a6: 05 8b        	andi	a4, a4, 1
   1c1a8: 19 c7        	beqz	a4, 0x1c1b6 <__multf3+0x430>
   1c1aa: 7d 57        	addi	a4, zero, -1
   1c1ac: 52 17        	slli	a4, a4, 52
   1c1ae: 7d 17        	addi	a4, a4, -1
   1c1b0: 91 66        	lui	a3, 4
   1c1b2: 79 8c        	and	s0, s0, a4
   1c1b4: ae 96        	add	a3, a3, a1
   1c1b6: a1 65        	lui	a1, 8
   1c1b8: 13 87 e5 ff  	addi	a4, a1, -2
   1c1bc: 63 40 d7 04  	blt	a4, a3, 0x1c1fc <__multf3+0x476>
   1c1c0: 8d 83        	srli	a5, a5, 3
   1c1c2: 13 15 d4 03  	slli	a0, s0, 61
   1c1c6: 5d 8d        	or	a0, a0, a5
   1c1c8: 13 57 34 00  	srli	a4, s0, 3
   1c1cc: 1b 16 f6 00  	slliw	a2, a2, 15
   1c1d0: 93 15 07 01  	slli	a1, a4, 16
   1c1d4: d1 8e        	or	a3, a3, a2
   1c1d6: c2 16        	slli	a3, a3, 48
   1c1d8: c1 81        	srli	a1, a1, 16
   1c1da: d5 8d        	or	a1, a1, a3
   1c1dc: 99 c0        	beqz	s1, 0x1c1e2 <__multf3+0x45c>
   1c1de: 73 a0 14 00  	csrs	fflags, s1
   1c1e2: e6 60        	ld	ra, 88(sp)
   1c1e4: 46 64        	ld	s0, 80(sp)
   1c1e6: a6 64        	ld	s1, 72(sp)
   1c1e8: 06 69        	ld	s2, 64(sp)
   1c1ea: e2 79        	ld	s3, 56(sp)
   1c1ec: 42 7a        	ld	s4, 48(sp)
   1c1ee: a2 7a        	ld	s5, 40(sp)
   1c1f0: 02 7b        	ld	s6, 32(sp)
   1c1f2: e2 6b        	ld	s7, 24(sp)
   1c1f4: 42 6c        	ld	s8, 16(sp)
   1c1f6: a2 6c        	ld	s9, 8(sp)
   1c1f8: 25 61        	addi	sp, sp, 96
   1c1fa: 82 80        	ret
   1c1fc: 89 47        	addi	a5, zero, 2
   1c1fe: 63 82 f9 02  	beq	s3, a5, 0x1c222 <__multf3+0x49c>
   1c202: 8d 47        	addi	a5, zero, 3
   1c204: 63 87 f9 00  	beq	s3, a5, 0x1c212 <__multf3+0x48c>
   1c208: 63 86 09 00  	beqz	s3, 0x1c214 <__multf3+0x48e>
   1c20c: 7d 55        	addi	a0, zero, -1
   1c20e: ba 86        	mv	a3, a4
   1c210: 29 a0        	j	0x1c21a <__multf3+0x494>
   1c212: 6d fe        	bnez	a2, 0x1c20c <__multf3+0x486>
   1c214: 01 45        	mv	a0, zero
   1c216: 93 86 f5 ff  	addi	a3, a1, -1
   1c21a: 93 e4 54 00  	ori	s1, s1, 5
   1c21e: 2a 87        	mv	a4, a0
   1c220: 75 b7        	j	0x1c1cc <__multf3+0x446>
   1c222: 6d d6        	beqz	a2, 0x1c20c <__multf3+0x486>
   1c224: c5 bf        	j	0x1c214 <__multf3+0x48e>
   1c226: c5 e2        	bnez	a3, 0x1c2c6 <__multf3+0x540>
   1c228: 93 f6 77 00  	andi	a3, a5, 7
   1c22c: 22 87        	mv	a4, s0
   1c22e: 8d ce        	beqz	a3, 0x1c268 <__multf3+0x4e2>
   1c230: 89 46        	addi	a3, zero, 2
   1c232: 93 e4 14 00  	ori	s1, s1, 1
   1c236: 63 88 d9 02  	beq	s3, a3, 0x1c266 <__multf3+0x4e0>
   1c23a: 8d 46        	addi	a3, zero, 3
   1c23c: 63 81 d9 02  	beq	s3, a3, 0x1c25e <__multf3+0x4d8>
   1c240: 63 94 09 02  	bnez	s3, 0x1c268 <__multf3+0x4e2>
   1c244: 93 f6 f7 00  	andi	a3, a5, 15
   1c248: 91 45        	addi	a1, zero, 4
   1c24a: 63 8f b6 00  	beq	a3, a1, 0x1c268 <__multf3+0x4e2>
   1c24e: 13 b7 c7 ff  	sltiu	a4, a5, -4
   1c252: 13 47 17 00  	xori	a4, a4, 1
   1c256: 02 17        	slli	a4, a4, 32
   1c258: 01 93        	srli	a4, a4, 32
   1c25a: 22 97        	add	a4, a4, s0
   1c25c: 31 a0        	j	0x1c268 <__multf3+0x4e2>
   1c25e: 09 e6        	bnez	a2, 0x1c268 <__multf3+0x4e2>
   1c260: 13 b7 87 ff  	sltiu	a4, a5, -8
   1c264: fd b7        	j	0x1c252 <__multf3+0x4cc>
   1c266: 6d fe        	bnez	a2, 0x1c260 <__multf3+0x4da>
   1c268: 51 93        	srli	a4, a4, 52
   1c26a: 13 47 17 00  	xori	a4, a4, 1
   1c26e: 93 75 17 00  	andi	a1, a4, 1
   1c272: 05 47        	addi	a4, zero, 1
   1c274: 93 06 00 04  	addi	a3, zero, 64
   1c278: 99 9e        	subw	a3, a3, a4
   1c27a: 33 15 d4 00  	sll	a0, s0, a3
   1c27e: 33 d8 e7 00  	srl	a6, a5, a4
   1c282: b3 96 d7 00  	sll	a3, a5, a3
   1c286: 33 65 05 01  	or	a0, a0, a6
   1c28a: b3 36 d0 00  	snez	a3, a3
   1c28e: 55 8d        	or	a0, a0, a3
   1c290: 33 57 e4 00  	srl	a4, s0, a4
   1c294: 93 77 75 00  	andi	a5, a0, 7
   1c298: bd cf        	beqz	a5, 0x1c316 <__multf3+0x590>
   1c29a: 89 47        	addi	a5, zero, 2
   1c29c: 93 e4 14 00  	ori	s1, s1, 1
   1c2a0: 63 8a f9 06  	beq	s3, a5, 0x1c314 <__multf3+0x58e>
   1c2a4: 8d 47        	addi	a5, zero, 3
   1c2a6: 63 83 f9 06  	beq	s3, a5, 0x1c30c <__multf3+0x586>
   1c2aa: 63 96 09 06  	bnez	s3, 0x1c316 <__multf3+0x590>
   1c2ae: 93 77 f5 00  	andi	a5, a0, 15
   1c2b2: 91 46        	addi	a3, zero, 4
   1c2b4: 63 81 d7 06  	beq	a5, a3, 0x1c316 <__multf3+0x590>
   1c2b8: 93 07 45 00  	addi	a5, a0, 4
   1c2bc: b3 b6 a7 00  	sltu	a3, a5, a0
   1c2c0: 36 97        	add	a4, a4, a3
   1c2c2: 3e 85        	mv	a0, a5
   1c2c4: 89 a8        	j	0x1c316 <__multf3+0x590>
   1c2c6: 05 47        	addi	a4, zero, 1
   1c2c8: 15 8f        	sub	a4, a4, a3
   1c2ca: 93 06 40 07  	addi	a3, zero, 116
   1c2ce: 63 cf e6 06  	blt	a3, a4, 0x1c34c <__multf3+0x5c6>
   1c2d2: 93 06 f0 03  	addi	a3, zero, 63
   1c2d6: 9b 05 07 00  	sext.w	a1, a4
   1c2da: 63 c4 e6 00  	blt	a3, a4, 0x1c2e2 <__multf3+0x55c>
   1c2de: 85 45        	addi	a1, zero, 1
   1c2e0: 51 bf        	j	0x1c274 <__multf3+0x4ee>
   1c2e2: 1b 85 05 fc  	addiw	a0, a1, -64
   1c2e6: 13 08 00 04  	addi	a6, zero, 64
   1c2ea: 33 55 a4 00  	srl	a0, s0, a0
   1c2ee: 81 46        	mv	a3, zero
   1c2f0: 63 07 07 01  	beq	a4, a6, 0x1c2fe <__multf3+0x578>
   1c2f4: 93 06 00 08  	addi	a3, zero, 128
   1c2f8: 8d 9e        	subw	a3, a3, a1
   1c2fa: b3 16 d4 00  	sll	a3, s0, a3
   1c2fe: dd 8e        	or	a3, a3, a5
   1c300: b3 36 d0 00  	snez	a3, a3
   1c304: 55 8d        	or	a0, a0, a3
   1c306: 85 45        	addi	a1, zero, 1
   1c308: 01 47        	mv	a4, zero
   1c30a: 69 b7        	j	0x1c294 <__multf3+0x50e>
   1c30c: 09 e6        	bnez	a2, 0x1c316 <__multf3+0x590>
   1c30e: 93 07 85 00  	addi	a5, a0, 8
   1c312: 6d b7        	j	0x1c2bc <__multf3+0x536>
   1c314: 6d fe        	bnez	a2, 0x1c30e <__multf3+0x588>
   1c316: 93 57 37 03  	srli	a5, a4, 51
   1c31a: 85 8b        	andi	a5, a5, 1
   1c31c: 99 cb        	beqz	a5, 0x1c332 <__multf3+0x5ac>
   1c31e: 93 e4 14 00  	ori	s1, s1, 1
   1c322: 01 47        	mv	a4, zero
   1c324: 01 45        	mv	a0, zero
   1c326: 85 46        	addi	a3, zero, 1
   1c328: e3 82 05 ea  	beqz	a1, 0x1c1cc <__multf3+0x446>
   1c32c: 93 e4 24 00  	ori	s1, s1, 2
   1c330: 71 bd        	j	0x1c1cc <__multf3+0x446>
   1c332: 93 17 d7 03  	slli	a5, a4, 61
   1c336: 0d 81        	srli	a0, a0, 3
   1c338: 5d 8d        	or	a0, a0, a5
   1c33a: 0d 83        	srli	a4, a4, 3
   1c33c: 81 46        	mv	a3, zero
   1c33e: e3 87 05 e8  	beqz	a1, 0x1c1cc <__multf3+0x446>
   1c342: 93 f7 14 00  	andi	a5, s1, 1
   1c346: e3 83 07 e8  	beqz	a5, 0x1c1cc <__multf3+0x446>
   1c34a: cd b7        	j	0x1c32c <__multf3+0x5a6>
   1c34c: 33 e5 87 00  	or	a0, a5, s0
   1c350: 11 cd        	beqz	a0, 0x1c36c <__multf3+0x5e6>
   1c352: 89 47        	addi	a5, zero, 2
   1c354: 93 e4 14 00  	ori	s1, s1, 1
   1c358: 63 83 f9 02  	beq	s3, a5, 0x1c37e <__multf3+0x5f8>
   1c35c: 8d 47        	addi	a5, zero, 3
   1c35e: 63 8c f9 00  	beq	s3, a5, 0x1c376 <__multf3+0x5f0>
   1c362: 15 45        	addi	a0, zero, 5
   1c364: 63 83 09 00  	beqz	s3, 0x1c36a <__multf3+0x5e4>
   1c368: 05 45        	addi	a0, zero, 1
   1c36a: 0d 81        	srli	a0, a0, 3
   1c36c: 93 e4 24 00  	ori	s1, s1, 2
   1c370: 01 47        	mv	a4, zero
   1c372: 81 46        	mv	a3, zero
   1c374: a1 bd        	j	0x1c1cc <__multf3+0x446>
   1c376: 05 45        	addi	a0, zero, 1
   1c378: 6d fa        	bnez	a2, 0x1c36a <__multf3+0x5e4>
   1c37a: 25 45        	addi	a0, zero, 9
   1c37c: fd b7        	j	0x1c36a <__multf3+0x5e4>
   1c37e: 05 45        	addi	a0, zero, 1
   1c380: 6d d6        	beqz	a2, 0x1c36a <__multf3+0x5e4>
   1c382: e5 bf        	j	0x1c37a <__multf3+0x5f4>
   1c384: a1 66        	lui	a3, 8
   1c386: 3e 17        	slli	a4, a4, 47
   1c388: fd 16        	addi	a3, a3, -1
   1c38a: 01 46        	mv	a2, zero
   1c38c: c1 44        	addi	s1, zero, 16
   1c38e: 3d bd        	j	0x1c1cc <__multf3+0x446>
   1c390: a1 66        	lui	a3, 8
   1c392: 01 47        	mv	a4, zero
   1c394: 01 45        	mv	a0, zero
   1c396: fd 16        	addi	a3, a3, -1
   1c398: 15 bd        	j	0x1c1cc <__multf3+0x446>
   1c39a: 05 47        	addi	a4, zero, 1
   1c39c: a1 66        	lui	a3, 8
   1c39e: 3e 17        	slli	a4, a4, 47
   1c3a0: 01 45        	mv	a0, zero
   1c3a2: fd 16        	addi	a3, a3, -1
   1c3a4: 01 46        	mv	a2, zero
   1c3a6: 1d b5        	j	0x1c1cc <__multf3+0x446>

000000000001c3a8 <__subtf3>:
   1c3a8: 79 71        	addi	sp, sp, -48
   1c3aa: 06 f4        	sd	ra, 40(sp)
   1c3ac: 22 f0        	sd	s0, 32(sp)
   1c3ae: 26 ec        	sd	s1, 24(sp)
   1c3b0: 4a e8        	sd	s2, 16(sp)
   1c3b2: 4e e4        	sd	s3, 8(sp)
   1c3b4: 52 e0        	sd	s4, 0(sp)
   1c3b6: 73 29 20 00  	frrm	s2
   1c3ba: 7d 53        	addi	t1, zero, -1
   1c3bc: 13 53 03 01  	srli	t1, t1, 16
   1c3c0: 13 d4 05 03  	srli	s0, a1, 48
   1c3c4: a1 68        	lui	a7, 8
   1c3c6: 93 d4 f5 03  	srli	s1, a1, 63
   1c3ca: b3 f5 65 00  	and	a1, a1, t1
   1c3ce: 93 97 35 00  	slli	a5, a1, 3
   1c3d2: fd 18        	addi	a7, a7, -1
   1c3d4: 93 55 d5 03  	srli	a1, a0, 61
   1c3d8: 13 d7 06 03  	srli	a4, a3, 48
   1c3dc: 13 d8 f6 03  	srli	a6, a3, 63
   1c3e0: b3 f6 66 00  	and	a3, a3, t1
   1c3e4: dd 8d        	or	a1, a1, a5
   1c3e6: 33 77 17 01  	and	a4, a4, a7
   1c3ea: 93 97 36 00  	slli	a5, a3, 3
   1c3ee: 93 56 d6 03  	srli	a3, a2, 61
   1c3f2: 01 29        	sext.w	s2, s2
   1c3f4: 33 74 14 01  	and	s0, s0, a7
   1c3f8: 0e 05        	slli	a0, a0, 3
   1c3fa: dd 8e        	or	a3, a3, a5
   1c3fc: 0e 06        	slli	a2, a2, 3
   1c3fe: 63 18 17 05  	bne	a4, a7, 0x1c44e <__subtf3+0xa6>
   1c402: e1 77        	lui	a5, 1048568
   1c404: 85 27        	addiw	a5, a5, 1
   1c406: b3 e8 c6 00  	or	a7, a3, a2
   1c40a: a1 9f        	addw	a5, a5, s0
   1c40c: 63 87 08 02  	beqz	a7, 0x1c43a <__subtf3+0x92>
   1c410: 42 83        	mv	t1, a6
   1c412: 63 93 04 4b  	bne	s1, a6, 0x1c8b8 <__subtf3+0x510>
   1c416: 63 8d 07 24  	beqz	a5, 0x1c670 <__subtf3+0x2c8>
   1c41a: 63 12 04 4a  	bnez	s0, 0x1c8be <__subtf3+0x516>
   1c41e: 33 e8 a5 00  	or	a6, a1, a0
   1c422: 63 17 08 1e  	bnez	a6, 0x1c610 <__subtf3+0x268>
   1c426: a1 65        	lui	a1, 8
   1c428: fd 15        	addi	a1, a1, -1
   1c42a: 81 47        	mv	a5, zero
   1c42c: 63 14 b7 0a  	bne	a4, a1, 0x1c4d4 <__subtf3+0x12c>
   1c430: 33 e8 c6 00  	or	a6, a3, a2
   1c434: 63 05 08 2c  	beqz	a6, 0x1c6fe <__subtf3+0x356>
   1c438: 59 a1        	j	0x1c8be <__subtf3+0x516>
   1c43a: 13 43 18 00  	xori	t1, a6, 1
   1c43e: 63 97 64 48  	bne	s1, t1, 0x1c8cc <__subtf3+0x524>
   1c442: 63 87 07 22  	beqz	a5, 0x1c670 <__subtf3+0x2c8>
   1c446: 61 dc        	beqz	s0, 0x1c41e <__subtf3+0x76>
   1c448: 01 48        	mv	a6, zero
   1c44a: 81 45        	mv	a1, zero
   1c44c: b5 ab        	j	0x1c9c8 <__subtf3+0x620>
   1c44e: 13 43 18 00  	xori	t1, a6, 1
   1c452: bb 07 e4 40  	subw	a5, s0, a4
   1c456: 3e 88        	mv	a6, a5
   1c458: 63 99 64 38  	bne	s1, t1, 0x1c7ea <__subtf3+0x442>
   1c45c: 63 50 f0 16  	blez	a5, 0x1c5bc <__subtf3+0x214>
   1c460: 69 eb        	bnez	a4, 0x1c532 <__subtf3+0x18a>
   1c462: b3 e7 c6 00  	or	a5, a3, a2
   1c466: 91 e7        	bnez	a5, 0x1c472 <__subtf3+0xca>
   1c468: 63 09 14 05  	beq	s0, a7, 0x1c4ba <__subtf3+0x112>
   1c46c: ae 86        	mv	a3, a1
   1c46e: 2a 86        	mv	a2, a0
   1c470: a1 ac        	j	0x1c6c8 <__subtf3+0x320>
   1c472: 9b 07 f8 ff  	addiw	a5, a6, -1
   1c476: a1 e3        	bnez	a5, 0x1c4b6 <__subtf3+0x10e>
   1c478: 2a 96        	add	a2, a2, a0
   1c47a: ae 96        	add	a3, a3, a1
   1c47c: 33 35 a6 00  	sltu	a0, a2, a0
   1c480: aa 96        	add	a3, a3, a0
   1c482: 22 87        	mv	a4, s0
   1c484: 93 d7 36 03  	srli	a5, a3, 51
   1c488: 85 8b        	andi	a5, a5, 1
   1c48a: 63 87 07 22  	beqz	a5, 0x1c6b8 <__subtf3+0x310>
   1c48e: a1 67        	lui	a5, 8
   1c490: 05 07        	addi	a4, a4, 1
   1c492: 93 85 f7 ff  	addi	a1, a5, -1
   1c496: 63 03 b7 32  	beq	a4, a1, 0x1c7bc <__subtf3+0x414>
   1c49a: fd 57        	addi	a5, zero, -1
   1c49c: ce 17        	slli	a5, a5, 51
   1c49e: fd 17        	addi	a5, a5, -1
   1c4a0: f5 8f        	and	a5, a5, a3
   1c4a2: 93 56 16 00  	srli	a3, a2, 1
   1c4a6: 05 8a        	andi	a2, a2, 1
   1c4a8: d1 8e        	or	a3, a3, a2
   1c4aa: 13 96 f7 03  	slli	a2, a5, 63
   1c4ae: 55 8e        	or	a2, a2, a3
   1c4b0: 93 d6 17 00  	srli	a3, a5, 1
   1c4b4: 11 a4        	j	0x1c6b8 <__subtf3+0x310>
   1c4b6: 63 15 14 0d  	bne	s0, a7, 0x1c580 <__subtf3+0x1d8>
   1c4ba: 33 e8 a5 00  	or	a6, a1, a0
   1c4be: 63 02 08 7e  	beqz	a6, 0x1cca2 <__subtf3+0x8fa>
   1c4c2: 93 d7 25 03  	srli	a5, a1, 50
   1c4c6: 93 c7 17 00  	xori	a5, a5, 1
   1c4ca: 85 8b        	andi	a5, a5, 1
   1c4cc: 92 07        	slli	a5, a5, 4
   1c4ce: ae 86        	mv	a3, a1
   1c4d0: 2a 86        	mv	a2, a0
   1c4d2: 22 87        	mv	a4, s0
   1c4d4: 93 75 76 00  	andi	a1, a2, 7
   1c4d8: 63 90 05 78  	bnez	a1, 0x1cc58 <__subtf3+0x8b0>
   1c4dc: b6 85        	mv	a1, a3
   1c4de: 32 85        	mv	a0, a2
   1c4e0: 93 d6 35 03  	srli	a3, a1, 51
   1c4e4: 85 8a        	andi	a3, a3, 1
   1c4e6: 63 81 06 7c  	beqz	a3, 0x1cca8 <__subtf3+0x900>
   1c4ea: a1 66        	lui	a3, 8
   1c4ec: 05 07        	addi	a4, a4, 1
   1c4ee: 13 86 f6 ff  	addi	a2, a3, -1
   1c4f2: 63 00 c7 78  	beq	a4, a2, 0x1cc72 <__subtf3+0x8ca>
   1c4f6: fd 56        	addi	a3, zero, -1
   1c4f8: ce 16        	slli	a3, a3, 51
   1c4fa: fd 16        	addi	a3, a3, -1
   1c4fc: f5 8d        	and	a1, a1, a3
   1c4fe: 93 56 35 00  	srli	a3, a0, 3
   1c502: 13 95 d5 03  	slli	a0, a1, 61
   1c506: 55 8d        	or	a0, a0, a3
   1c508: 8d 81        	srli	a1, a1, 3
   1c50a: 46 17        	slli	a4, a4, 49
   1c50c: 9b 94 f4 00  	slliw	s1, s1, 15
   1c510: 45 93        	srli	a4, a4, 49
   1c512: c2 05        	slli	a1, a1, 16
   1c514: 45 8f        	or	a4, a4, s1
   1c516: 42 17        	slli	a4, a4, 48
   1c518: c1 81        	srli	a1, a1, 16
   1c51a: d9 8d        	or	a1, a1, a4
   1c51c: 99 c3        	beqz	a5, 0x1c522 <__subtf3+0x17a>
   1c51e: 73 a0 17 00  	csrs	fflags, a5
   1c522: a2 70        	ld	ra, 40(sp)
   1c524: 02 74        	ld	s0, 32(sp)
   1c526: e2 64        	ld	s1, 24(sp)
   1c528: 42 69        	ld	s2, 16(sp)
   1c52a: a2 69        	ld	s3, 8(sp)
   1c52c: 02 6a        	ld	s4, 0(sp)
   1c52e: 45 61        	addi	sp, sp, 48
   1c530: 82 80        	ret
   1c532: e3 04 14 f9  	beq	s0, a7, 0x1c4ba <__subtf3+0x112>
   1c536: 13 07 40 07  	addi	a4, zero, 116
   1c53a: 63 47 f7 04  	blt	a4, a5, 0x1c588 <__subtf3+0x1e0>
   1c53e: 05 47        	addi	a4, zero, 1
   1c540: 4e 17        	slli	a4, a4, 51
   1c542: d9 8e        	or	a3, a3, a4
   1c544: 13 07 f0 03  	addi	a4, zero, 63
   1c548: 63 43 f7 04  	blt	a4, a5, 0x1c58e <__subtf3+0x1e6>
   1c54c: 13 08 00 04  	addi	a6, zero, 64
   1c550: 3b 08 f8 40  	subw	a6, a6, a5
   1c554: 33 97 06 01  	sll	a4, a3, a6
   1c558: b3 58 f6 00  	srl	a7, a2, a5
   1c55c: 33 18 06 01  	sll	a6, a2, a6
   1c560: 33 67 17 01  	or	a4, a4, a7
   1c564: 33 38 00 01  	snez	a6, a6
   1c568: 33 67 07 01  	or	a4, a4, a6
   1c56c: b3 d7 f6 00  	srl	a5, a3, a5
   1c570: 33 06 a7 00  	add	a2, a4, a0
   1c574: ae 97        	add	a5, a5, a1
   1c576: 33 37 e6 00  	sltu	a4, a2, a4
   1c57a: b3 86 e7 00  	add	a3, a5, a4
   1c57e: 11 b7        	j	0x1c482 <__subtf3+0xda>
   1c580: 13 07 40 07  	addi	a4, zero, 116
   1c584: e3 50 f7 fc  	bge	a4, a5, 0x1c544 <__subtf3+0x19c>
   1c588: 81 47        	mv	a5, zero
   1c58a: 05 47        	addi	a4, zero, 1
   1c58c: d5 b7        	j	0x1c570 <__subtf3+0x1c8>
   1c58e: 1b 87 07 fc  	addiw	a4, a5, -64
   1c592: 93 08 00 04  	addi	a7, zero, 64
   1c596: 33 d7 e6 00  	srl	a4, a3, a4
   1c59a: 01 48        	mv	a6, zero
   1c59c: 63 88 17 01  	beq	a5, a7, 0x1c5ac <__subtf3+0x204>
   1c5a0: 13 08 00 08  	addi	a6, zero, 128
   1c5a4: 3b 08 f8 40  	subw	a6, a6, a5
   1c5a8: 33 98 06 01  	sll	a6, a3, a6
   1c5ac: 33 68 c8 00  	or	a6, a6, a2
   1c5b0: 33 38 00 01  	snez	a6, a6
   1c5b4: 33 67 07 01  	or	a4, a4, a6
   1c5b8: 81 47        	mv	a5, zero
   1c5ba: 5d bf        	j	0x1c570 <__subtf3+0x1c8>
   1c5bc: d5 cb        	beqz	a5, 0x1c670 <__subtf3+0x2c8>
   1c5be: e3 00 04 e6  	beqz	s0, 0x1c41e <__subtf3+0x76>
   1c5c2: 93 08 c0 f8  	addi	a7, zero, -116
   1c5c6: 63 cb 17 07  	blt	a5, a7, 0x1c63c <__subtf3+0x294>
   1c5ca: 05 48        	addi	a6, zero, 1
   1c5cc: 4e 18        	slli	a6, a6, 51
   1c5ce: bb 07 f0 40  	negw	a5, a5
   1c5d2: b3 e5 05 01  	or	a1, a1, a6
   1c5d6: 13 08 f0 03  	addi	a6, zero, 63
   1c5da: 63 44 f8 06  	blt	a6, a5, 0x1c642 <__subtf3+0x29a>
   1c5de: 93 08 00 04  	addi	a7, zero, 64
   1c5e2: bb 88 f8 40  	subw	a7, a7, a5
   1c5e6: 33 53 f5 00  	srl	t1, a0, a5
   1c5ea: 33 98 15 01  	sll	a6, a1, a7
   1c5ee: b3 18 15 01  	sll	a7, a0, a7
   1c5f2: 33 68 68 00  	or	a6, a6, t1
   1c5f6: b3 38 10 01  	snez	a7, a7
   1c5fa: 33 65 18 01  	or	a0, a6, a7
   1c5fe: b3 d7 f5 00  	srl	a5, a1, a5
   1c602: 2a 96        	add	a2, a2, a0
   1c604: b6 97        	add	a5, a5, a3
   1c606: 33 35 a6 00  	sltu	a0, a2, a0
   1c60a: b3 86 a7 00  	add	a3, a5, a0
   1c60e: 9d bd        	j	0x1c484 <__subtf3+0xdc>
   1c610: 93 c7 f7 ff  	not	a5, a5
   1c614: 99 e7        	bnez	a5, 0x1c622 <__subtf3+0x27a>
   1c616: 2a 96        	add	a2, a2, a0
   1c618: ae 96        	add	a3, a3, a1
   1c61a: 33 35 a6 00  	sltu	a0, a2, a0
   1c61e: aa 96        	add	a3, a3, a0
   1c620: 95 b5        	j	0x1c484 <__subtf3+0xdc>
   1c622: 21 68        	lui	a6, 8
   1c624: 7d 18        	addi	a6, a6, -1
   1c626: 63 17 07 01  	bne	a4, a6, 0x1c634 <__subtf3+0x28c>
   1c62a: 33 e8 c6 00  	or	a6, a3, a2
   1c62e: e3 0e 08 e0  	beqz	a6, 0x1c44a <__subtf3+0xa2>
   1c632: 71 a4        	j	0x1c8be <__subtf3+0x516>
   1c634: 13 08 40 07  	addi	a6, zero, 116
   1c638: e3 5f f8 f8  	bge	a6, a5, 0x1c5d6 <__subtf3+0x22e>
   1c63c: 81 47        	mv	a5, zero
   1c63e: 05 45        	addi	a0, zero, 1
   1c640: c9 b7        	j	0x1c602 <__subtf3+0x25a>
   1c642: 1b 88 07 fc  	addiw	a6, a5, -64
   1c646: 13 03 00 04  	addi	t1, zero, 64
   1c64a: 33 d8 05 01  	srl	a6, a1, a6
   1c64e: 81 48        	mv	a7, zero
   1c650: 63 88 67 00  	beq	a5, t1, 0x1c660 <__subtf3+0x2b8>
   1c654: 93 08 00 08  	addi	a7, zero, 128
   1c658: bb 88 f8 40  	subw	a7, a7, a5
   1c65c: b3 98 15 01  	sll	a7, a1, a7
   1c660: b3 e8 a8 00  	or	a7, a7, a0
   1c664: b3 38 10 01  	snez	a7, a7
   1c668: 33 65 18 01  	or	a0, a6, a7
   1c66c: 81 47        	mv	a5, zero
   1c66e: 51 bf        	j	0x1c602 <__subtf3+0x25a>
   1c670: a1 68        	lui	a7, 8
   1c672: 13 88 e8 ff  	addi	a6, a7, -2
   1c676: 93 07 14 00  	addi	a5, s0, 1
   1c67a: 33 f3 07 01  	and	t1, a5, a6
   1c67e: fd 18        	addi	a7, a7, -1
   1c680: 63 19 03 0c  	bnez	t1, 0x1c752 <__subtf3+0x3aa>
   1c684: 33 e8 a5 00  	or	a6, a1, a0
   1c688: 15 e8        	bnez	s0, 0x1c6bc <__subtf3+0x314>
   1c68a: 63 00 08 54  	beqz	a6, 0x1cbca <__subtf3+0x822>
   1c68e: b3 e7 c6 00  	or	a5, a3, a2
   1c692: 63 8b 07 46  	beqz	a5, 0x1cb08 <__subtf3+0x760>
   1c696: 2a 96        	add	a2, a2, a0
   1c698: b3 37 a6 00  	sltu	a5, a2, a0
   1c69c: ae 96        	add	a3, a3, a1
   1c69e: b3 85 f6 00  	add	a1, a3, a5
   1c6a2: 93 d7 35 03  	srli	a5, a1, 51
   1c6a6: 85 8b        	andi	a5, a5, 1
   1c6a8: 32 85        	mv	a0, a2
   1c6aa: 63 8f 07 44  	beqz	a5, 0x1cb08 <__subtf3+0x760>
   1c6ae: fd 56        	addi	a3, zero, -1
   1c6b0: ce 16        	slli	a3, a3, 51
   1c6b2: fd 16        	addi	a3, a3, -1
   1c6b4: ed 8e        	and	a3, a3, a1
   1c6b6: 05 47        	addi	a4, zero, 1
   1c6b8: 81 47        	mv	a5, zero
   1c6ba: 29 bd        	j	0x1c4d4 <__subtf3+0x12c>
   1c6bc: 63 18 14 03  	bne	s0, a7, 0x1c6ec <__subtf3+0x344>
   1c6c0: 63 16 08 00  	bnez	a6, 0x1c6cc <__subtf3+0x324>
   1c6c4: e3 03 87 f6  	beq	a4, s0, 0x1c62a <__subtf3+0x282>
   1c6c8: 22 87        	mv	a4, s0
   1c6ca: fd b7        	j	0x1c6b8 <__subtf3+0x310>
   1c6cc: 85 48        	addi	a7, zero, 1
   1c6ce: ca 18        	slli	a7, a7, 50
   1c6d0: b3 f7 15 01  	and	a5, a1, a7
   1c6d4: 33 e3 c6 00  	or	t1, a3, a2
   1c6d8: 63 8a 07 5c  	beqz	a5, 0x1ccac <__subtf3+0x904>
   1c6dc: 81 47        	mv	a5, zero
   1c6de: 63 13 87 06  	bne	a4, s0, 0x1c744 <__subtf3+0x39c>
   1c6e2: 63 10 03 02  	bnez	t1, 0x1c702 <__subtf3+0x35a>
   1c6e6: ae 86        	mv	a3, a1
   1c6e8: 2a 86        	mv	a2, a0
   1c6ea: ed b3        	j	0x1c4d4 <__subtf3+0x12c>
   1c6ec: 81 47        	mv	a5, zero
   1c6ee: 63 19 17 05  	bne	a4, a7, 0x1c740 <__subtf3+0x398>
   1c6f2: b3 e8 c6 00  	or	a7, a3, a2
   1c6f6: 63 90 08 5c  	bnez	a7, 0x1ccb6 <__subtf3+0x90e>
   1c6fa: e3 16 08 fe  	bnez	a6, 0x1c6e6 <__subtf3+0x33e>
   1c6fe: 81 45        	mv	a1, zero
   1c700: 19 a8        	j	0x1c716 <__subtf3+0x36e>
   1c702: b3 f8 16 01  	and	a7, a3, a7
   1c706: 63 84 08 04  	beqz	a7, 0x1c74e <__subtf3+0x3a6>
   1c70a: 85 45        	addi	a1, zero, 1
   1c70c: 21 67        	lui	a4, 8
   1c70e: 81 44        	mv	s1, zero
   1c710: ca 15        	slli	a1, a1, 50
   1c712: 01 48        	mv	a6, zero
   1c714: 7d 17        	addi	a4, a4, -1
   1c716: a1 66        	lui	a3, 8
   1c718: 13 95 d5 03  	slli	a0, a1, 61
   1c71c: 13 58 38 00  	srli	a6, a6, 3
   1c720: fd 16        	addi	a3, a3, -1
   1c722: 33 65 05 01  	or	a0, a0, a6
   1c726: 8d 81        	srli	a1, a1, 3
   1c728: e3 11 d7 de  	bne	a4, a3, 0x1c50a <__subtf3+0x162>
   1c72c: 4d 8d        	or	a0, a0, a1
   1c72e: 81 45        	mv	a1, zero
   1c730: e3 0d 05 dc  	beqz	a0, 0x1c50a <__subtf3+0x162>
   1c734: 85 45        	addi	a1, zero, 1
   1c736: be 15        	slli	a1, a1, 47
   1c738: 01 45        	mv	a0, zero
   1c73a: 81 44        	mv	s1, zero
   1c73c: f9 b3        	j	0x1c50a <__subtf3+0x162>
   1c73e: c1 47        	addi	a5, zero, 16
   1c740: 63 02 08 58  	beqz	a6, 0x1ccc4 <__subtf3+0x91c>
   1c744: d1 8e        	or	a3, a3, a2
   1c746: f1 f2        	bnez	a3, 0x1c70a <__subtf3+0x362>
   1c748: ae 86        	mv	a3, a1
   1c74a: 2a 86        	mv	a2, a0
   1c74c: a5 ab        	j	0x1ccc4 <__subtf3+0x91c>
   1c74e: c1 47        	addi	a5, zero, 16
   1c750: d5 bf        	j	0x1c744 <__subtf3+0x39c>
   1c752: 63 8e 17 01  	beq	a5, a7, 0x1c76e <__subtf3+0x3c6>
   1c756: 2a 96        	add	a2, a2, a0
   1c758: 33 35 a6 00  	sltu	a0, a2, a0
   1c75c: ae 96        	add	a3, a3, a1
   1c75e: aa 96        	add	a3, a3, a0
   1c760: 13 97 f6 03  	slli	a4, a3, 63
   1c764: 05 82        	srli	a2, a2, 1
   1c766: 59 8e        	or	a2, a2, a4
   1c768: 85 82        	srli	a3, a3, 1
   1c76a: 3e 87        	mv	a4, a5
   1c76c: b1 b7        	j	0x1c6b8 <__subtf3+0x310>
   1c76e: 63 05 09 52  	beqz	s2, 0x1cc98 <__subtf3+0x8f0>
   1c772: 0d 47        	addi	a4, zero, 3
   1c774: 63 1a e9 02  	bne	s2, a4, 0x1c7a8 <__subtf3+0x400>
   1c778: 63 80 04 52  	beqz	s1, 0x1cc98 <__subtf3+0x8f0>
   1c77c: 81 46        	mv	a3, zero
   1c77e: fd 55        	addi	a1, zero, -1
   1c780: 7d 55        	addi	a0, zero, -1
   1c782: 42 87        	mv	a4, a6
   1c784: 95 47        	addi	a5, zero, 5
   1c786: 0d 46        	addi	a2, zero, 3
   1c788: 63 0e c9 4c  	beq	s2, a2, 0x1cc64 <__subtf3+0x8bc>
   1c78c: 63 1d 09 4c  	bnez	s2, 0x1cc66 <__subtf3+0x8be>
   1c790: 13 76 f5 00  	andi	a2, a0, 15
   1c794: 11 48        	addi	a6, zero, 4
   1c796: 63 08 06 4d  	beq	a2, a6, 0x1cc66 <__subtf3+0x8be>
   1c79a: 13 06 45 00  	addi	a2, a0, 4
   1c79e: 33 38 a6 00  	sltu	a6, a2, a0
   1c7a2: c2 95        	add	a1, a1, a6
   1c7a4: 32 85        	mv	a0, a2
   1c7a6: c1 a1        	j	0x1cc66 <__subtf3+0x8be>
   1c7a8: 09 47        	addi	a4, zero, 2
   1c7aa: 63 14 e9 00  	bne	s2, a4, 0x1c7b2 <__subtf3+0x40a>
   1c7ae: 63 95 04 4e  	bnez	s1, 0x1cc98 <__subtf3+0x8f0>
   1c7b2: fd 55        	addi	a1, zero, -1
   1c7b4: 7d 55        	addi	a0, zero, -1
   1c7b6: 42 87        	mv	a4, a6
   1c7b8: 95 47        	addi	a5, zero, 5
   1c7ba: 1d b3        	j	0x1c4e0 <__subtf3+0x138>
   1c7bc: 63 0f 09 4c  	beqz	s2, 0x1cc9a <__subtf3+0x8f2>
   1c7c0: 8d 46        	addi	a3, zero, 3
   1c7c2: 63 1a d9 00  	bne	s2, a3, 0x1c7d6 <__subtf3+0x42e>
   1c7c6: 63 8a 04 4c  	beqz	s1, 0x1cc9a <__subtf3+0x8f2>
   1c7ca: 81 46        	mv	a3, zero
   1c7cc: fd 55        	addi	a1, zero, -1
   1c7ce: 7d 55        	addi	a0, zero, -1
   1c7d0: 13 87 e7 ff  	addi	a4, a5, -2
   1c7d4: 45 bf        	j	0x1c784 <__subtf3+0x3dc>
   1c7d6: 89 46        	addi	a3, zero, 2
   1c7d8: 63 14 d9 00  	bne	s2, a3, 0x1c7e0 <__subtf3+0x438>
   1c7dc: 63 9f 04 4a  	bnez	s1, 0x1cc9a <__subtf3+0x8f2>
   1c7e0: fd 55        	addi	a1, zero, -1
   1c7e2: 7d 55        	addi	a0, zero, -1
   1c7e4: 13 87 e7 ff  	addi	a4, a5, -2
   1c7e8: c1 bf        	j	0x1c7b8 <__subtf3+0x410>
   1c7ea: 63 54 f0 10  	blez	a5, 0x1c8f2 <__subtf3+0x54a>
   1c7ee: 29 e7        	bnez	a4, 0x1c838 <__subtf3+0x490>
   1c7f0: b3 e7 c6 00  	or	a5, a3, a2
   1c7f4: e3 8a 07 c6  	beqz	a5, 0x1c468 <__subtf3+0xc0>
   1c7f8: 9b 07 f8 ff  	addiw	a5, a6, -1
   1c7fc: 8d e7        	bnez	a5, 0x1c826 <__subtf3+0x47e>
   1c7fe: 33 06 c5 40  	sub	a2, a0, a2
   1c802: b3 86 d5 40  	sub	a3, a1, a3
   1c806: 33 35 c5 00  	sltu	a0, a0, a2
   1c80a: 89 8e        	sub	a3, a3, a0
   1c80c: 22 87        	mv	a4, s0
   1c80e: 93 d7 36 03  	srli	a5, a3, 51
   1c812: 85 8b        	andi	a5, a5, 1
   1c814: e3 82 07 ea  	beqz	a5, 0x1c6b8 <__subtf3+0x310>
   1c818: 93 99 d6 00  	slli	s3, a3, 13
   1c81c: 93 d9 d9 00  	srli	s3, s3, 13
   1c820: 32 8a        	mv	s4, a2
   1c822: 3a 84        	mv	s0, a4
   1c824: 69 ac        	j	0x1cabe <__subtf3+0x716>
   1c826: e3 0a 14 c9  	beq	s0, a7, 0x1c4ba <__subtf3+0x112>
   1c82a: 13 07 40 07  	addi	a4, zero, 116
   1c82e: 63 5e f7 00  	bge	a4, a5, 0x1c84a <__subtf3+0x4a2>
   1c832: 81 47        	mv	a5, zero
   1c834: 05 47        	addi	a4, zero, 1
   1c836: 81 a0        	j	0x1c876 <__subtf3+0x4ce>
   1c838: e3 01 14 c9  	beq	s0, a7, 0x1c4ba <__subtf3+0x112>
   1c83c: 13 07 40 07  	addi	a4, zero, 116
   1c840: e3 49 f7 fe  	blt	a4, a5, 0x1c832 <__subtf3+0x48a>
   1c844: 05 47        	addi	a4, zero, 1
   1c846: 4e 17        	slli	a4, a4, 51
   1c848: d9 8e        	or	a3, a3, a4
   1c84a: 13 07 f0 03  	addi	a4, zero, 63
   1c84e: 63 4e f7 02  	blt	a4, a5, 0x1c88a <__subtf3+0x4e2>
   1c852: 13 08 00 04  	addi	a6, zero, 64
   1c856: 3b 08 f8 40  	subw	a6, a6, a5
   1c85a: 33 97 06 01  	sll	a4, a3, a6
   1c85e: b3 58 f6 00  	srl	a7, a2, a5
   1c862: 33 18 06 01  	sll	a6, a2, a6
   1c866: 33 67 17 01  	or	a4, a4, a7
   1c86a: 33 38 00 01  	snez	a6, a6
   1c86e: 33 67 07 01  	or	a4, a4, a6
   1c872: b3 d7 f6 00  	srl	a5, a3, a5
   1c876: 33 07 e5 40  	sub	a4, a0, a4
   1c87a: b3 87 f5 40  	sub	a5, a1, a5
   1c87e: b3 36 e5 00  	sltu	a3, a0, a4
   1c882: 3a 86        	mv	a2, a4
   1c884: b3 86 d7 40  	sub	a3, a5, a3
   1c888: 51 b7        	j	0x1c80c <__subtf3+0x464>
   1c88a: 1b 87 07 fc  	addiw	a4, a5, -64
   1c88e: 93 08 00 04  	addi	a7, zero, 64
   1c892: 33 d7 e6 00  	srl	a4, a3, a4
   1c896: 01 48        	mv	a6, zero
   1c898: 63 88 17 01  	beq	a5, a7, 0x1c8a8 <__subtf3+0x500>
   1c89c: 13 08 00 08  	addi	a6, zero, 128
   1c8a0: 3b 08 f8 40  	subw	a6, a6, a5
   1c8a4: 33 98 06 01  	sll	a6, a3, a6
   1c8a8: 33 68 c8 00  	or	a6, a6, a2
   1c8ac: 33 38 00 01  	snez	a6, a6
   1c8b0: 33 67 07 01  	or	a4, a4, a6
   1c8b4: 81 47        	mv	a5, zero
   1c8b6: c1 b7        	j	0x1c876 <__subtf3+0x4ce>
   1c8b8: e5 c3        	beqz	a5, 0x1c998 <__subtf3+0x5f0>
   1c8ba: c2 84        	mv	s1, a6
   1c8bc: 01 cc        	beqz	s0, 0x1c8d4 <__subtf3+0x52c>
   1c8be: 93 d7 26 03  	srli	a5, a3, 50
   1c8c2: 93 c7 17 00  	xori	a5, a5, 1
   1c8c6: 85 8b        	andi	a5, a5, 1
   1c8c8: 92 07        	slli	a5, a5, 4
   1c8ca: 29 b1        	j	0x1c4d4 <__subtf3+0x12c>
   1c8cc: f1 c7        	beqz	a5, 0x1c998 <__subtf3+0x5f0>
   1c8ce: 9a 84        	mv	s1, t1
   1c8d0: e3 1c 04 b6  	bnez	s0, 0x1c448 <__subtf3+0xa0>
   1c8d4: 33 e8 a5 00  	or	a6, a1, a0
   1c8d8: e3 07 08 b4  	beqz	a6, 0x1c426 <__subtf3+0x7e>
   1c8dc: 93 c7 f7 ff  	not	a5, a5
   1c8e0: 89 ef        	bnez	a5, 0x1c8fa <__subtf3+0x552>
   1c8e2: 33 05 a6 40  	sub	a0, a2, a0
   1c8e6: b3 37 a6 00  	sltu	a5, a2, a0
   1c8ea: 8d 8e        	sub	a3, a3, a1
   1c8ec: 2a 86        	mv	a2, a0
   1c8ee: 9d 8e        	sub	a3, a3, a5
   1c8f0: 39 bf        	j	0x1c80e <__subtf3+0x466>
   1c8f2: dd c3        	beqz	a5, 0x1c998 <__subtf3+0x5f0>
   1c8f4: 11 ec        	bnez	s0, 0x1c910 <__subtf3+0x568>
   1c8f6: 9a 84        	mv	s1, t1
   1c8f8: f1 bf        	j	0x1c8d4 <__subtf3+0x52c>
   1c8fa: 21 68        	lui	a6, 8
   1c8fc: 7d 18        	addi	a6, a6, -1
   1c8fe: e3 06 07 d3  	beq	a4, a6, 0x1c62a <__subtf3+0x282>
   1c902: 13 08 40 07  	addi	a6, zero, 116
   1c906: 63 50 f8 02  	bge	a6, a5, 0x1c926 <__subtf3+0x57e>
   1c90a: 81 47        	mv	a5, zero
   1c90c: 05 45        	addi	a0, zero, 1
   1c90e: 91 a0        	j	0x1c952 <__subtf3+0x5aa>
   1c910: 93 08 c0 f8  	addi	a7, zero, -116
   1c914: 63 c0 17 09  	blt	a5, a7, 0x1c994 <__subtf3+0x5ec>
   1c918: 05 48        	addi	a6, zero, 1
   1c91a: 4e 18        	slli	a6, a6, 51
   1c91c: bb 07 f0 40  	negw	a5, a5
   1c920: b3 e5 05 01  	or	a1, a1, a6
   1c924: 9a 84        	mv	s1, t1
   1c926: 13 08 f0 03  	addi	a6, zero, 63
   1c92a: 63 4e f8 02  	blt	a6, a5, 0x1c966 <__subtf3+0x5be>
   1c92e: 93 08 00 04  	addi	a7, zero, 64
   1c932: bb 88 f8 40  	subw	a7, a7, a5
   1c936: 33 53 f5 00  	srl	t1, a0, a5
   1c93a: 33 98 15 01  	sll	a6, a1, a7
   1c93e: b3 18 15 01  	sll	a7, a0, a7
   1c942: 33 68 68 00  	or	a6, a6, t1
   1c946: b3 38 10 01  	snez	a7, a7
   1c94a: 33 65 18 01  	or	a0, a6, a7
   1c94e: b3 d7 f5 00  	srl	a5, a1, a5
   1c952: 33 05 a6 40  	sub	a0, a2, a0
   1c956: b3 35 a6 00  	sltu	a1, a2, a0
   1c95a: b3 87 f6 40  	sub	a5, a3, a5
   1c95e: 2a 86        	mv	a2, a0
   1c960: b3 86 b7 40  	sub	a3, a5, a1
   1c964: 6d b5        	j	0x1c80e <__subtf3+0x466>
   1c966: 1b 88 07 fc  	addiw	a6, a5, -64
   1c96a: 13 03 00 04  	addi	t1, zero, 64
   1c96e: 33 d8 05 01  	srl	a6, a1, a6
   1c972: 81 48        	mv	a7, zero
   1c974: 63 88 67 00  	beq	a5, t1, 0x1c984 <__subtf3+0x5dc>
   1c978: 93 08 00 08  	addi	a7, zero, 128
   1c97c: bb 88 f8 40  	subw	a7, a7, a5
   1c980: b3 98 15 01  	sll	a7, a1, a7
   1c984: b3 e8 a8 00  	or	a7, a7, a0
   1c988: b3 38 10 01  	snez	a7, a7
   1c98c: 33 65 18 01  	or	a0, a6, a7
   1c990: 81 47        	mv	a5, zero
   1c992: c1 b7        	j	0x1c952 <__subtf3+0x5aa>
   1c994: 9a 84        	mv	s1, t1
   1c996: 95 bf        	j	0x1c90a <__subtf3+0x562>
   1c998: 21 6e        	lui	t3, 8
   1c99a: 93 07 14 00  	addi	a5, s0, 1
   1c99e: 13 08 ee ff  	addi	a6, t3, -2
   1c9a2: b3 f7 07 01  	and	a5, a5, a6
   1c9a6: f5 e7        	bnez	a5, 0x1ca92 <__subtf3+0x6ea>
   1c9a8: 33 e8 c6 00  	or	a6, a3, a2
   1c9ac: b3 e8 a5 00  	or	a7, a1, a0
   1c9b0: 29 ec        	bnez	s0, 0x1ca0a <__subtf3+0x662>
   1c9b2: 63 9d 08 00  	bnez	a7, 0x1c9cc <__subtf3+0x624>
   1c9b6: 63 1d 08 20  	bnez	a6, 0x1cbd0 <__subtf3+0x828>
   1c9ba: 93 04 e9 ff  	addi	s1, s2, -2
   1c9be: 93 b4 14 00  	seqz	s1, s1
   1c9c2: 81 45        	mv	a1, zero
   1c9c4: 01 48        	mv	a6, zero
   1c9c6: 01 47        	mv	a4, zero
   1c9c8: 81 47        	mv	a5, zero
   1c9ca: b1 b3        	j	0x1c716 <__subtf3+0x36e>
   1c9cc: 63 0e 08 12  	beqz	a6, 0x1cb08 <__subtf3+0x760>
   1c9d0: 33 07 c5 40  	sub	a4, a0, a2
   1c9d4: 33 38 e5 00  	sltu	a6, a0, a4
   1c9d8: b3 87 d5 40  	sub	a5, a1, a3
   1c9dc: b3 87 07 41  	sub	a5, a5, a6
   1c9e0: 13 d8 37 03  	srli	a6, a5, 51
   1c9e4: 13 78 18 00  	andi	a6, a6, 1
   1c9e8: 63 0b 08 00  	beqz	a6, 0x1c9fe <__subtf3+0x656>
   1c9ec: 33 05 a6 40  	sub	a0, a2, a0
   1c9f0: 8d 8e        	sub	a3, a3, a1
   1c9f2: 33 36 a6 00  	sltu	a2, a2, a0
   1c9f6: b3 85 c6 40  	sub	a1, a3, a2
   1c9fa: 9a 84        	mv	s1, t1
   1c9fc: 31 a2        	j	0x1cb08 <__subtf3+0x760>
   1c9fe: b3 66 f7 00  	or	a3, a4, a5
   1ca02: c5 de        	beqz	a3, 0x1c9ba <__subtf3+0x612>
   1ca04: be 85        	mv	a1, a5
   1ca06: 3a 85        	mv	a0, a4
   1ca08: 01 a2        	j	0x1cb08 <__subtf3+0x760>
   1ca0a: 7d 1e        	addi	t3, t3, -1
   1ca0c: 63 10 c4 05  	bne	s0, t3, 0x1ca4c <__subtf3+0x6a4>
   1ca10: 63 9e 08 00  	bnez	a7, 0x1ca2c <__subtf3+0x684>
   1ca14: 81 47        	mv	a5, zero
   1ca16: 63 13 87 06  	bne	a4, s0, 0x1ca7c <__subtf3+0x6d4>
   1ca1a: 85 45        	addi	a1, zero, 1
   1ca1c: ca 15        	slli	a1, a1, 50
   1ca1e: 81 44        	mv	s1, zero
   1ca20: 63 00 08 04  	beqz	a6, 0x1ca60 <__subtf3+0x6b8>
   1ca24: f5 8d        	and	a1, a1, a3
   1ca26: a5 c5        	beqz	a1, 0x1ca8e <__subtf3+0x6e6>
   1ca28: 9a 84        	mv	s1, t1
   1ca2a: 6d b4        	j	0x1c4d4 <__subtf3+0x12c>
   1ca2c: 05 46        	addi	a2, zero, 1
   1ca2e: 4a 16        	slli	a2, a2, 50
   1ca30: b3 f7 c5 00  	and	a5, a1, a2
   1ca34: 63 8b 07 28  	beqz	a5, 0x1ccca <__subtf3+0x922>
   1ca38: 81 47        	mv	a5, zero
   1ca3a: 63 10 87 2a  	bne	a4, s0, 0x1ccda <__subtf3+0x932>
   1ca3e: e3 05 08 d0  	beqz	a6, 0x1c748 <__subtf3+0x3a0>
   1ca42: 75 8e        	and	a2, a2, a3
   1ca44: e3 13 06 cc  	bnez	a2, 0x1c70a <__subtf3+0x362>
   1ca48: c1 47        	addi	a5, zero, 16
   1ca4a: 41 ac        	j	0x1ccda <__subtf3+0x932>
   1ca4c: 81 47        	mv	a5, zero
   1ca4e: 63 15 c7 03  	bne	a4, t3, 0x1ca78 <__subtf3+0x6d0>
   1ca52: 63 19 08 00  	bnez	a6, 0x1ca64 <__subtf3+0x6bc>
   1ca56: e3 99 08 ce  	bnez	a7, 0x1c748 <__subtf3+0x3a0>
   1ca5a: 85 45        	addi	a1, zero, 1
   1ca5c: 81 44        	mv	s1, zero
   1ca5e: ca 15        	slli	a1, a1, 50
   1ca60: c1 47        	addi	a5, zero, 16
   1ca62: 55 b9        	j	0x1c716 <__subtf3+0x36e>
   1ca64: 13 de 26 03  	srli	t3, a3, 50
   1ca68: 13 7e 1e 00  	andi	t3, t3, 1
   1ca6c: 63 05 0e 00  	beqz	t3, 0x1ca76 <__subtf3+0x6ce>
   1ca70: e3 9d 08 c8  	bnez	a7, 0x1c70a <__subtf3+0x362>
   1ca74: 55 bf        	j	0x1ca28 <__subtf3+0x680>
   1ca76: c1 47        	addi	a5, zero, 16
   1ca78: 63 91 08 26  	bnez	a7, 0x1ccda <__subtf3+0x932>
   1ca7c: 9a 84        	mv	s1, t1
   1ca7e: 63 13 08 24  	bnez	a6, 0x1ccc4 <__subtf3+0x91c>
   1ca82: 85 45        	addi	a1, zero, 1
   1ca84: 21 67        	lui	a4, 8
   1ca86: 81 44        	mv	s1, zero
   1ca88: ca 15        	slli	a1, a1, 50
   1ca8a: 7d 17        	addi	a4, a4, -1
   1ca8c: d1 bf        	j	0x1ca60 <__subtf3+0x6b8>
   1ca8e: c1 47        	addi	a5, zero, 16
   1ca90: f5 b7        	j	0x1ca7c <__subtf3+0x6d4>
   1ca92: b3 07 c5 40  	sub	a5, a0, a2
   1ca96: 33 37 f5 00  	sltu	a4, a0, a5
   1ca9a: b3 89 d5 40  	sub	s3, a1, a3
   1ca9e: b3 89 e9 40  	sub	s3, s3, a4
   1caa2: 13 d7 39 03  	srli	a4, s3, 51
   1caa6: 05 8b        	andi	a4, a4, 1
   1caa8: 3e 8a        	mv	s4, a5
   1caaa: 45 cf        	beqz	a4, 0x1cb62 <__subtf3+0x7ba>
   1caac: 33 0a a6 40  	sub	s4, a2, a0
   1cab0: b3 89 b6 40  	sub	s3, a3, a1
   1cab4: 33 36 46 01  	sltu	a2, a2, s4
   1cab8: b3 89 c9 40  	sub	s3, s3, a2
   1cabc: 9a 84        	mv	s1, t1
   1cabe: 63 86 09 0a  	beqz	s3, 0x1cb6a <__subtf3+0x7c2>
   1cac2: 4e 85        	mv	a0, s3
   1cac4: ef 00 00 64  	jal	0x1d104 <__clzdi2>
   1cac8: 1b 07 45 ff  	addiw	a4, a0, -12
   1cacc: 93 06 00 04  	addi	a3, zero, 64
   1cad0: 99 9e        	subw	a3, a3, a4
   1cad2: b3 99 e9 00  	sll	s3, s3, a4
   1cad6: b3 56 da 00  	srl	a3, s4, a3
   1cada: b3 e6 36 01  	or	a3, a3, s3
   1cade: b3 19 ea 00  	sll	s3, s4, a4
   1cae2: 63 4c 87 0c  	blt	a4, s0, 0x1cbba <__subtf3+0x812>
   1cae6: 01 9f        	subw	a4, a4, s0
   1cae8: 05 27        	addiw	a4, a4, 1
   1caea: 93 07 00 04  	addi	a5, zero, 64
   1caee: 99 9f        	subw	a5, a5, a4
   1caf0: 33 95 f6 00  	sll	a0, a3, a5
   1caf4: 33 d6 e9 00  	srl	a2, s3, a4
   1caf8: b3 97 f9 00  	sll	a5, s3, a5
   1cafc: 51 8d        	or	a0, a0, a2
   1cafe: b3 37 f0 00  	snez	a5, a5
   1cb02: 5d 8d        	or	a0, a0, a5
   1cb04: b3 d5 e6 00  	srl	a1, a3, a4
   1cb08: b3 67 b5 00  	or	a5, a0, a1
   1cb0c: e3 8b 07 ea  	beqz	a5, 0x1c9c2 <__subtf3+0x61a>
   1cb10: 93 57 f5 03  	srli	a5, a0, 63
   1cb14: 13 97 15 00  	slli	a4, a1, 1
   1cb18: 3e 97        	add	a4, a4, a5
   1cb1a: 93 17 15 00  	slli	a5, a0, 1
   1cb1e: 13 f6 77 00  	andi	a2, a5, 7
   1cb22: 93 76 75 00  	andi	a3, a0, 7
   1cb26: 63 0f 06 0e  	beqz	a2, 0x1cc24 <__subtf3+0x87c>
   1cb2a: 09 46        	addi	a2, zero, 2
   1cb2c: 63 07 c9 0c  	beq	s2, a2, 0x1cbfa <__subtf3+0x852>
   1cb30: 05 46        	addi	a2, zero, 1
   1cb32: 0d 48        	addi	a6, zero, 3
   1cb34: 52 16        	slli	a2, a2, 52
   1cb36: 63 00 09 0b  	beq	s2, a6, 0x1cbd6 <__subtf3+0x82e>
   1cb3a: 63 1f 09 08  	bnez	s2, 0x1cbd8 <__subtf3+0x830>
   1cb3e: 13 f8 f7 00  	andi	a6, a5, 15
   1cb42: 91 48        	addi	a7, zero, 4
   1cb44: 63 0a 18 09  	beq	a6, a7, 0x1cbd8 <__subtf3+0x830>
   1cb48: 93 b7 c7 ff  	sltiu	a5, a5, -4
   1cb4c: 93 c7 17 00  	xori	a5, a5, 1
   1cb50: 82 17        	slli	a5, a5, 32
   1cb52: 81 93        	srli	a5, a5, 32
   1cb54: ba 97        	add	a5, a5, a4
   1cb56: f1 8f        	and	a5, a5, a2
   1cb58: e5 e3        	bnez	a5, 0x1cc38 <__subtf3+0x890>
   1cb5a: f9 c2        	beqz	a3, 0x1cc20 <__subtf3+0x878>
   1cb5c: 85 46        	addi	a3, zero, 1
   1cb5e: 01 47        	mv	a4, zero
   1cb60: 01 a2        	j	0x1cc60 <__subtf3+0x8b8>
   1cb62: b3 e7 37 01  	or	a5, a5, s3
   1cb66: a1 ff        	bnez	a5, 0x1cabe <__subtf3+0x716>
   1cb68: 89 bd        	j	0x1c9ba <__subtf3+0x612>
   1cb6a: 52 85        	mv	a0, s4
   1cb6c: ef 00 80 59  	jal	0x1d104 <__clzdi2>
   1cb70: 1b 07 45 03  	addiw	a4, a0, 52
   1cb74: 93 05 f0 03  	addi	a1, zero, 63
   1cb78: e3 da e5 f4  	bge	a1, a4, 0x1cacc <__subtf3+0x724>
   1cb7c: 9b 06 45 ff  	addiw	a3, a0, -12
   1cb80: b3 16 da 00  	sll	a3, s4, a3
   1cb84: 63 4b 87 02  	blt	a4, s0, 0x1cbba <__subtf3+0x812>
   1cb88: bb 07 87 40  	subw	a5, a4, s0
   1cb8c: 1b 87 17 00  	addiw	a4, a5, 1
   1cb90: e3 dd e5 f4  	bge	a1, a4, 0x1caea <__subtf3+0x742>
   1cb94: 9b 87 17 fc  	addiw	a5, a5, -63
   1cb98: 93 05 00 04  	addi	a1, zero, 64
   1cb9c: b3 d7 f6 00  	srl	a5, a3, a5
   1cba0: 01 45        	mv	a0, zero
   1cba2: 63 07 b7 00  	beq	a4, a1, 0x1cbb0 <__subtf3+0x808>
   1cba6: 13 05 00 08  	addi	a0, zero, 128
   1cbaa: 19 9d        	subw	a0, a0, a4
   1cbac: 33 95 a6 00  	sll	a0, a3, a0
   1cbb0: 33 35 a0 00  	snez	a0, a0
   1cbb4: 5d 8d        	or	a0, a0, a5
   1cbb6: 81 45        	mv	a1, zero
   1cbb8: 81 bf        	j	0x1cb08 <__subtf3+0x760>
   1cbba: fd 57        	addi	a5, zero, -1
   1cbbc: ce 17        	slli	a5, a5, 51
   1cbbe: fd 17        	addi	a5, a5, -1
   1cbc0: 33 07 e4 40  	sub	a4, s0, a4
   1cbc4: fd 8e        	and	a3, a3, a5
   1cbc6: 4e 86        	mv	a2, s3
   1cbc8: c5 bc        	j	0x1c6b8 <__subtf3+0x310>
   1cbca: b6 85        	mv	a1, a3
   1cbcc: 32 85        	mv	a0, a2
   1cbce: 2d bf        	j	0x1cb08 <__subtf3+0x760>
   1cbd0: b6 85        	mv	a1, a3
   1cbd2: 32 85        	mv	a0, a2
   1cbd4: 1d b5        	j	0x1c9fa <__subtf3+0x652>
   1cbd6: 89 c4        	beqz	s1, 0x1cbe0 <__subtf3+0x838>
   1cbd8: 71 8f        	and	a4, a4, a2
   1cbda: 39 ef        	bnez	a4, 0x1cc38 <__subtf3+0x890>
   1cbdc: c1 f2        	bnez	a3, 0x1cb5c <__subtf3+0x7b4>
   1cbde: 21 a8        	j	0x1cbf6 <__subtf3+0x84e>
   1cbe0: 93 b7 87 ff  	sltiu	a5, a5, -8
   1cbe4: 93 c7 17 00  	xori	a5, a5, 1
   1cbe8: 82 17        	slli	a5, a5, 32
   1cbea: 81 93        	srli	a5, a5, 32
   1cbec: ba 97        	add	a5, a5, a4
   1cbee: f1 8f        	and	a5, a5, a2
   1cbf0: a1 e7        	bnez	a5, 0x1cc38 <__subtf3+0x890>
   1cbf2: 01 47        	mv	a4, zero
   1cbf4: ad e6        	bnez	a3, 0x1cc5e <__subtf3+0x8b6>
   1cbf6: 85 47        	addi	a5, zero, 1
   1cbf8: 31 a0        	j	0x1cc04 <__subtf3+0x85c>
   1cbfa: 81 e8        	bnez	s1, 0x1cc0a <__subtf3+0x862>
   1cbfc: 51 93        	srli	a4, a4, 52
   1cbfe: 05 8b        	andi	a4, a4, 1
   1cc00: 85 47        	addi	a5, zero, 1
   1cc02: 05 ef        	bnez	a4, 0x1cc3a <__subtf3+0x892>
   1cc04: 93 e7 27 00  	ori	a5, a5, 2
   1cc08: e1 b8        	j	0x1c4e0 <__subtf3+0x138>
   1cc0a: 93 b7 87 ff  	sltiu	a5, a5, -8
   1cc0e: 93 c7 17 00  	xori	a5, a5, 1
   1cc12: 82 17        	slli	a5, a5, 32
   1cc14: 81 93        	srli	a5, a5, 32
   1cc16: ba 97        	add	a5, a5, a4
   1cc18: d1 93        	srli	a5, a5, 52
   1cc1a: 85 8b        	andi	a5, a5, 1
   1cc1c: 91 ef        	bnez	a5, 0x1cc38 <__subtf3+0x890>
   1cc1e: b1 e6        	bnez	a3, 0x1cc6a <__subtf3+0x8c2>
   1cc20: 01 47        	mv	a4, zero
   1cc22: d1 bf        	j	0x1cbf6 <__subtf3+0x84e>
   1cc24: 51 93        	srli	a4, a4, 52
   1cc26: 05 8b        	andi	a4, a4, 1
   1cc28: 15 e7        	bnez	a4, 0x1cc54 <__subtf3+0x8ac>
   1cc2a: 81 47        	mv	a5, zero
   1cc2c: e3 8a 06 8a  	beqz	a3, 0x1c4e0 <__subtf3+0x138>
   1cc30: 85 46        	addi	a3, zero, 1
   1cc32: 01 47        	mv	a4, zero
   1cc34: 81 47        	mv	a5, zero
   1cc36: 31 a0        	j	0x1cc42 <__subtf3+0x89a>
   1cc38: 85 47        	addi	a5, zero, 1
   1cc3a: 01 47        	mv	a4, zero
   1cc3c: e3 82 06 8a  	beqz	a3, 0x1c4e0 <__subtf3+0x138>
   1cc40: 81 46        	mv	a3, zero
   1cc42: 09 46        	addi	a2, zero, 2
   1cc44: 93 e7 17 00  	ori	a5, a5, 1
   1cc48: e3 1f c9 b2  	bne	s2, a2, 0x1c786 <__subtf3+0x3de>
   1cc4c: 89 cc        	beqz	s1, 0x1cc66 <__subtf3+0x8be>
   1cc4e: 13 06 85 00  	addi	a2, a0, 8
   1cc52: b1 b6        	j	0x1c79e <__subtf3+0x3f6>
   1cc54: 81 47        	mv	a5, zero
   1cc56: d5 b7        	j	0x1cc3a <__subtf3+0x892>
   1cc58: b6 85        	mv	a1, a3
   1cc5a: 32 85        	mv	a0, a2
   1cc5c: d5 b7        	j	0x1cc40 <__subtf3+0x898>
   1cc5e: 85 46        	addi	a3, zero, 1
   1cc60: 85 47        	addi	a5, zero, 1
   1cc62: 15 b6        	j	0x1c786 <__subtf3+0x3de>
   1cc64: ed d4        	beqz	s1, 0x1cc4e <__subtf3+0x8a6>
   1cc66: d9 fe        	bnez	a3, 0x1cc04 <__subtf3+0x85c>
   1cc68: a5 b8        	j	0x1c4e0 <__subtf3+0x138>
   1cc6a: 85 46        	addi	a3, zero, 1
   1cc6c: 01 47        	mv	a4, zero
   1cc6e: 85 47        	addi	a5, zero, 1
   1cc70: f1 bf        	j	0x1cc4c <__subtf3+0x8a4>
   1cc72: 01 48        	mv	a6, zero
   1cc74: 63 06 09 00  	beqz	s2, 0x1cc80 <__subtf3+0x8d8>
   1cc78: 0d 46        	addi	a2, zero, 3
   1cc7a: 63 17 c9 00  	bne	s2, a2, 0x1cc88 <__subtf3+0x8e0>
   1cc7e: 89 e8        	bnez	s1, 0x1cc90 <__subtf3+0x8e8>
   1cc80: 93 e7 57 00  	ori	a5, a5, 5
   1cc84: c2 85        	mv	a1, a6
   1cc86: 41 bc        	j	0x1c716 <__subtf3+0x36e>
   1cc88: 09 46        	addi	a2, zero, 2
   1cc8a: 63 13 c9 00  	bne	s2, a2, 0x1cc90 <__subtf3+0x8e8>
   1cc8e: ed f8        	bnez	s1, 0x1cc80 <__subtf3+0x8d8>
   1cc90: 7d 58        	addi	a6, zero, -1
   1cc92: 13 87 e6 ff  	addi	a4, a3, -2
   1cc96: ed b7        	j	0x1cc80 <__subtf3+0x8d8>
   1cc98: 3e 87        	mv	a4, a5
   1cc9a: 81 45        	mv	a1, zero
   1cc9c: 01 48        	mv	a6, zero
   1cc9e: 95 47        	addi	a5, zero, 5
   1cca0: 9d bc        	j	0x1c716 <__subtf3+0x36e>
   1cca2: 81 45        	mv	a1, zero
   1cca4: 22 87        	mv	a4, s0
   1cca6: 0d b3        	j	0x1c9c8 <__subtf3+0x620>
   1cca8: 2a 88        	mv	a6, a0
   1ccaa: b5 b4        	j	0x1c716 <__subtf3+0x36e>
   1ccac: c1 47        	addi	a5, zero, 16
   1ccae: e3 1b 87 a8  	bne	a4, s0, 0x1c744 <__subtf3+0x39c>
   1ccb2: e3 0a 03 a2  	beqz	t1, 0x1c6e6 <__subtf3+0x33e>
   1ccb6: 13 d7 26 03  	srli	a4, a3, 50
   1ccba: 05 8b        	andi	a4, a4, 1
   1ccbc: e3 01 07 a8  	beqz	a4, 0x1c73e <__subtf3+0x396>
   1ccc0: e3 15 08 a4  	bnez	a6, 0x1c70a <__subtf3+0x362>
   1ccc4: 21 67        	lui	a4, 8
   1ccc6: 7d 17        	addi	a4, a4, -1
   1ccc8: 31 b0        	j	0x1c4d4 <__subtf3+0x12c>
   1ccca: c1 47        	addi	a5, zero, 16
   1cccc: 63 17 87 00  	bne	a4, s0, 0x1ccda <__subtf3+0x932>
   1ccd0: e3 0c 08 a6  	beqz	a6, 0x1c748 <__subtf3+0x3a0>
   1ccd4: 75 8e        	and	a2, a2, a3
   1ccd6: e3 1a 06 a2  	bnez	a2, 0x1c70a <__subtf3+0x362>
   1ccda: e3 18 08 a2  	bnez	a6, 0x1c70a <__subtf3+0x362>
   1ccde: ad b4        	j	0x1c748 <__subtf3+0x3a0>

000000000001cce0 <__fixtfsi>:
   1cce0: 2a 86        	mv	a2, a0
   1cce2: f3 27 20 00  	frrm	a5
   1cce6: 11 68        	lui	a6, 4
   1cce8: 93 96 15 00  	slli	a3, a1, 1
   1ccec: 93 97 05 01  	slli	a5, a1, 16
   1ccf0: c5 92        	srli	a3, a3, 49
   1ccf2: 13 07 e8 ff  	addi	a4, a6, -2
   1ccf6: c1 83        	srli	a5, a5, 16
   1ccf8: fd 91        	srli	a1, a1, 63
   1ccfa: 63 47 d7 00  	blt	a4, a3, 0x1cd08 <__fixtfsi+0x28>
   1ccfe: bd e2        	bnez	a3, 0x1cd64 <__fixtfsi+0x84>
   1cd00: c9 8f        	or	a5, a5, a0
   1cd02: 01 45        	mv	a0, zero
   1cd04: b1 ef        	bnez	a5, 0x1cd60 <__fixtfsi+0x80>
   1cd06: 82 80        	ret
   1cd08: 13 07 d8 01  	addi	a4, a6, 29
   1cd0c: 63 54 d7 02  	bge	a4, a3, 0x1cd34 <__fixtfsi+0x54>
   1cd10: 37 07 00 80  	lui	a4, 524288
   1cd14: 13 47 f7 ff  	not	a4, a4
   1cd18: 3b 05 b7 00  	addw	a0, a4, a1
   1cd1c: b1 c5        	beqz	a1, 0x1cd68 <__fixtfsi+0x88>
   1cd1e: 79 08        	addi	a6, a6, 30
   1cd20: 41 47        	addi	a4, zero, 16
   1cd22: 63 94 06 05  	bne	a3, a6, 0x1cd6a <__fixtfsi+0x8a>
   1cd26: 93 d6 17 01  	srli	a3, a5, 17
   1cd2a: a1 e2        	bnez	a3, 0x1cd6a <__fixtfsi+0x8a>
   1cd2c: be 17        	slli	a5, a5, 47
   1cd2e: d1 8f        	or	a5, a5, a2
   1cd30: 85 eb        	bnez	a5, 0x1cd60 <__fixtfsi+0x80>
   1cd32: 82 80        	ret
   1cd34: 05 47        	addi	a4, zero, 1
   1cd36: 42 17        	slli	a4, a4, 48
   1cd38: d9 8f        	or	a5, a5, a4
   1cd3a: 11 66        	lui	a2, 4
   1cd3c: 71 77        	lui	a4, 1048572
   1cd3e: 45 27        	addiw	a4, a4, 17
   1cd40: 1b 06 f6 02  	addiw	a2, a2, 47
   1cd44: 35 9f        	addw	a4, a4, a3
   1cd46: bb 06 d6 40  	subw	a3, a2, a3
   1cd4a: 33 97 e7 00  	sll	a4, a5, a4
   1cd4e: b3 d7 d7 00  	srl	a5, a5, a3
   1cd52: 49 8f        	or	a4, a4, a0
   1cd54: 1b 85 07 00  	sext.w	a0, a5
   1cd58: 99 c1        	beqz	a1, 0x1cd5e <__fixtfsi+0x7e>
   1cd5a: 3b 05 a0 40  	negw	a0, a0
   1cd5e: 01 cb        	beqz	a4, 0x1cd6e <__fixtfsi+0x8e>
   1cd60: 05 47        	addi	a4, zero, 1
   1cd62: 21 a0        	j	0x1cd6a <__fixtfsi+0x8a>
   1cd64: 01 45        	mv	a0, zero
   1cd66: ed bf        	j	0x1cd60 <__fixtfsi+0x80>
   1cd68: 41 47        	addi	a4, zero, 16
   1cd6a: 73 20 17 00  	csrs	fflags, a4
   1cd6e: 82 80        	ret

000000000001cd70 <__floatsitf>:
   1cd70: 01 11        	addi	sp, sp, -32
   1cd72: 06 ec        	sd	ra, 24(sp)
   1cd74: 22 e8        	sd	s0, 16(sp)
   1cd76: 26 e4        	sd	s1, 8(sp)
   1cd78: 39 c1        	beqz	a0, 0x1cdbe <__floatsitf+0x4e>
   1cd7a: 9b 07 05 00  	sext.w	a5, a0
   1cd7e: 93 54 f5 03  	srli	s1, a0, 63
   1cd82: 63 54 05 00  	bgez	a0, 0x1cd8a <__floatsitf+0x1a>
   1cd86: bb 07 f0 40  	negw	a5, a5
   1cd8a: 13 94 07 02  	slli	s0, a5, 32
   1cd8e: 01 90        	srli	s0, s0, 32
   1cd90: 22 85        	mv	a0, s0
   1cd92: ef 00 20 37  	jal	0x1d104 <__clzdi2>
   1cd96: 91 65        	lui	a1, 4
   1cd98: 9b 85 e5 03  	addiw	a1, a1, 62
   1cd9c: 89 9d        	subw	a1, a1, a0
   1cd9e: 45 35        	addiw	a0, a0, -15
   1cda0: b3 17 a4 00  	sll	a5, s0, a0
   1cda4: e2 60        	ld	ra, 24(sp)
   1cda6: 42 64        	ld	s0, 16(sp)
   1cda8: 9b 94 f4 00  	slliw	s1, s1, 15
   1cdac: c5 8d        	or	a1, a1, s1
   1cdae: c2 07        	slli	a5, a5, 16
   1cdb0: c2 15        	slli	a1, a1, 48
   1cdb2: c1 83        	srli	a5, a5, 16
   1cdb4: a2 64        	ld	s1, 8(sp)
   1cdb6: 01 45        	mv	a0, zero
   1cdb8: dd 8d        	or	a1, a1, a5
   1cdba: 05 61        	addi	sp, sp, 32
   1cdbc: 82 80        	ret
   1cdbe: 81 47        	mv	a5, zero
   1cdc0: 81 45        	mv	a1, zero
   1cdc2: 81 44        	mv	s1, zero
   1cdc4: c5 b7        	j	0x1cda4 <__floatsitf+0x34>

000000000001cdc6 <__extenddftf2>:
   1cdc6: 01 11        	addi	sp, sp, -32
   1cdc8: d3 07 05 e2  	fmv.x.d	a5, fa0
   1cdcc: 06 ec        	sd	ra, 24(sp)
   1cdce: 22 e8        	sd	s0, 16(sp)
   1cdd0: 26 e4        	sd	s1, 8(sp)
   1cdd2: 73 27 20 00  	frrm	a4
   1cdd6: 13 d5 47 03  	srli	a0, a5, 52
   1cdda: 13 75 f5 7f  	andi	a0, a0, 2047
   1cdde: 13 94 c7 00  	slli	s0, a5, 12
   1cde2: 93 d4 f7 03  	srli	s1, a5, 63
   1cde6: 93 07 15 00  	addi	a5, a0, 1
   1cdea: 93 f7 e7 7f  	andi	a5, a5, 2046
   1cdee: 31 80        	srli	s0, s0, 12
   1cdf0: 91 cb        	beqz	a5, 0x1ce04 <__extenddftf2+0x3e>
   1cdf2: 91 67        	lui	a5, 4
   1cdf4: 93 87 07 c0  	addi	a5, a5, -1024
   1cdf8: 3e 95        	add	a0, a0, a5
   1cdfa: 93 57 44 00  	srli	a5, s0, 4
   1cdfe: 72 14        	slli	s0, s0, 60
   1ce00: 01 47        	mv	a4, zero
   1ce02: b9 a8        	j	0x1ce60 <__extenddftf2+0x9a>
   1ce04: 1d ed        	bnez	a0, 0x1ce42 <__extenddftf2+0x7c>
   1ce06: 81 47        	mv	a5, zero
   1ce08: 65 dc        	beqz	s0, 0x1ce00 <__extenddftf2+0x3a>
   1ce0a: 22 85        	mv	a0, s0
   1ce0c: ef 00 80 2f  	jal	0x1d104 <__clzdi2>
   1ce10: 1b 07 05 00  	sext.w	a4, a0
   1ce14: b9 47        	addi	a5, zero, 14
   1ce16: 63 c0 e7 02  	blt	a5, a4, 0x1ce36 <__extenddftf2+0x70>
   1ce1a: bd 47        	addi	a5, zero, 15
   1ce1c: 89 9f        	subw	a5, a5, a0
   1ce1e: 1b 07 15 03  	addiw	a4, a0, 49
   1ce22: b3 57 f4 00  	srl	a5, s0, a5
   1ce26: 33 14 e4 00  	sll	s0, s0, a4
   1ce2a: 11 67        	lui	a4, 4
   1ce2c: 1b 07 c7 c0  	addiw	a4, a4, -1012
   1ce30: 3b 05 a7 40  	subw	a0, a4, a0
   1ce34: f1 b7        	j	0x1ce00 <__extenddftf2+0x3a>
   1ce36: 9b 07 15 ff  	addiw	a5, a0, -15
   1ce3a: b3 17 f4 00  	sll	a5, s0, a5
   1ce3e: 01 44        	mv	s0, zero
   1ce40: ed b7        	j	0x1ce2a <__extenddftf2+0x64>
   1ce42: 21 c0        	beqz	s0, 0x1ce82 <__extenddftf2+0xbc>
   1ce44: 85 46        	addi	a3, zero, 1
   1ce46: 13 97 36 03  	slli	a4, a3, 51
   1ce4a: 33 37 e4 00  	sltu	a4, s0, a4
   1ce4e: 93 57 44 00  	srli	a5, s0, 4
   1ce52: be 16        	slli	a3, a3, 47
   1ce54: 21 65        	lui	a0, 8
   1ce56: 1b 17 47 00  	slliw	a4, a4, 4
   1ce5a: d5 8f        	or	a5, a5, a3
   1ce5c: 72 14        	slli	s0, s0, 60
   1ce5e: 7d 15        	addi	a0, a0, -1
   1ce60: 93 95 07 01  	slli	a1, a5, 16
   1ce64: 9b 97 f4 00  	slliw	a5, s1, 15
   1ce68: 5d 8d        	or	a0, a0, a5
   1ce6a: 42 15        	slli	a0, a0, 48
   1ce6c: c1 81        	srli	a1, a1, 16
   1ce6e: c9 8d        	or	a1, a1, a0
   1ce70: 19 c3        	beqz	a4, 0x1ce76 <__extenddftf2+0xb0>
   1ce72: 73 60 18 00  	csrsi	fflags, 16
   1ce76: e2 60        	ld	ra, 24(sp)
   1ce78: 22 85        	mv	a0, s0
   1ce7a: 42 64        	ld	s0, 16(sp)
   1ce7c: a2 64        	ld	s1, 8(sp)
   1ce7e: 05 61        	addi	sp, sp, 32
   1ce80: 82 80        	ret
   1ce82: 21 65        	lui	a0, 8
   1ce84: 81 47        	mv	a5, zero
   1ce86: 7d 15        	addi	a0, a0, -1
   1ce88: a5 bf        	j	0x1ce00 <__extenddftf2+0x3a>

000000000001ce8a <__trunctfdf2>:
   1ce8a: 73 28 20 00  	frrm	a6
   1ce8e: a1 68        	lui	a7, 8
   1ce90: 13 d7 05 03  	srli	a4, a1, 48
   1ce94: 13 8e f8 ff  	addi	t3, a7, -1
   1ce98: fd 57        	addi	a5, zero, -1
   1ce9a: 93 d6 07 01  	srli	a3, a5, 16
   1ce9e: 33 77 c7 01  	and	a4, a4, t3
   1cea2: 13 d6 f5 03  	srli	a2, a1, 63
   1cea6: 13 03 17 00  	addi	t1, a4, 1
   1ceaa: f5 8d        	and	a1, a1, a3
   1ceac: f9 18        	addi	a7, a7, -2
   1ceae: 8e 05        	slli	a1, a1, 3
   1ceb0: 93 56 d5 03  	srli	a3, a0, 61
   1ceb4: b3 78 13 01  	and	a7, t1, a7
   1ceb8: cd 8e        	or	a3, a3, a1
   1ceba: 01 28        	sext.w	a6, a6
   1cebc: 93 15 35 00  	slli	a1, a0, 3
   1cec0: 63 81 08 10  	beqz	a7, 0x1cfc2 <__trunctfdf2+0x138>
   1cec4: f1 78        	lui	a7, 1048572
   1cec6: 93 88 08 40  	addi	a7, a7, 1024
   1ceca: 46 97        	add	a4, a4, a7
   1cecc: 93 08 e0 7f  	addi	a7, zero, 2046
   1ced0: 63 d4 e8 02  	bge	a7, a4, 0x1cef8 <__trunctfdf2+0x6e>
   1ced4: 63 05 08 20  	beqz	a6, 0x1d0de <__trunctfdf2+0x254>
   1ced8: 0d 47        	addi	a4, zero, 3
   1ceda: 63 18 e8 00  	bne	a6, a4, 0x1ceea <__trunctfdf2+0x60>
   1cede: 63 0f 06 20  	beqz	a2, 0x1d0fc <__trunctfdf2+0x272>
   1cee2: 13 07 e0 7f  	addi	a4, zero, 2046
   1cee6: 15 45        	addi	a0, zero, 5
   1cee8: 59 a2        	j	0x1d06e <__trunctfdf2+0x1e4>
   1ceea: 09 47        	addi	a4, zero, 2
   1ceec: 63 14 e8 00  	bne	a6, a4, 0x1cef4 <__trunctfdf2+0x6a>
   1cef0: 63 18 06 20  	bnez	a2, 0x1d100 <__trunctfdf2+0x276>
   1cef4: fd 57        	addi	a5, zero, -1
   1cef6: f5 b7        	j	0x1cee2 <__trunctfdf2+0x58>
   1cef8: 63 47 e0 0a  	bgtz	a4, 0x1cfa6 <__trunctfdf2+0x11c>
   1cefc: 93 07 c0 fc  	addi	a5, zero, -52
   1cf00: 63 46 f7 0c  	blt	a4, a5, 0x1cfcc <__trunctfdf2+0x142>
   1cf04: 85 47        	addi	a5, zero, 1
   1cf06: ce 17        	slli	a5, a5, 51
   1cf08: 93 08 d0 03  	addi	a7, zero, 61
   1cf0c: dd 8e        	or	a3, a3, a5
   1cf0e: b3 88 e8 40  	sub	a7, a7, a4
   1cf12: 93 07 f0 03  	addi	a5, zero, 63
   1cf16: 01 27        	sext.w	a4, a4
   1cf18: 63 c5 17 07  	blt	a5, a7, 0x1cf82 <__trunctfdf2+0xf8>
   1cf1c: 93 07 d0 03  	addi	a5, zero, 61
   1cf20: 1b 05 37 00  	addiw	a0, a4, 3
   1cf24: 99 9f        	subw	a5, a5, a4
   1cf26: b3 d7 f5 00  	srl	a5, a1, a5
   1cf2a: b3 95 a5 00  	sll	a1, a1, a0
   1cf2e: b3 35 b0 00  	snez	a1, a1
   1cf32: cd 8f        	or	a5, a5, a1
   1cf34: b3 96 a6 00  	sll	a3, a3, a0
   1cf38: d5 8f        	or	a5, a5, a3
   1cf3a: 13 97 17 00  	slli	a4, a5, 1
   1cf3e: 93 76 77 00  	andi	a3, a4, 7
   1cf42: d9 e6        	bnez	a3, 0x1cfd0 <__trunctfdf2+0x146>
   1cf44: 93 f5 77 00  	andi	a1, a5, 7
   1cf48: 01 47        	mv	a4, zero
   1cf4a: 85 46        	addi	a3, zero, 1
   1cf4c: 01 45        	mv	a0, zero
   1cf4e: 85 c1        	beqz	a1, 0x1cf6e <__trunctfdf2+0xe4>
   1cf50: 89 45        	addi	a1, zero, 2
   1cf52: 63 0f b8 12  	beq	a6, a1, 0x1d090 <__trunctfdf2+0x206>
   1cf56: 8d 45        	addi	a1, zero, 3
   1cf58: 63 08 b8 10  	beq	a6, a1, 0x1d068 <__trunctfdf2+0x1de>
   1cf5c: 63 1d 08 12  	bnez	a6, 0x1d096 <__trunctfdf2+0x20c>
   1cf60: 93 f5 f7 00  	andi	a1, a5, 15
   1cf64: 11 45        	addi	a0, zero, 4
   1cf66: 63 9b a5 0e  	bne	a1, a0, 0x1d05c <__trunctfdf2+0x1d2>
   1cf6a: 05 45        	addi	a0, zero, 1
   1cf6c: e9 e6        	bnez	a3, 0x1d036 <__trunctfdf2+0x1ac>
   1cf6e: 93 06 f0 7f  	addi	a3, zero, 2047
   1cf72: 8d 83        	srli	a5, a5, 3
   1cf74: 63 1b d7 08  	bne	a4, a3, 0x1d00a <__trunctfdf2+0x180>
   1cf78: c9 cb        	beqz	a5, 0x1d00a <__trunctfdf2+0x180>
   1cf7a: 85 47        	addi	a5, zero, 1
   1cf7c: ce 17        	slli	a5, a5, 51
   1cf7e: 01 46        	mv	a2, zero
   1cf80: 69 a0        	j	0x1d00a <__trunctfdf2+0x180>
   1cf82: 75 55        	addi	a0, zero, -3
   1cf84: 19 9d        	subw	a0, a0, a4
   1cf86: 13 03 00 04  	addi	t1, zero, 64
   1cf8a: 33 d5 a6 00  	srl	a0, a3, a0
   1cf8e: 81 47        	mv	a5, zero
   1cf90: 63 86 68 00  	beq	a7, t1, 0x1cf9c <__trunctfdf2+0x112>
   1cf94: 1b 07 37 04  	addiw	a4, a4, 67
   1cf98: b3 97 e6 00  	sll	a5, a3, a4
   1cf9c: cd 8f        	or	a5, a5, a1
   1cf9e: b3 37 f0 00  	snez	a5, a5
   1cfa2: c9 8f        	or	a5, a5, a0
   1cfa4: 59 bf        	j	0x1cf3a <__trunctfdf2+0xb0>
   1cfa6: 1e 05        	slli	a0, a0, 7
   1cfa8: 33 35 a0 00  	snez	a0, a0
   1cfac: f1 91        	srli	a1, a1, 60
   1cfae: 4d 8d        	or	a0, a0, a1
   1cfb0: 93 97 46 00  	slli	a5, a3, 4
   1cfb4: c9 8f        	or	a5, a5, a0
   1cfb6: 1d 89        	andi	a0, a0, 7
   1cfb8: 81 46        	mv	a3, zero
   1cfba: 59 f9        	bnez	a0, 0x1cf50 <__trunctfdf2+0xc6>
   1cfbc: 8d 83        	srli	a5, a5, 3
   1cfbe: 01 45        	mv	a0, zero
   1cfc0: a9 a0        	j	0x1d00a <__trunctfdf2+0x180>
   1cfc2: d5 8d        	or	a1, a1, a3
   1cfc4: 05 e7        	bnez	a4, 0x1cfec <__trunctfdf2+0x162>
   1cfc6: 93 17 f6 03  	slli	a5, a2, 63
   1cfca: b1 c9        	beqz	a1, 0x1d01e <__trunctfdf2+0x194>
   1cfcc: 85 47        	addi	a5, zero, 1
   1cfce: 09 47        	addi	a4, zero, 2
   1cfd0: 89 46        	addi	a3, zero, 2
   1cfd2: 13 f5 77 00  	andi	a0, a5, 7
   1cfd6: 63 01 d8 08  	beq	a6, a3, 0x1d058 <__trunctfdf2+0x1ce>
   1cfda: 8d 46        	addi	a3, zero, 3
   1cfdc: 63 0f d8 04  	beq	a6, a3, 0x1d03a <__trunctfdf2+0x1b0>
   1cfe0: 63 02 08 04  	beqz	a6, 0x1d024 <__trunctfdf2+0x19a>
   1cfe4: 01 47        	mv	a4, zero
   1cfe6: 85 46        	addi	a3, zero, 1
   1cfe8: 25 f5        	bnez	a0, 0x1cf50 <__trunctfdf2+0xc6>
   1cfea: b1 a0        	j	0x1d036 <__trunctfdf2+0x1ac>
   1cfec: 63 81 05 10  	beqz	a1, 0x1d0ee <__trunctfdf2+0x264>
   1cff0: 01 45        	mv	a0, zero
   1cff2: 63 17 c7 01  	bne	a4, t3, 0x1d000 <__trunctfdf2+0x176>
   1cff6: c9 92        	srli	a3, a3, 50
   1cff8: 93 c6 16 00  	xori	a3, a3, 1
   1cffc: 13 95 46 00  	slli	a0, a3, 4
   1d000: 85 47        	addi	a5, zero, 1
   1d002: 01 46        	mv	a2, zero
   1d004: 13 07 f0 7f  	addi	a4, zero, 2047
   1d008: ce 17        	slli	a5, a5, 51
   1d00a: 13 77 f7 7f  	andi	a4, a4, 2047
   1d00e: b2 07        	slli	a5, a5, 12
   1d010: b1 83        	srli	a5, a5, 12
   1d012: 52 17        	slli	a4, a4, 52
   1d014: 5d 8f        	or	a4, a4, a5
   1d016: 93 17 f6 03  	slli	a5, a2, 63
   1d01a: d9 8f        	or	a5, a5, a4
   1d01c: 5d e5        	bnez	a0, 0x1d0ca <__trunctfdf2+0x240>
   1d01e: 53 85 07 f2  	fmv.d.x	fa0, a5
   1d022: 82 80        	ret
   1d024: 93 75 f7 00  	andi	a1, a4, 15
   1d028: 91 46        	addi	a3, zero, 4
   1d02a: 11 07        	addi	a4, a4, 4
   1d02c: 63 99 d5 00  	bne	a1, a3, 0x1d03e <__trunctfdf2+0x1b4>
   1d030: 85 46        	addi	a3, zero, 1
   1d032: 01 47        	mv	a4, zero
   1d034: 0d f1        	bnez	a0, 0x1cf56 <__trunctfdf2+0xcc>
   1d036: 0d 45        	addi	a0, zero, 3
   1d038: 1d bf        	j	0x1cf6e <__trunctfdf2+0xe4>
   1d03a: 4d f6        	bnez	a2, 0x1cfe4 <__trunctfdf2+0x15a>
   1d03c: 21 07        	addi	a4, a4, 8
   1d03e: 85 46        	addi	a3, zero, 1
   1d040: e2 16        	slli	a3, a3, 56
   1d042: b3 75 d7 00  	and	a1, a4, a3
   1d046: 93 b6 15 00  	seqz	a3, a1
   1d04a: 01 47        	mv	a4, zero
   1d04c: 11 f1        	bnez	a0, 0x1cf50 <__trunctfdf2+0xc6>
   1d04e: 0d 45        	addi	a0, zero, 3
   1d050: 99 dd        	beqz	a1, 0x1cf6e <__trunctfdf2+0xe4>
   1d052: 8d 83        	srli	a5, a5, 3
   1d054: 05 45        	addi	a0, zero, 1
   1d056: 55 bf        	j	0x1d00a <__trunctfdf2+0x180>
   1d058: 75 f2        	bnez	a2, 0x1d03c <__trunctfdf2+0x1b2>
   1d05a: 69 b7        	j	0x1cfe4 <__trunctfdf2+0x15a>
   1d05c: 91 07        	addi	a5, a5, 4
   1d05e: 05 45        	addi	a0, zero, 1
   1d060: 99 c6        	beqz	a3, 0x1d06e <__trunctfdf2+0x1e4>
   1d062: 13 e5 26 00  	ori	a0, a3, 2
   1d066: 21 a0        	j	0x1d06e <__trunctfdf2+0x1e4>
   1d068: 0d c6        	beqz	a2, 0x1d092 <__trunctfdf2+0x208>
   1d06a: 05 45        	addi	a0, zero, 1
   1d06c: 85 ea        	bnez	a3, 0x1d09c <__trunctfdf2+0x212>
   1d06e: 93 d6 77 03  	srli	a3, a5, 55
   1d072: 85 8a        	andi	a3, a3, 1
   1d074: e3 8d 06 ee  	beqz	a3, 0x1cf6e <__trunctfdf2+0xe4>
   1d078: 05 07        	addi	a4, a4, 1
   1d07a: 93 06 f0 7f  	addi	a3, zero, 2047
   1d07e: 63 01 d7 02  	beq	a4, a3, 0x1d0a0 <__trunctfdf2+0x216>
   1d082: 93 06 f0 1f  	addi	a3, zero, 511
   1d086: d2 16        	slli	a3, a3, 52
   1d088: 8d 83        	srli	a5, a5, 3
   1d08a: fd 16        	addi	a3, a3, -1
   1d08c: f5 8f        	and	a5, a5, a3
   1d08e: b5 bf        	j	0x1d00a <__trunctfdf2+0x180>
   1d090: 79 d6        	beqz	a2, 0x1d05e <__trunctfdf2+0x1d4>
   1d092: a1 07        	addi	a5, a5, 8
   1d094: e9 b7        	j	0x1d05e <__trunctfdf2+0x1d4>
   1d096: 05 45        	addi	a0, zero, 1
   1d098: e9 f6        	bnez	a3, 0x1d062 <__trunctfdf2+0x1d8>
   1d09a: d1 bf        	j	0x1d06e <__trunctfdf2+0x1e4>
   1d09c: 85 46        	addi	a3, zero, 1
   1d09e: d1 b7        	j	0x1d062 <__trunctfdf2+0x1d8>
   1d0a0: 13 65 55 00  	ori	a0, a0, 5
   1d0a4: 63 06 08 02  	beqz	a6, 0x1d0d0 <__trunctfdf2+0x246>
   1d0a8: 8d 47        	addi	a5, zero, 3
   1d0aa: 63 1a f8 00  	bne	a6, a5, 0x1d0be <__trunctfdf2+0x234>
   1d0ae: 0d c2        	beqz	a2, 0x1d0d0 <__trunctfdf2+0x246>
   1d0b0: 13 07 f0 7f  	addi	a4, zero, 2047
   1d0b4: 52 17        	slli	a4, a4, 52
   1d0b6: 93 17 f6 03  	slli	a5, a2, 63
   1d0ba: 7d 17        	addi	a4, a4, -1
   1d0bc: 39 a8        	j	0x1d0da <__trunctfdf2+0x250>
   1d0be: 89 47        	addi	a5, zero, 2
   1d0c0: e3 18 f8 fe  	bne	a6, a5, 0x1d0b0 <__trunctfdf2+0x226>
   1d0c4: 11 e6        	bnez	a2, 0x1d0d0 <__trunctfdf2+0x246>
   1d0c6: 83 b7 01 83  	ld	a5, -2000(gp)
   1d0ca: 73 20 15 00  	csrs	fflags, a0
   1d0ce: 81 bf        	j	0x1d01e <__trunctfdf2+0x194>
   1d0d0: 13 07 f0 7f  	addi	a4, zero, 2047
   1d0d4: 93 17 f6 03  	slli	a5, a2, 63
   1d0d8: 52 17        	slli	a4, a4, 52
   1d0da: d9 8f        	or	a5, a5, a4
   1d0dc: fd b7        	j	0x1d0ca <__trunctfdf2+0x240>
   1d0de: 13 07 f0 7f  	addi	a4, zero, 2047
   1d0e2: 93 17 f6 03  	slli	a5, a2, 63
   1d0e6: 52 17        	slli	a4, a4, 52
   1d0e8: d9 8f        	or	a5, a5, a4
   1d0ea: 15 45        	addi	a0, zero, 5
   1d0ec: f9 bf        	j	0x1d0ca <__trunctfdf2+0x240>
   1d0ee: 93 07 f0 7f  	addi	a5, zero, 2047
   1d0f2: 13 17 f6 03  	slli	a4, a2, 63
   1d0f6: d2 17        	slli	a5, a5, 52
   1d0f8: d9 8f        	or	a5, a5, a4
   1d0fa: 15 b7        	j	0x1d01e <__trunctfdf2+0x194>
   1d0fc: 93 07 f0 7f  	addi	a5, zero, 2047
   1d100: d2 17        	slli	a5, a5, 52
   1d102: e5 b7        	j	0x1d0ea <__trunctfdf2+0x260>

000000000001d104 <__clzdi2>:
   1d104: 93 57 85 03  	srli	a5, a0, 56
   1d108: 85 eb        	bnez	a5, 0x1d138 <__clzdi2+0x34>
   1d10a: 93 07 80 03  	addi	a5, zero, 56
   1d10e: e1 17        	addi	a5, a5, -8
   1d110: 91 c7        	beqz	a5, 0x1d11c <__clzdi2+0x18>
   1d112: 33 57 f5 00  	srl	a4, a0, a5
   1d116: 13 77 f7 0f  	andi	a4, a4, 255
   1d11a: 75 db        	beqz	a4, 0x1d10e <__clzdi2+0xa>
   1d11c: 79 67        	lui	a4, 30
   1d11e: 93 06 00 04  	addi	a3, zero, 64
   1d122: 9d 8e        	sub	a3, a3, a5
   1d124: 33 55 f5 00  	srl	a0, a0, a5
   1d128: 93 07 07 53  	addi	a5, a4, 1328
   1d12c: aa 97        	add	a5, a5, a0
   1d12e: 03 c5 07 00  	lbu	a0, 0(a5)
   1d132: 3b 85 a6 40  	subw	a0, a3, a0
   1d136: 82 80        	ret
   1d138: 93 07 80 03  	addi	a5, zero, 56
   1d13c: c5 b7        	j	0x1d11c <__clzdi2+0x18>
