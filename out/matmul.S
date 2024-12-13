
/nfs/home/cailuoshan/AISim/llvm-project/matrix-tests/out/matmul.out:	file format elf64-littleriscv

Disassembly of section .text:

0000000000010120 <exit>:
   10120: 41 11        	addi	sp, sp, -16
   10122: 81 45        	mv	a1, zero
   10124: 22 e0        	sd	s0, 0(sp)
   10126: 06 e4        	sd	ra, 8(sp)
   10128: 2a 84        	mv	s0, a0
   1012a: ef 50 20 0f  	jal	0x1521c <__call_exitprocs>
   1012e: 03 b5 81 83  	ld	a0, -1992(gp)
   10132: 3c 6d        	ld	a5, 88(a0)
   10134: 91 c3        	beqz	a5, 0x10138 <exit+0x18>
   10136: 82 97        	jalr	a5
   10138: 22 85        	mv	a0, s0
   1013a: ef e0 50 01  	jal	0x1e94e <_exit>

000000000001013e <register_fini>:
   1013e: 93 07 00 00  	mv	a5, zero
   10142: 91 c7        	beqz	a5, 0x1014e <register_fini+0x10>
   10144: 55 65        	lui	a0, 21
   10146: 13 05 a5 75  	addi	a0, a0, 1882
   1014a: 6f c0 e0 6f  	j	0x1c848 <atexit>
   1014e: 82 80        	ret

0000000000010150 <_start>:
   10150: 97 41 01 00  	auipc	gp, 20
   10154: 93 81 01 7d  	addi	gp, gp, 2000
   10158: 13 85 01 95  	addi	a0, gp, -1712
   1015c: 17 66 01 00  	auipc	a2, 22
   10160: 13 06 c6 40  	addi	a2, a2, 1036
   10164: 09 8e        	sub	a2, a2, a0
   10166: 81 45        	mv	a1, zero
   10168: ef 30 e0 28  	jal	0x133f6 <memset>
   1016c: 17 c5 00 00  	auipc	a0, 12
   10170: 13 05 c5 6d  	addi	a0, a0, 1756
   10174: 19 c5        	beqz	a0, 0x10182 <_start+0x32>
   10176: 17 55 00 00  	auipc	a0, 5
   1017a: 13 05 45 5e  	addi	a0, a0, 1508
   1017e: ef c0 a0 6c  	jal	0x1c848 <atexit>
   10182: ef 30 20 20  	jal	0x13384 <__libc_init_array>
   10186: 02 45        	lw	a0, 0(sp)
   10188: 2c 00        	addi	a1, sp, 8
   1018a: 01 46        	mv	a2, zero
   1018c: ef 00 40 05  	jal	0x101e0 <main>
   10190: 41 bf        	j	0x10120 <exit>

0000000000010192 <__do_global_dtors_aux>:
   10192: 41 11        	addi	sp, sp, -16
   10194: 22 e0        	sd	s0, 0(sp)
   10196: 83 c7 81 98  	lbu	a5, -1656(gp)
   1019a: 06 e4        	sd	ra, 8(sp)
   1019c: 99 ef        	bnez	a5, 0x101ba <__do_global_dtors_aux+0x28>
   1019e: 93 07 00 00  	mv	a5, zero
   101a2: 89 cb        	beqz	a5, 0x101b4 <__do_global_dtors_aux+0x22>
   101a4: 37 35 02 00  	lui	a0, 35
   101a8: 13 05 05 00  	mv	a0, a0
   101ac: 97 00 00 00  	auipc	ra, 0
   101b0: e7 00 00 00  	jalr	zero
   101b4: 85 47        	addi	a5, zero, 1
   101b6: 23 84 f1 98  	sb	a5, -1656(gp)
   101ba: a2 60        	ld	ra, 8(sp)
   101bc: 02 64        	ld	s0, 0(sp)
   101be: 41 01        	addi	sp, sp, 16
   101c0: 82 80        	ret

00000000000101c2 <frame_dummy>:
   101c2: 93 07 00 00  	mv	a5, zero
   101c6: 99 cb        	beqz	a5, 0x101dc <frame_dummy+0x1a>
   101c8: 37 35 02 00  	lui	a0, 35
   101cc: 93 85 01 99  	addi	a1, gp, -1648
   101d0: 13 05 05 00  	mv	a0, a0
   101d4: 17 03 00 00  	auipc	t1, 0
   101d8: 67 00 00 00  	jr	zero
   101dc: 82 80        	ret
   101de: 00 00        	unimp	

00000000000101e0 <main>:
   101e0: 13 01 01 f0  	addi	sp, sp, -256
   101e4: 23 3c 11 0e  	sd	ra, 248(sp)
   101e8: 23 38 81 0e  	sd	s0, 240(sp)
   101ec: 23 34 91 0e  	sd	s1, 232(sp)
   101f0: 23 30 21 0f  	sd	s2, 224(sp)
   101f4: 23 3c 31 0d  	sd	s3, 216(sp)
   101f8: 23 38 41 0d  	sd	s4, 208(sp)
   101fc: 23 34 51 0d  	sd	s5, 200(sp)
   10200: 23 30 61 0d  	sd	s6, 192(sp)
   10204: 23 3c 71 0b  	sd	s7, 184(sp)
   10208: 23 38 81 0b  	sd	s8, 176(sp)
   1020c: 23 34 91 0b  	sd	s9, 168(sp)
   10210: 23 30 a1 0b  	sd	s10, 160(sp)
   10214: 23 3c b1 09  	sd	s11, 152(sp)
   10218: 27 38 81 08  	fsd	fs0, 144(sp)
   1021c: 27 34 91 08  	fsd	fs1, 136(sp)
   10220: 77 75 10 00  	msettypei	a0, e8, m1, ba
   10224: 77 f5 01 20  	msettilemi	a0, 3
   10228: 77 f5 03 40  	msettileki	a0, 7
   1022c: 77 75 02 60  	msettileni	a0, 4
   10230: 17 05 01 00  	auipc	a0, 16
   10234: 13 05 15 f0  	addi	a0, a0, -255
   10238: 93 05 40 00  	addi	a1, zero, 4
   1023c: 77 00 b5 00  	mlce8.m	tr0, (a0), a1, m1
   10240: 17 05 01 00  	auipc	a0, 16
   10244: 13 05 05 ec  	addi	a0, a0, -320
   10248: 13 06 70 00  	addi	a2, zero, 7
   1024c: f7 00 c5 04  	mlae8.m	tr1, (a0), a2, m1
   10250: 17 05 01 00  	auipc	a0, 16
   10254: 13 05 55 ec  	addi	a0, a0, -315
   10258: 77 01 b5 08  	mlbe8.m	tr2, (a0), a1, m1
   1025c: 77 e0 20 00  	mmau.mm	tr0, tr1, tr2
   10260: 93 8b 81 57  	addi	s7, gp, 1400
   10264: 77 80 bb 02  	msce8.m	tr0, (s7), a1, m1
   10268: 93 89 01 95  	addi	s3, gp, -1712
   1026c: 03 a5 09 00  	lw	a0, 0(s3)
   10270: 03 c8 0b 00  	lbu	a6, 0(s7)
   10274: 1b 05 15 00  	addiw	a0, a0, 1
   10278: 93 07 10 01  	addi	a5, zero, 17
   1027c: 23 a0 a9 00  	sw	a0, 0(s3)
   10280: 63 14 f8 0e  	bne	a6, a5, 0x10368 <main+0x188>
   10284: 03 c8 1b 00  	lbu	a6, 1(s7)
   10288: 93 07 d0 02  	addi	a5, zero, 45
   1028c: 13 07 10 00  	addi	a4, zero, 1
   10290: 63 1e f8 0c  	bne	a6, a5, 0x1036c <main+0x18c>
   10294: 03 c8 2b 00  	lbu	a6, 2(s7)
   10298: 93 07 20 02  	addi	a5, zero, 34
   1029c: 13 07 20 00  	addi	a4, zero, 2
   102a0: 63 16 f8 0c  	bne	a6, a5, 0x1036c <main+0x18c>
   102a4: 03 c8 3b 00  	lbu	a6, 3(s7)
   102a8: 93 07 20 02  	addi	a5, zero, 34
   102ac: 13 07 30 00  	addi	a4, zero, 3
   102b0: 63 1e f8 0a  	bne	a6, a5, 0x1036c <main+0x18c>
   102b4: 03 c8 4b 00  	lbu	a6, 4(s7)
   102b8: 93 07 00 02  	addi	a5, zero, 32
   102bc: 13 07 40 00  	addi	a4, zero, 4
   102c0: 63 16 f8 0a  	bne	a6, a5, 0x1036c <main+0x18c>
   102c4: 03 c8 5b 00  	lbu	a6, 5(s7)
   102c8: 93 07 80 01  	addi	a5, zero, 24
   102cc: 13 07 50 00  	addi	a4, zero, 5
   102d0: 63 1e f8 08  	bne	a6, a5, 0x1036c <main+0x18c>
   102d4: 03 c8 6b 00  	lbu	a6, 6(s7)
   102d8: 93 07 e0 00  	addi	a5, zero, 14
   102dc: 13 07 60 00  	addi	a4, zero, 6
   102e0: 63 16 f8 08  	bne	a6, a5, 0x1036c <main+0x18c>
   102e4: 03 c8 7b 00  	lbu	a6, 7(s7)
   102e8: 93 07 40 01  	addi	a5, zero, 20
   102ec: 13 07 70 00  	addi	a4, zero, 7
   102f0: 63 1e f8 06  	bne	a6, a5, 0x1036c <main+0x18c>
   102f4: 03 c8 8b 00  	lbu	a6, 8(s7)
   102f8: 93 07 30 01  	addi	a5, zero, 19
   102fc: 13 07 80 00  	addi	a4, zero, 8
   10300: 63 16 f8 06  	bne	a6, a5, 0x1036c <main+0x18c>
   10304: 03 c8 9b 00  	lbu	a6, 9(s7)
   10308: 93 07 30 01  	addi	a5, zero, 19
   1030c: 13 07 90 00  	addi	a4, zero, 9
   10310: 63 1e f8 04  	bne	a6, a5, 0x1036c <main+0x18c>
   10314: 03 c8 ab 00  	lbu	a6, 10(s7)
   10318: 93 07 30 01  	addi	a5, zero, 19
   1031c: 13 07 a0 00  	addi	a4, zero, 10
   10320: 63 16 f8 04  	bne	a6, a5, 0x1036c <main+0x18c>
   10324: 03 c8 bb 00  	lbu	a6, 11(s7)
   10328: 13 07 b0 00  	addi	a4, zero, 11
   1032c: 93 07 b0 00  	addi	a5, zero, 11
   10330: 63 1e e8 02  	bne	a6, a4, 0x1036c <main+0x18c>
   10334: 13 85 41 95  	addi	a0, gp, -1708
   10338: 83 25 05 00  	lw	a1, 0(a0)
   1033c: 9b 85 15 00  	addiw	a1, a1, 1
   10340: 23 20 b5 00  	sw	a1, 0(a0)
   10344: 17 15 01 00  	auipc	a0, 17
   10348: 13 05 15 16  	addi	a0, a0, 353
   1034c: 97 15 01 00  	auipc	a1, 17
   10350: 93 85 55 13  	addi	a1, a1, 309
   10354: 97 16 01 00  	auipc	a3, 17
   10358: 93 86 c6 13  	addi	a3, a3, 316
   1035c: 13 06 60 01  	addi	a2, zero, 22
   10360: ef 30 20 16  	jal	0x134c2 <printf>
   10364: 6f 00 80 02  	j	0x1038c <main+0x1ac>
   10368: 13 07 00 00  	mv	a4, zero
   1036c: 17 15 01 00  	auipc	a0, 17
   10370: 13 05 45 0d  	addi	a0, a0, 212
   10374: 97 15 01 00  	auipc	a1, 17
   10378: 93 85 d5 10  	addi	a1, a1, 269
   1037c: 97 16 01 00  	auipc	a3, 17
   10380: 93 86 46 11  	addi	a3, a3, 276
   10384: 13 06 60 01  	addi	a2, zero, 22
   10388: ef 30 a0 13  	jal	0x134c2 <printf>
   1038c: 77 75 12 00  	msettypei	a0, e16, m1, ba
   10390: 77 75 01 20  	msettilemi	a0, 2
   10394: 77 f5 02 40  	msettileki	a0, 5
   10398: 77 f5 01 60  	msettileni	a0, 3
   1039c: 17 05 01 00  	auipc	a0, 16
   103a0: 13 05 45 dd  	addi	a0, a0, -556
   103a4: 93 05 60 00  	addi	a1, zero, 6
   103a8: 77 10 b5 00  	mlce16.m	tr0, (a0), a1, m1
   103ac: 17 05 01 00  	auipc	a0, 16
   103b0: 13 05 25 d9  	addi	a0, a0, -622
   103b4: 13 06 a0 00  	addi	a2, zero, 10
   103b8: f7 10 c5 04  	mlae16.m	tr1, (a0), a2, m1
   103bc: 17 05 01 00  	auipc	a0, 16
   103c0: 13 05 65 d9  	addi	a0, a0, -618
   103c4: 77 11 b5 08  	mlbe16.m	tr2, (a0), a1, m1
   103c8: 77 e0 20 00  	mmau.mm	tr0, tr1, tr2
   103cc: 93 8d 01 64  	addi	s11, gp, 1600
   103d0: 77 90 bd 02  	msce16.m	tr0, (s11), a1, m1
   103d4: 03 a5 09 00  	lw	a0, 0(s3)
   103d8: 03 d8 0d 00  	lhu	a6, 0(s11)
   103dc: 1b 05 15 00  	addiw	a0, a0, 1
   103e0: b7 35 00 00  	lui	a1, 3
   103e4: 9b 87 15 62  	addiw	a5, a1, 1569
   103e8: 23 a0 a9 00  	sw	a0, 0(s3)
   103ec: 63 1c f8 08  	bne	a6, a5, 0x10484 <main+0x2a4>
   103f0: 13 85 01 64  	addi	a0, gp, 1600
   103f4: 03 58 25 00  	lhu	a6, 2(a0)
   103f8: b7 25 00 00  	lui	a1, 2
   103fc: 9b 87 85 70  	addiw	a5, a1, 1800
   10400: 13 07 10 00  	addi	a4, zero, 1
   10404: 63 12 f8 08  	bne	a6, a5, 0x10488 <main+0x2a8>
   10408: 03 58 45 00  	lhu	a6, 4(a0)
   1040c: 9b 87 e5 19  	addiw	a5, a1, 414
   10410: 13 07 20 00  	addi	a4, zero, 2
   10414: 63 1a f8 06  	bne	a6, a5, 0x10488 <main+0x2a8>
   10418: 03 58 65 00  	lhu	a6, 6(a0)
   1041c: b7 25 00 00  	lui	a1, 2
   10420: 9b 87 a5 90  	addiw	a5, a1, -1782
   10424: 13 07 30 00  	addi	a4, zero, 3
   10428: 63 10 f8 06  	bne	a6, a5, 0x10488 <main+0x2a8>
   1042c: 03 58 85 00  	lhu	a6, 8(a0)
   10430: 9b 87 35 bf  	addiw	a5, a1, -1037
   10434: 13 07 40 00  	addi	a4, zero, 4
   10438: 63 18 f8 04  	bne	a6, a5, 0x10488 <main+0x2a8>
   1043c: 03 58 a5 00  	lhu	a6, 10(a0)
   10440: 37 15 00 00  	lui	a0, 1
   10444: 9b 07 35 49  	addiw	a5, a0, 1171
   10448: 13 07 50 00  	addi	a4, zero, 5
   1044c: 63 1e f8 02  	bne	a6, a5, 0x10488 <main+0x2a8>
   10450: 13 85 41 95  	addi	a0, gp, -1708
   10454: 83 25 05 00  	lw	a1, 0(a0)
   10458: 9b 85 15 00  	addiw	a1, a1, 1
   1045c: 23 20 b5 00  	sw	a1, 0(a0)
   10460: 17 15 01 00  	auipc	a0, 17
   10464: 13 05 55 04  	addi	a0, a0, 69
   10468: 97 15 01 00  	auipc	a1, 17
   1046c: 93 85 95 01  	addi	a1, a1, 25
   10470: 97 16 01 00  	auipc	a3, 17
   10474: 93 86 76 08  	addi	a3, a3, 135
   10478: 13 06 60 02  	addi	a2, zero, 38
   1047c: ef 30 60 04  	jal	0x134c2 <printf>
   10480: 6f 00 80 02  	j	0x104a8 <main+0x2c8>
   10484: 13 07 00 00  	mv	a4, zero
   10488: 17 15 01 00  	auipc	a0, 17
   1048c: 13 05 e5 02  	addi	a0, a0, 46
   10490: 97 15 01 00  	auipc	a1, 17
   10494: 93 85 15 ff  	addi	a1, a1, -15
   10498: 97 16 01 00  	auipc	a3, 17
   1049c: 93 86 f6 05  	addi	a3, a3, 95
   104a0: 13 06 60 02  	addi	a2, zero, 38
   104a4: ef 30 e0 01  	jal	0x134c2 <printf>
   104a8: 77 75 14 00  	msettypei	a0, e32, m1, ba
   104ac: 77 75 01 20  	msettilemi	a0, 2
   104b0: 77 f5 01 40  	msettileki	a0, 3
   104b4: 77 75 01 60  	msettileni	a0, 2
   104b8: 17 15 01 00  	auipc	a0, 17
   104bc: 13 05 45 36  	addi	a0, a0, 868
   104c0: 93 05 80 00  	addi	a1, zero, 8
   104c4: 77 20 b5 00  	mlce32.m	tr0, (a0), a1, m1
   104c8: 17 05 01 00  	auipc	a0, 16
   104cc: 13 05 45 cb  	addi	a0, a0, -844
   104d0: 13 06 c0 00  	addi	a2, zero, 12
   104d4: f7 20 c5 04  	mlae32.m	tr1, (a0), a2, m1
   104d8: 17 05 01 00  	auipc	a0, 16
   104dc: 13 05 c5 cb  	addi	a0, a0, -836
   104e0: 77 21 b5 08  	mlbe32.m	tr2, (a0), a1, m1
   104e4: 77 e0 20 00  	mmau.mm	tr0, tr1, tr2
   104e8: 17 5d 01 00  	auipc	s10, 21
   104ec: 13 0d 8d c0  	addi	s10, s10, -1016
   104f0: 77 20 bd 02  	msce32.m	tr0, (s10), a1, m1
   104f4: 83 a5 09 00  	lw	a1, 0(s3)
   104f8: 03 65 0d 00  	lwu	a0, 0(s10)
   104fc: 9b 85 15 00  	addiw	a1, a1, 1
   10500: 37 36 08 00  	lui	a2, 131
   10504: 9b 07 46 cb  	addiw	a5, a2, -844
   10508: 23 a0 b9 00  	sw	a1, 0(s3)
   1050c: 63 1e f5 06  	bne	a0, a5, 0x10588 <main+0x3a8>
   10510: 97 55 01 00  	auipc	a1, 21
   10514: 93 85 05 be  	addi	a1, a1, -1056
   10518: 03 e5 45 00  	lwu	a0, 4(a1)
   1051c: 37 f6 0a 00  	lui	a2, 175
   10520: 9b 07 96 96  	addiw	a5, a2, -1687
   10524: 13 07 10 00  	addi	a4, zero, 1
   10528: 63 12 f5 06  	bne	a0, a5, 0x1058c <main+0x3ac>
   1052c: 03 e5 85 00  	lwu	a0, 8(a1)
   10530: 37 56 02 00  	lui	a2, 37
   10534: 9b 07 a6 53  	addiw	a5, a2, 1338
   10538: 13 07 20 00  	addi	a4, zero, 2
   1053c: 63 18 f5 04  	bne	a0, a5, 0x1058c <main+0x3ac>
   10540: 03 e5 c5 00  	lwu	a0, 12(a1)
   10544: b7 c5 08 00  	lui	a1, 140
   10548: 9b 87 b5 d9  	addiw	a5, a1, -613
   1054c: 13 07 30 00  	addi	a4, zero, 3
   10550: 63 1e f5 02  	bne	a0, a5, 0x1058c <main+0x3ac>
   10554: 13 85 41 95  	addi	a0, gp, -1708
   10558: 83 25 05 00  	lw	a1, 0(a0)
   1055c: 9b 85 15 00  	addiw	a1, a1, 1
   10560: 23 20 b5 00  	sw	a1, 0(a0)
   10564: 17 15 01 00  	auipc	a0, 17
   10568: 13 05 15 f4  	addi	a0, a0, -191
   1056c: 97 15 01 00  	auipc	a1, 17
   10570: 93 85 55 f1  	addi	a1, a1, -235
   10574: 97 16 01 00  	auipc	a3, 17
   10578: 93 86 86 f9  	addi	a3, a3, -104
   1057c: 13 06 50 03  	addi	a2, zero, 53
   10580: ef 20 30 74  	jal	0x134c2 <printf>
   10584: 6f 00 c0 02  	j	0x105b0 <main+0x3d0>
   10588: 13 07 00 00  	mv	a4, zero
   1058c: 1b 08 05 00  	sext.w	a6, a0
   10590: 17 15 01 00  	auipc	a0, 17
   10594: 13 05 65 f2  	addi	a0, a0, -218
   10598: 97 15 01 00  	auipc	a1, 17
   1059c: 93 85 95 ee  	addi	a1, a1, -279
   105a0: 97 16 01 00  	auipc	a3, 17
   105a4: 93 86 c6 f6  	addi	a3, a3, -148
   105a8: 13 06 50 03  	addi	a2, zero, 53
   105ac: ef 20 70 71  	jal	0x134c2 <printf>
   105b0: 77 75 16 00  	msettypei	a0, e64, m1, ba
   105b4: 77 75 01 20  	msettilemi	a0, 2
   105b8: 77 75 01 40  	msettileki	a0, 2
   105bc: 77 75 01 60  	msettileni	a0, 2
   105c0: 17 15 01 00  	auipc	a0, 17
   105c4: 13 05 05 33  	addi	a0, a0, 816
   105c8: 93 05 00 01  	addi	a1, zero, 16
   105cc: 77 30 b5 00  	mlce64.m	tr0, (a0), a1, m1
   105d0: 17 15 01 00  	auipc	a0, 17
   105d4: 13 05 05 2e  	addi	a0, a0, 736
   105d8: f7 30 b5 04  	mlae64.m	tr1, (a0), a1, m1
   105dc: 17 15 01 00  	auipc	a0, 17
   105e0: 13 05 45 2f  	addi	a0, a0, 756
   105e4: 77 31 b5 08  	mlbe64.m	tr2, (a0), a1, m1
   105e8: 77 e0 20 00  	mmau.mm	tr0, tr1, tr2
   105ec: 97 5a 01 00  	auipc	s5, 21
   105f0: 93 8a 4a e2  	addi	s5, s5, -476
   105f4: 77 b0 ba 02  	msce64.m	tr0, (s5), a1, m1
   105f8: 03 a5 09 00  	lw	a0, 0(s3)
   105fc: 03 b8 0a 00  	ld	a6, 0(s5)
   10600: 1b 05 15 00  	addiw	a0, a0, 1
   10604: b7 25 32 01  	lui	a1, 4898
   10608: 9b 87 25 28  	addiw	a5, a1, 642
   1060c: 23 a0 a9 00  	sw	a0, 0(s3)
   10610: 63 1a f8 06  	bne	a6, a5, 0x10684 <main+0x4a4>
   10614: 03 b8 8a 00  	ld	a6, 8(s5)
   10618: 37 c5 8f 01  	lui	a0, 6396
   1061c: 9b 07 d5 83  	addiw	a5, a0, -1987
   10620: 13 07 10 00  	addi	a4, zero, 1
   10624: 63 12 f8 06  	bne	a6, a5, 0x10688 <main+0x4a8>
   10628: 03 b8 0a 01  	ld	a6, 16(s5)
   1062c: 37 55 c3 01  	lui	a0, 7221
   10630: 9b 07 e5 0f  	addiw	a5, a0, 254
   10634: 13 07 20 00  	addi	a4, zero, 2
   10638: 63 18 f8 04  	bne	a6, a5, 0x10688 <main+0x4a8>
   1063c: 03 b8 8a 01  	ld	a6, 24(s5)
   10640: 37 b5 cf 01  	lui	a0, 7419
   10644: 9b 07 55 c5  	addiw	a5, a0, -939
   10648: 13 07 30 00  	addi	a4, zero, 3
   1064c: 63 1e f8 02  	bne	a6, a5, 0x10688 <main+0x4a8>
   10650: 13 85 41 95  	addi	a0, gp, -1708
   10654: 83 25 05 00  	lw	a1, 0(a0)
   10658: 9b 85 15 00  	addiw	a1, a1, 1
   1065c: 23 20 b5 00  	sw	a1, 0(a0)
   10660: 17 15 01 00  	auipc	a0, 17
   10664: 13 05 55 e4  	addi	a0, a0, -443
   10668: 97 15 01 00  	auipc	a1, 17
   1066c: 93 85 95 e1  	addi	a1, a1, -487
   10670: 97 16 01 00  	auipc	a3, 17
   10674: 93 86 46 ef  	addi	a3, a3, -268
   10678: 13 06 40 04  	addi	a2, zero, 68
   1067c: ef 20 70 64  	jal	0x134c2 <printf>
   10680: 6f 00 80 02  	j	0x106a8 <main+0x4c8>
   10684: 13 07 00 00  	mv	a4, zero
   10688: 17 15 01 00  	auipc	a0, 17
   1068c: 13 05 95 e9  	addi	a0, a0, -359
   10690: 97 15 01 00  	auipc	a1, 17
   10694: 93 85 15 df  	addi	a1, a1, -527
   10698: 97 16 01 00  	auipc	a3, 17
   1069c: 93 86 c6 ec  	addi	a3, a3, -308
   106a0: 13 06 40 04  	addi	a2, zero, 68
   106a4: ef 20 f0 61  	jal	0x134c2 <printf>
   106a8: 77 75 10 00  	msettypei	a0, e8, m1, ba
   106ac: 77 f5 01 20  	msettilemi	a0, 3
   106b0: 77 f5 03 40  	msettileki	a0, 7
   106b4: 77 75 02 60  	msettileni	a0, 4
   106b8: 17 05 01 00  	auipc	a0, 16
   106bc: 13 05 55 b2  	addi	a0, a0, -1243
   106c0: 93 05 40 00  	addi	a1, zero, 4
   106c4: 77 00 b5 00  	mlce8.m	tr0, (a0), a1, m1
   106c8: 17 05 01 00  	auipc	a0, 16
   106cc: 13 05 45 ae  	addi	a0, a0, -1308
   106d0: 13 06 70 00  	addi	a2, zero, 7
   106d4: f7 00 c5 04  	mlae8.m	tr1, (a0), a2, m1
   106d8: 17 05 01 00  	auipc	a0, 16
   106dc: 13 05 95 ae  	addi	a0, a0, -1303
   106e0: 77 01 b5 08  	mlbe8.m	tr2, (a0), a1, m1
   106e4: 77 e0 28 00  	mma.mm	tr0, tr1, tr2
   106e8: 93 8c 01 9c  	addi	s9, gp, -1600
   106ec: 77 80 bc 02  	msce8.m	tr0, (s9), a1, m1
   106f0: 83 a5 09 00  	lw	a1, 0(s3)
   106f4: 03 c5 0c 00  	lbu	a0, 0(s9)
   106f8: 1b 86 15 00  	addiw	a2, a1, 1
   106fc: 93 05 70 01  	addi	a1, zero, 23
   10700: 23 a0 c9 00  	sw	a2, 0(s3)
   10704: 63 14 b5 0e  	bne	a0, a1, 0x107ec <main+0x60c>
   10708: 03 c5 1c 00  	lbu	a0, 1(s9)
   1070c: 93 05 80 01  	addi	a1, zero, 24
   10710: 13 07 10 00  	addi	a4, zero, 1
   10714: 63 1e b5 0c  	bne	a0, a1, 0x107f0 <main+0x610>
   10718: 03 c5 2c 00  	lbu	a0, 2(s9)
   1071c: 93 05 70 0e  	addi	a1, zero, 231
   10720: 13 07 20 00  	addi	a4, zero, 2
   10724: 63 16 b5 0c  	bne	a0, a1, 0x107f0 <main+0x610>
   10728: 03 c5 3c 00  	lbu	a0, 3(s9)
   1072c: 93 05 40 0e  	addi	a1, zero, 228
   10730: 13 07 30 00  	addi	a4, zero, 3
   10734: 63 1e b5 0a  	bne	a0, a1, 0x107f0 <main+0x610>
   10738: 03 c5 4c 00  	lbu	a0, 4(s9)
   1073c: 93 05 70 0f  	addi	a1, zero, 247
   10740: 13 07 40 00  	addi	a4, zero, 4
   10744: 63 16 b5 0a  	bne	a0, a1, 0x107f0 <main+0x610>
   10748: 03 c5 5c 00  	lbu	a0, 5(s9)
   1074c: 93 05 90 01  	addi	a1, zero, 25
   10750: 13 07 50 00  	addi	a4, zero, 5
   10754: 63 1e b5 08  	bne	a0, a1, 0x107f0 <main+0x610>
   10758: 03 c5 6c 00  	lbu	a0, 6(s9)
   1075c: 93 05 e0 02  	addi	a1, zero, 46
   10760: 13 07 60 00  	addi	a4, zero, 6
   10764: 63 16 b5 08  	bne	a0, a1, 0x107f0 <main+0x610>
   10768: 03 c5 7c 00  	lbu	a0, 7(s9)
   1076c: 93 05 b0 03  	addi	a1, zero, 59
   10770: 13 07 70 00  	addi	a4, zero, 7
   10774: 63 1e b5 06  	bne	a0, a1, 0x107f0 <main+0x610>
   10778: 03 c5 8c 00  	lbu	a0, 8(s9)
   1077c: 93 05 d0 00  	addi	a1, zero, 13
   10780: 13 07 80 00  	addi	a4, zero, 8
   10784: 63 16 b5 06  	bne	a0, a1, 0x107f0 <main+0x610>
   10788: 03 c5 9c 00  	lbu	a0, 9(s9)
   1078c: 93 05 d0 00  	addi	a1, zero, 13
   10790: 13 07 90 00  	addi	a4, zero, 9
   10794: 63 1e b5 04  	bne	a0, a1, 0x107f0 <main+0x610>
   10798: 03 c5 ac 00  	lbu	a0, 10(s9)
   1079c: 93 05 f0 0e  	addi	a1, zero, 239
   107a0: 13 07 a0 00  	addi	a4, zero, 10
   107a4: 63 16 b5 04  	bne	a0, a1, 0x107f0 <main+0x610>
   107a8: 03 c5 bc 00  	lbu	a0, 11(s9)
   107ac: 93 05 30 00  	addi	a1, zero, 3
   107b0: 13 07 b0 00  	addi	a4, zero, 11
   107b4: 63 1e b5 02  	bne	a0, a1, 0x107f0 <main+0x610>
   107b8: 13 85 41 95  	addi	a0, gp, -1708
   107bc: 83 25 05 00  	lw	a1, 0(a0)
   107c0: 9b 85 15 00  	addiw	a1, a1, 1
   107c4: 23 20 b5 00  	sw	a1, 0(a0)
   107c8: 17 15 01 00  	auipc	a0, 17
   107cc: 13 05 d5 cd  	addi	a0, a0, -803
   107d0: 97 15 01 00  	auipc	a1, 17
   107d4: 93 85 15 cb  	addi	a1, a1, -847
   107d8: 97 16 01 00  	auipc	a3, 17
   107dc: 93 86 16 da  	addi	a3, a3, -607
   107e0: 13 06 90 05  	addi	a2, zero, 89
   107e4: ef 20 f0 4d  	jal	0x134c2 <printf>
   107e8: 6f 00 80 03  	j	0x10820 <main+0x640>
   107ec: 13 07 00 00  	mv	a4, zero
   107f0: 93 95 85 03  	slli	a1, a1, 56
   107f4: 93 d7 85 43  	srai	a5, a1, 56
   107f8: 13 15 85 03  	slli	a0, a0, 56
   107fc: 13 58 85 43  	srai	a6, a0, 56
   10800: 17 15 01 00  	auipc	a0, 17
   10804: 13 05 05 c4  	addi	a0, a0, -960
   10808: 97 15 01 00  	auipc	a1, 17
   1080c: 93 85 95 c7  	addi	a1, a1, -903
   10810: 97 16 01 00  	auipc	a3, 17
   10814: 93 86 96 d6  	addi	a3, a3, -663
   10818: 13 06 90 05  	addi	a2, zero, 89
   1081c: ef 20 70 4a  	jal	0x134c2 <printf>
   10820: 77 75 12 00  	msettypei	a0, e16, m1, ba
   10824: 77 75 01 20  	msettilemi	a0, 2
   10828: 77 f5 02 40  	msettileki	a0, 5
   1082c: 77 f5 01 60  	msettileni	a0, 3
   10830: 17 05 01 00  	auipc	a0, 16
   10834: 13 05 c5 9e  	addi	a0, a0, -1556
   10838: 93 05 60 00  	addi	a1, zero, 6
   1083c: 77 10 b5 00  	mlce16.m	tr0, (a0), a1, m1
   10840: 17 05 01 00  	auipc	a0, 16
   10844: 13 05 a5 9a  	addi	a0, a0, -1622
   10848: 13 06 a0 00  	addi	a2, zero, 10
   1084c: f7 10 c5 04  	mlae16.m	tr1, (a0), a2, m1
   10850: 17 05 01 00  	auipc	a0, 16
   10854: 13 05 e5 9a  	addi	a0, a0, -1618
   10858: 77 11 b5 08  	mlbe16.m	tr2, (a0), a1, m1
   1085c: 77 e0 28 00  	mma.mm	tr0, tr1, tr2
   10860: 13 84 81 a8  	addi	s0, gp, -1400
   10864: 77 10 b4 02  	msce16.m	tr0, (s0), a1, m1
   10868: 83 a5 09 00  	lw	a1, 0(s3)
   1086c: 03 55 04 00  	lhu	a0, 0(s0)
   10870: 1b 86 15 00  	addiw	a2, a1, 1
   10874: b7 05 01 00  	lui	a1, 16
   10878: 9b 85 05 f4  	addiw	a1, a1, -192
   1087c: 23 a0 c9 00  	sw	a2, 0(s3)
   10880: 63 1a b5 08  	bne	a0, a1, 0x10914 <main+0x734>
   10884: 13 86 81 a8  	addi	a2, gp, -1400
   10888: 03 55 26 00  	lhu	a0, 2(a2)
   1088c: 93 05 f0 03  	addi	a1, zero, 63
   10890: 13 07 10 00  	addi	a4, zero, 1
   10894: 63 12 b5 08  	bne	a0, a1, 0x10918 <main+0x738>
   10898: 03 55 46 00  	lhu	a0, 4(a2)
   1089c: b7 06 01 00  	lui	a3, 16
   108a0: 9b 85 b6 f3  	addiw	a1, a3, -197
   108a4: 13 07 20 00  	addi	a4, zero, 2
   108a8: 63 18 b5 06  	bne	a0, a1, 0x10918 <main+0x738>
   108ac: 03 55 66 00  	lhu	a0, 6(a2)
   108b0: 9b 85 56 f3  	addiw	a1, a3, -203
   108b4: 13 07 30 00  	addi	a4, zero, 3
   108b8: 63 10 b5 06  	bne	a0, a1, 0x10918 <main+0x738>
   108bc: 03 55 86 00  	lhu	a0, 8(a2)
   108c0: b7 06 01 00  	lui	a3, 16
   108c4: 9b 85 66 f9  	addiw	a1, a3, -106
   108c8: 13 07 40 00  	addi	a4, zero, 4
   108cc: 63 16 b5 04  	bne	a0, a1, 0x10918 <main+0x738>
   108d0: 03 55 a6 00  	lhu	a0, 10(a2)
   108d4: 9b 85 36 fa  	addiw	a1, a3, -93
   108d8: 13 07 50 00  	addi	a4, zero, 5
   108dc: 63 1e b5 02  	bne	a0, a1, 0x10918 <main+0x738>
   108e0: 13 85 41 95  	addi	a0, gp, -1708
   108e4: 83 25 05 00  	lw	a1, 0(a0)
   108e8: 9b 85 15 00  	addiw	a1, a1, 1
   108ec: 23 20 b5 00  	sw	a1, 0(a0)
   108f0: 17 15 01 00  	auipc	a0, 17
   108f4: 13 05 55 bb  	addi	a0, a0, -1099
   108f8: 97 15 01 00  	auipc	a1, 17
   108fc: 93 85 95 b8  	addi	a1, a1, -1143
   10900: 97 16 01 00  	auipc	a3, 17
   10904: 93 86 e6 c8  	addi	a3, a3, -882
   10908: 13 06 90 06  	addi	a2, zero, 105
   1090c: ef 20 70 3b  	jal	0x134c2 <printf>
   10910: 6f 00 80 03  	j	0x10948 <main+0x768>
   10914: 13 07 00 00  	mv	a4, zero
   10918: 93 95 05 03  	slli	a1, a1, 48
   1091c: 93 d7 05 43  	srai	a5, a1, 48
   10920: 13 15 05 03  	slli	a0, a0, 48
   10924: 13 58 05 43  	srai	a6, a0, 48
   10928: 17 15 01 00  	auipc	a0, 17
   1092c: 13 05 85 b1  	addi	a0, a0, -1256
   10930: 97 15 01 00  	auipc	a1, 17
   10934: 93 85 15 b5  	addi	a1, a1, -1199
   10938: 97 16 01 00  	auipc	a3, 17
   1093c: 93 86 66 c5  	addi	a3, a3, -938
   10940: 13 06 90 06  	addi	a2, zero, 105
   10944: ef 20 f0 37  	jal	0x134c2 <printf>
   10948: 77 75 14 00  	msettypei	a0, e32, m1, ba
   1094c: 77 75 01 20  	msettilemi	a0, 2
   10950: 77 f5 01 40  	msettileki	a0, 3
   10954: 77 75 01 60  	msettileni	a0, 2
   10958: 17 15 01 00  	auipc	a0, 17
   1095c: 13 05 45 ed  	addi	a0, a0, -300
   10960: 93 05 80 00  	addi	a1, zero, 8
   10964: 77 20 b5 00  	mlce32.m	tr0, (a0), a1, m1
   10968: 17 05 01 00  	auipc	a0, 16
   1096c: 13 05 05 8c  	addi	a0, a0, -1856
   10970: 13 06 c0 00  	addi	a2, zero, 12
   10974: f7 20 c5 04  	mlae32.m	tr1, (a0), a2, m1
   10978: 17 05 01 00  	auipc	a0, 16
   1097c: 13 05 85 8c  	addi	a0, a0, -1848
   10980: 77 21 b5 08  	mlbe32.m	tr2, (a0), a1, m1
   10984: 77 e0 28 00  	mma.mm	tr0, tr1, tr2
   10988: 13 8c 81 c1  	addi	s8, gp, -1000
   1098c: 77 20 bc 02  	msce32.m	tr0, (s8), a1, m1
   10990: 83 a5 09 00  	lw	a1, 0(s3)
   10994: 03 65 0c 00  	lwu	a0, 0(s8)
   10998: 9b 85 15 00  	addiw	a1, a1, 1
   1099c: 37 b6 0c 00  	lui	a2, 203
   109a0: 9b 07 66 84  	addiw	a5, a2, -1978
   109a4: 23 a0 b9 00  	sw	a1, 0(s3)
   109a8: 63 1c f5 06  	bne	a0, a5, 0x10a20 <main+0x840>
   109ac: 93 85 81 c1  	addi	a1, gp, -1000
   109b0: 03 e5 45 00  	lwu	a0, 4(a1)
   109b4: 37 76 0b 00  	lui	a2, 183
   109b8: 9b 07 96 6e  	addiw	a5, a2, 1769
   109bc: 13 07 10 00  	addi	a4, zero, 1
   109c0: 63 12 f5 06  	bne	a0, a5, 0x10a24 <main+0x844>
   109c4: 03 e5 85 00  	lwu	a0, 8(a1)
   109c8: 37 e6 06 00  	lui	a2, 110
   109cc: 9b 07 66 96  	addiw	a5, a2, -1690
   109d0: 13 07 20 00  	addi	a4, zero, 2
   109d4: 63 18 f5 04  	bne	a0, a5, 0x10a24 <main+0x844>
   109d8: 03 e5 c5 00  	lwu	a0, 12(a1)
   109dc: b7 75 0a 00  	lui	a1, 167
   109e0: 9b 87 d5 c7  	addiw	a5, a1, -899
   109e4: 13 07 30 00  	addi	a4, zero, 3
   109e8: 63 1e f5 02  	bne	a0, a5, 0x10a24 <main+0x844>
   109ec: 13 85 41 95  	addi	a0, gp, -1708
   109f0: 83 25 05 00  	lw	a1, 0(a0)
   109f4: 9b 85 15 00  	addiw	a1, a1, 1
   109f8: 23 20 b5 00  	sw	a1, 0(a0)
   109fc: 17 15 01 00  	auipc	a0, 17
   10a00: 13 05 95 aa  	addi	a0, a0, -1367
   10a04: 97 15 01 00  	auipc	a1, 17
   10a08: 93 85 d5 a7  	addi	a1, a1, -1411
   10a0c: 97 16 01 00  	auipc	a3, 17
   10a10: 93 86 66 b9  	addi	a3, a3, -1130
   10a14: 13 06 80 07  	addi	a2, zero, 120
   10a18: ef 20 b0 2a  	jal	0x134c2 <printf>
   10a1c: 6f 00 c0 02  	j	0x10a48 <main+0x868>
   10a20: 13 07 00 00  	mv	a4, zero
   10a24: 1b 08 05 00  	sext.w	a6, a0
   10a28: 17 15 01 00  	auipc	a0, 17
   10a2c: 13 05 85 a1  	addi	a0, a0, -1512
   10a30: 97 15 01 00  	auipc	a1, 17
   10a34: 93 85 15 a5  	addi	a1, a1, -1455
   10a38: 97 16 01 00  	auipc	a3, 17
   10a3c: 93 86 a6 b6  	addi	a3, a3, -1174
   10a40: 13 06 80 07  	addi	a2, zero, 120
   10a44: ef 20 f0 27  	jal	0x134c2 <printf>
   10a48: 77 75 16 00  	msettypei	a0, e64, m1, ba
   10a4c: 77 75 01 20  	msettilemi	a0, 2
   10a50: 77 75 01 40  	msettileki	a0, 2
   10a54: 77 75 01 60  	msettileni	a0, 2
   10a58: 17 15 01 00  	auipc	a0, 17
   10a5c: 13 05 85 ef  	addi	a0, a0, -264
   10a60: 93 05 00 01  	addi	a1, zero, 16
   10a64: 77 30 b5 00  	mlce64.m	tr0, (a0), a1, m1
   10a68: 17 15 01 00  	auipc	a0, 17
   10a6c: 13 05 85 ea  	addi	a0, a0, -344
   10a70: f7 30 b5 04  	mlae64.m	tr1, (a0), a1, m1
   10a74: 17 15 01 00  	auipc	a0, 17
   10a78: 13 05 c5 eb  	addi	a0, a0, -324
   10a7c: 77 31 b5 08  	mlbe64.m	tr2, (a0), a1, m1
   10a80: 77 e0 28 00  	mma.mm	tr0, tr1, tr2
   10a84: 13 89 81 f3  	addi	s2, gp, -200
   10a88: 77 30 b9 02  	msce64.m	tr0, (s2), a1, m1
   10a8c: 03 a5 09 00  	lw	a0, 0(s3)
   10a90: 03 38 09 00  	ld	a6, 0(s2)
   10a94: 1b 05 15 00  	addiw	a0, a0, 1
   10a98: b7 f5 82 fe  	lui	a1, 1042479
   10a9c: 9b 87 b5 98  	addiw	a5, a1, -1653
   10aa0: 23 a0 a9 00  	sw	a0, 0(s3)
   10aa4: 63 1a f8 06  	bne	a6, a5, 0x10b18 <main+0x938>
   10aa8: 03 38 89 00  	ld	a6, 8(s2)
   10aac: 37 55 a3 00  	lui	a0, 2613
   10ab0: 9b 07 55 72  	addiw	a5, a0, 1829
   10ab4: 13 07 10 00  	addi	a4, zero, 1
   10ab8: 63 12 f8 06  	bne	a6, a5, 0x10b1c <main+0x93c>
   10abc: 03 38 09 01  	ld	a6, 16(s2)
   10ac0: 37 a5 91 ff  	lui	a0, 1046810
   10ac4: 9b 07 e5 26  	addiw	a5, a0, 622
   10ac8: 13 07 20 00  	addi	a4, zero, 2
   10acc: 63 18 f8 04  	bne	a6, a5, 0x10b1c <main+0x93c>
   10ad0: 03 38 89 01  	ld	a6, 24(s2)
   10ad4: 37 f5 3e 01  	lui	a0, 5103
   10ad8: 9b 07 75 0a  	addiw	a5, a0, 167
   10adc: 13 07 30 00  	addi	a4, zero, 3
   10ae0: 63 1e f8 02  	bne	a6, a5, 0x10b1c <main+0x93c>
   10ae4: 13 85 41 95  	addi	a0, gp, -1708
   10ae8: 83 25 05 00  	lw	a1, 0(a0)
   10aec: 9b 85 15 00  	addiw	a1, a1, 1
   10af0: 23 20 b5 00  	sw	a1, 0(a0)
   10af4: 17 15 01 00  	auipc	a0, 17
   10af8: 13 05 15 9b  	addi	a0, a0, -1615
   10afc: 97 15 01 00  	auipc	a1, 17
   10b00: 93 85 55 98  	addi	a1, a1, -1659
   10b04: 97 16 01 00  	auipc	a3, 17
   10b08: 93 86 56 af  	addi	a3, a3, -1291
   10b0c: 13 06 70 08  	addi	a2, zero, 135
   10b10: ef 20 30 1b  	jal	0x134c2 <printf>
   10b14: 6f 00 80 02  	j	0x10b3c <main+0x95c>
   10b18: 13 07 00 00  	mv	a4, zero
   10b1c: 17 15 01 00  	auipc	a0, 17
   10b20: 13 05 a5 a9  	addi	a0, a0, -1382
   10b24: 97 15 01 00  	auipc	a1, 17
   10b28: 93 85 d5 95  	addi	a1, a1, -1699
   10b2c: 97 16 01 00  	auipc	a3, 17
   10b30: 93 86 d6 ac  	addi	a3, a3, -1331
   10b34: 13 06 70 08  	addi	a2, zero, 135
   10b38: ef 20 b0 18  	jal	0x134c2 <printf>
   10b3c: 77 75 12 00  	msettypei	a0, e16, m1, ba
   10b40: 77 75 01 20  	msettilemi	a0, 2
   10b44: 77 75 02 40  	msettileki	a0, 4
   10b48: 77 f5 01 60  	msettileni	a0, 3
   10b4c: 17 f5 00 00  	auipc	a0, 15
   10b50: 13 05 45 72  	addi	a0, a0, 1828
   10b54: 93 05 60 00  	addi	a1, zero, 6
   10b58: 77 10 b5 00  	mlce16.m	tr0, (a0), a1, m1
   10b5c: 17 15 01 00  	auipc	a0, 17
   10b60: 13 05 05 ce  	addi	a0, a0, -800
   10b64: 13 06 80 00  	addi	a2, zero, 8
   10b68: f7 10 c5 04  	mlae16.m	tr1, (a0), a2, m1
   10b6c: 17 f5 00 00  	auipc	a0, 15
   10b70: 13 05 c5 6e  	addi	a0, a0, 1772
   10b74: 77 11 b5 08  	mlbe16.m	tr2, (a0), a1, m1
   10b78: 13 07 00 00  	mv	a4, zero
   10b7c: 77 e0 20 02  	mfma.mm	tr0, tr1, tr2
   10b80: 17 55 01 00  	auipc	a0, 21
   10b84: 13 05 05 ed  	addi	a0, a0, -304
   10b88: 77 10 b5 02  	msce16.m	tr0, (a0), a1, m1
   10b8c: 83 a5 09 00  	lw	a1, 0(s3)
   10b90: 07 10 05 00  	flh	ft0, 0(a0)
   10b94: 13 86 81 84  	addi	a2, gp, -1976
   10b98: 87 10 06 00  	flh	ft1, 0(a2)
   10b9c: 13 86 01 85  	addi	a2, gp, -1968
   10ba0: 07 34 06 00  	fld	fs0, 0(a2)
   10ba4: 13 86 81 85  	addi	a2, gp, -1960
   10ba8: 87 34 06 00  	fld	fs1, 0(a2)
   10bac: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10bb0: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10bb4: 53 16 24 a2  	flt.d	a2, fs0, ft2
   10bb8: d3 16 91 a2  	flt.d	a3, ft2, fs1
   10bbc: 33 66 d6 00  	or	a2, a2, a3
   10bc0: 9b 85 15 00  	addiw	a1, a1, 1
   10bc4: 23 a0 b9 00  	sw	a1, 0(s3)
   10bc8: 63 18 06 0c  	bnez	a2, 0x10c98 <main+0xab8>
   10bcc: 07 10 25 00  	flh	ft0, 2(a0)
   10bd0: 93 85 01 86  	addi	a1, gp, -1952
   10bd4: 87 90 05 00  	flh	ft1, 0(a1)
   10bd8: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10bdc: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10be0: d3 15 24 a2  	flt.d	a1, fs0, ft2
   10be4: 53 16 91 a2  	flt.d	a2, ft2, fs1
   10be8: b3 e5 c5 00  	or	a1, a1, a2
   10bec: 13 07 10 00  	addi	a4, zero, 1
   10bf0: 63 94 05 0a  	bnez	a1, 0x10c98 <main+0xab8>
   10bf4: 07 10 45 00  	flh	ft0, 4(a0)
   10bf8: 93 85 21 86  	addi	a1, gp, -1950
   10bfc: 87 90 05 00  	flh	ft1, 0(a1)
   10c00: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10c04: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10c08: d3 15 24 a2  	flt.d	a1, fs0, ft2
   10c0c: 53 16 91 a2  	flt.d	a2, ft2, fs1
   10c10: b3 e5 c5 00  	or	a1, a1, a2
   10c14: 13 07 20 00  	addi	a4, zero, 2
   10c18: 63 90 05 08  	bnez	a1, 0x10c98 <main+0xab8>
   10c1c: 07 10 65 00  	flh	ft0, 6(a0)
   10c20: 93 85 41 86  	addi	a1, gp, -1948
   10c24: 87 90 05 00  	flh	ft1, 0(a1)
   10c28: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10c2c: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10c30: d3 15 24 a2  	flt.d	a1, fs0, ft2
   10c34: 53 16 91 a2  	flt.d	a2, ft2, fs1
   10c38: b3 e5 c5 00  	or	a1, a1, a2
   10c3c: 13 07 30 00  	addi	a4, zero, 3
   10c40: 63 9c 05 04  	bnez	a1, 0x10c98 <main+0xab8>
   10c44: 07 10 85 00  	flh	ft0, 8(a0)
   10c48: 93 85 61 86  	addi	a1, gp, -1946
   10c4c: 87 90 05 00  	flh	ft1, 0(a1)
   10c50: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10c54: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10c58: d3 15 24 a2  	flt.d	a1, fs0, ft2
   10c5c: 53 16 91 a2  	flt.d	a2, ft2, fs1
   10c60: b3 e5 c5 00  	or	a1, a1, a2
   10c64: 13 07 40 00  	addi	a4, zero, 4
   10c68: 63 98 05 02  	bnez	a1, 0x10c98 <main+0xab8>
   10c6c: 07 10 a5 00  	flh	ft0, 10(a0)
   10c70: 13 85 81 86  	addi	a0, gp, -1944
   10c74: 87 10 05 00  	flh	ft1, 0(a0)
   10c78: 53 f1 00 0c  	fsub.h	ft2, ft1, ft0
   10c7c: 53 01 21 42  	fcvt.d.h	ft2, ft2
   10c80: 53 15 24 a2  	flt.d	a0, fs0, ft2
   10c84: d3 15 91 a2  	flt.d	a1, ft2, fs1
   10c88: 33 65 b5 00  	or	a0, a0, a1
   10c8c: 13 07 50 00  	addi	a4, zero, 5
   10c90: 63 14 05 00  	bnez	a0, 0x10c98 <main+0xab8>
   10c94: 6f 20 40 52  	j	0x131b8 <main+0x2fd8>
   10c98: d3 87 00 e4  	fmv.x.h	a5, ft1
   10c9c: 53 08 00 e4  	fmv.x.h	a6, ft0
   10ca0: 17 15 01 00  	auipc	a0, 17
   10ca4: 13 05 d5 96  	addi	a0, a0, -1683
   10ca8: 97 05 01 00  	auipc	a1, 16
   10cac: 93 85 95 7d  	addi	a1, a1, 2009
   10cb0: 97 16 01 00  	auipc	a3, 17
   10cb4: 93 86 e6 99  	addi	a3, a3, -1634
   10cb8: 13 06 f0 09  	addi	a2, zero, 159
   10cbc: ef 20 70 00  	jal	0x134c2 <printf>
   10cc0: 77 75 14 00  	msettypei	a0, e32, m1, ba
   10cc4: 77 75 01 20  	msettilemi	a0, 2
   10cc8: 77 75 02 40  	msettileki	a0, 4
   10ccc: 77 f5 01 60  	msettileni	a0, 3
   10cd0: 17 f5 00 00  	auipc	a0, 15
   10cd4: 13 05 c5 5d  	addi	a0, a0, 1500
   10cd8: 93 05 c0 00  	addi	a1, zero, 12
   10cdc: 77 20 b5 00  	mlce32.m	tr0, (a0), a1, m1
   10ce0: 17 15 01 00  	auipc	a0, 17
   10ce4: 13 05 05 c9  	addi	a0, a0, -880
   10ce8: 13 06 00 01  	addi	a2, zero, 16
   10cec: f7 20 c5 04  	mlae32.m	tr1, (a0), a2, m1
   10cf0: 17 f5 00 00  	auipc	a0, 15
   10cf4: 13 05 c5 58  	addi	a0, a0, 1420
   10cf8: 77 21 b5 08  	mlbe32.m	tr2, (a0), a1, m1
   10cfc: 77 e0 20 02  	mfma.mm	tr0, tr1, tr2
   10d00: 17 5b 01 00  	auipc	s6, 21
   10d04: 13 0b 0b ee  	addi	s6, s6, -288
   10d08: 77 20 bb 02  	msce32.m	tr0, (s6), a1, m1
   10d0c: 07 20 0b 00  	flw	ft0, 0(s6)
   10d10: 13 85 c1 86  	addi	a0, gp, -1940
   10d14: 87 20 05 00  	flw	ft1, 0(a0)
   10d18: 03 a5 09 00  	lw	a0, 0(s3)
   10d1c: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10d20: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10d24: d3 15 14 a2  	flt.d	a1, fs0, ft1
   10d28: 53 96 90 a2  	flt.d	a2, ft1, fs1
   10d2c: b3 e5 c5 00  	or	a1, a1, a2
   10d30: 1b 05 15 00  	addiw	a0, a0, 1
   10d34: 23 a0 a9 00  	sw	a0, 0(s3)
   10d38: 63 8a 05 00  	beqz	a1, 0x10d4c <main+0xb6c>
   10d3c: 13 85 01 87  	addi	a0, gp, -1936
   10d40: 87 30 05 00  	fld	ft1, 0(a0)
   10d44: 13 07 00 00  	mv	a4, zero
   10d48: 6f 00 80 10  	j	0x10e50 <main+0xc70>
   10d4c: 07 20 4b 00  	flw	ft0, 4(s6)
   10d50: 13 85 81 87  	addi	a0, gp, -1928
   10d54: 87 20 05 00  	flw	ft1, 0(a0)
   10d58: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10d5c: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10d60: 53 15 14 a2  	flt.d	a0, fs0, ft1
   10d64: d3 95 90 a2  	flt.d	a1, ft1, fs1
   10d68: 33 65 b5 00  	or	a0, a0, a1
   10d6c: 63 0a 05 00  	beqz	a0, 0x10d80 <main+0xba0>
   10d70: 13 85 01 88  	addi	a0, gp, -1920
   10d74: 87 30 05 00  	fld	ft1, 0(a0)
   10d78: 13 07 10 00  	addi	a4, zero, 1
   10d7c: 6f 00 40 0d  	j	0x10e50 <main+0xc70>
   10d80: 07 20 8b 00  	flw	ft0, 8(s6)
   10d84: 13 85 81 88  	addi	a0, gp, -1912
   10d88: 87 20 05 00  	flw	ft1, 0(a0)
   10d8c: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10d90: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10d94: 53 15 14 a2  	flt.d	a0, fs0, ft1
   10d98: d3 95 90 a2  	flt.d	a1, ft1, fs1
   10d9c: 33 65 b5 00  	or	a0, a0, a1
   10da0: 63 0a 05 00  	beqz	a0, 0x10db4 <main+0xbd4>
   10da4: 13 85 01 89  	addi	a0, gp, -1904
   10da8: 87 30 05 00  	fld	ft1, 0(a0)
   10dac: 13 07 20 00  	addi	a4, zero, 2
   10db0: 6f 00 00 0a  	j	0x10e50 <main+0xc70>
   10db4: 07 20 cb 00  	flw	ft0, 12(s6)
   10db8: 13 85 81 89  	addi	a0, gp, -1896
   10dbc: 87 20 05 00  	flw	ft1, 0(a0)
   10dc0: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10dc4: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10dc8: 53 15 14 a2  	flt.d	a0, fs0, ft1
   10dcc: d3 95 90 a2  	flt.d	a1, ft1, fs1
   10dd0: 33 65 b5 00  	or	a0, a0, a1
   10dd4: 63 0a 05 00  	beqz	a0, 0x10de8 <main+0xc08>
   10dd8: 13 85 01 8a  	addi	a0, gp, -1888
   10ddc: 87 30 05 00  	fld	ft1, 0(a0)
   10de0: 13 07 30 00  	addi	a4, zero, 3
   10de4: 6f 00 c0 06  	j	0x10e50 <main+0xc70>
   10de8: 07 20 0b 01  	flw	ft0, 16(s6)
   10dec: 13 85 81 8a  	addi	a0, gp, -1880
   10df0: 87 20 05 00  	flw	ft1, 0(a0)
   10df4: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10df8: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10dfc: 53 15 14 a2  	flt.d	a0, fs0, ft1
   10e00: d3 95 90 a2  	flt.d	a1, ft1, fs1
   10e04: 33 65 b5 00  	or	a0, a0, a1
   10e08: 63 0a 05 00  	beqz	a0, 0x10e1c <main+0xc3c>
   10e0c: 13 85 01 8b  	addi	a0, gp, -1872
   10e10: 87 30 05 00  	fld	ft1, 0(a0)
   10e14: 13 07 40 00  	addi	a4, zero, 4
   10e18: 6f 00 80 03  	j	0x10e50 <main+0xc70>
   10e1c: 07 20 4b 01  	flw	ft0, 20(s6)
   10e20: 13 85 81 8b  	addi	a0, gp, -1864
   10e24: 87 20 05 00  	flw	ft1, 0(a0)
   10e28: d3 f0 00 08  	fsub.s	ft1, ft1, ft0
   10e2c: d3 80 00 42  	fcvt.d.s	ft1, ft1
   10e30: 53 15 14 a2  	flt.d	a0, fs0, ft1
   10e34: d3 95 90 a2  	flt.d	a1, ft1, fs1
   10e38: 33 65 b5 00  	or	a0, a0, a1
   10e3c: 63 14 05 00  	bnez	a0, 0x10e44 <main+0xc64>
   10e40: 6f 20 c0 3a  	j	0x131ec <main+0x300c>
   10e44: 13 85 01 8c  	addi	a0, gp, -1856
   10e48: 87 30 05 00  	fld	ft1, 0(a0)
   10e4c: 13 07 50 00  	addi	a4, zero, 5
   10e50: 53 00 00 42  	fcvt.d.s	ft0, ft0
   10e54: d3 87 00 e2  	fmv.x.d	a5, ft1
   10e58: 53 08 00 e2  	fmv.x.d	a6, ft0
   10e5c: 17 05 01 00  	auipc	a0, 16
   10e60: 13 05 15 7b  	addi	a0, a0, 1969
   10e64: 97 05 01 00  	auipc	a1, 16
   10e68: 93 85 d5 61  	addi	a1, a1, 1565
   10e6c: 97 06 01 00  	auipc	a3, 16
   10e70: 93 86 66 7f  	addi	a3, a3, 2038
   10e74: 13 06 30 0b  	addi	a2, zero, 179
   10e78: ef 20 a0 64  	jal	0x134c2 <printf>
   10e7c: 77 75 16 00  	msettypei	a0, e64, m1, ba
   10e80: 77 f5 01 20  	msettilemi	a0, 3
   10e84: 77 75 01 40  	msettileki	a0, 2
   10e88: 77 75 01 60  	msettileni	a0, 2
   10e8c: 37 15 20 00  	lui	a0, 513
   10e90: 9b 05 b5 c0  	addiw	a1, a0, -1013
   10e94: 93 95 c5 00  	slli	a1, a1, 12
   10e98: 93 85 d5 26  	addi	a1, a1, 621
   10e9c: 93 95 f5 00  	slli	a1, a1, 15
   10ea0: 93 85 35 27  	addi	a1, a1, 627
   10ea4: 93 95 e5 00  	slli	a1, a1, 14
   10ea8: 93 85 a5 31  	addi	a1, a1, 794
   10eac: 23 34 b1 04  	sd	a1, 72(sp)
   10eb0: 1b 05 55 cd  	addiw	a0, a0, -811
   10eb4: 13 15 c5 00  	slli	a0, a0, 12
   10eb8: 13 05 75 0e  	addi	a0, a0, 231
   10ebc: 13 15 e5 00  	slli	a0, a0, 14
   10ec0: 13 05 75 77  	addi	a0, a0, 1911
   10ec4: 13 15 f5 00  	slli	a0, a0, 15
   10ec8: 13 05 25 bc  	addi	a0, a0, -1086
   10ecc: 23 38 a1 04  	sd	a0, 80(sp)
   10ed0: 37 25 80 ff  	lui	a0, 1046530
   10ed4: 1b 05 b5 3e  	addiw	a0, a0, 1003
   10ed8: 13 15 c5 00  	slli	a0, a0, 12
   10edc: 13 05 b5 d6  	addi	a0, a0, -661
   10ee0: 13 15 f5 00  	slli	a0, a0, 15
   10ee4: 13 05 b5 3d  	addi	a0, a0, 987
   10ee8: 13 15 c5 00  	slli	a0, a0, 12
   10eec: 13 05 75 eb  	addi	a0, a0, -329
   10ef0: 23 3c a1 04  	sd	a0, 88(sp)
   10ef4: 37 15 80 ff  	lui	a0, 1046529
   10ef8: 1b 05 55 09  	addiw	a0, a0, 149
   10efc: 13 15 c5 00  	slli	a0, a0, 12
   10f00: 13 05 15 2f  	addi	a0, a0, 753
   10f04: 13 15 d5 00  	slli	a0, a0, 13
   10f08: 13 05 55 27  	addi	a0, a0, 629
   10f0c: 13 15 e5 00  	slli	a0, a0, 14
   10f10: 13 05 a5 7f  	addi	a0, a0, 2042
   10f14: 23 30 a1 06  	sd	a0, 96(sp)
   10f18: 37 45 86 82  	lui	a0, 534628
   10f1c: 1b 05 95 8a  	addiw	a0, a0, -1879
   10f20: 13 15 c5 00  	slli	a0, a0, 12
   10f24: 13 05 d5 3e  	addi	a0, a0, 1005
   10f28: 13 15 c5 00  	slli	a0, a0, 12
   10f2c: 13 05 75 32  	addi	a0, a0, 807
   10f30: 13 55 25 00  	srli	a0, a0, 2
   10f34: 23 34 a1 06  	sd	a0, 104(sp)
   10f38: 37 05 f0 ff  	lui	a0, 1048320
   10f3c: 1b 05 35 5b  	addiw	a0, a0, 1459
   10f40: 13 15 c5 00  	slli	a0, a0, 12
   10f44: 13 05 55 a0  	addi	a0, a0, -1531
   10f48: 13 15 15 01  	slli	a0, a0, 17
   10f4c: 13 05 55 d8  	addi	a0, a0, -635
   10f50: 13 15 d5 00  	slli	a0, a0, 13
   10f54: 13 05 15 a1  	addi	a0, a0, -1519
   10f58: 23 38 a1 06  	sd	a0, 112(sp)
   10f5c: 37 35 00 01  	lui	a0, 4099
   10f60: 1b 05 75 21  	addiw	a0, a0, 535
   10f64: 13 15 d5 00  	slli	a0, a0, 13
   10f68: 13 05 d5 9e  	addi	a0, a0, -1555
   10f6c: 13 15 c5 00  	slli	a0, a0, 12
   10f70: 13 05 f5 9a  	addi	a0, a0, -1617
   10f74: 13 15 d5 00  	slli	a0, a0, 13
   10f78: 13 05 25 6c  	addi	a0, a0, 1730
   10f7c: 23 34 a1 00  	sd	a0, 8(sp)
   10f80: 37 85 01 04  	lui	a0, 16408
   10f84: 1b 05 f5 4d  	addiw	a0, a0, 1247
   10f88: 13 15 c5 00  	slli	a0, a0, 12
   10f8c: 13 05 95 8f  	addi	a0, a0, -1799
   10f90: 13 15 c5 00  	slli	a0, a0, 12
   10f94: 13 05 35 7e  	addi	a0, a0, 2019
   10f98: 13 15 c5 00  	slli	a0, a0, 12
   10f9c: 13 05 25 0b  	addi	a0, a0, 178
   10fa0: 23 38 a1 00  	sd	a0, 16(sp)
   10fa4: 37 35 80 00  	lui	a0, 2051
   10fa8: 1b 05 f5 fb  	addiw	a0, a0, -65
   10fac: 13 15 d5 00  	slli	a0, a0, 13
   10fb0: 13 05 b5 af  	addi	a0, a0, -1285
   10fb4: 13 15 e5 00  	slli	a0, a0, 14
   10fb8: 13 05 55 37  	addi	a0, a0, 885
   10fbc: 13 15 c5 00  	slli	a0, a0, 12
   10fc0: 13 05 65 b4  	addi	a0, a0, -1210
   10fc4: 23 3c a1 00  	sd	a0, 24(sp)
   10fc8: 37 95 00 ff  	lui	a0, 1044489
   10fcc: 1b 05 d5 d3  	addiw	a0, a0, -707
   10fd0: 13 15 d5 00  	slli	a0, a0, 13
   10fd4: 13 05 75 76  	addi	a0, a0, 1895
   10fd8: 13 15 d5 00  	slli	a0, a0, 13
   10fdc: 13 05 75 c3  	addi	a0, a0, -969
   10fe0: 13 15 c5 00  	slli	a0, a0, 12
   10fe4: 13 05 25 cd  	addi	a0, a0, -814
   10fe8: 23 30 a1 02  	sd	a0, 32(sp)
   10fec: 37 95 00 04  	lui	a0, 16393
   10ff0: 1b 05 75 d4  	addiw	a0, a0, -697
   10ff4: 13 15 c5 00  	slli	a0, a0, 12
   10ff8: 13 05 75 67  	addi	a0, a0, 1655
   10ffc: 13 15 c5 00  	slli	a0, a0, 12
   11000: 13 05 b5 2a  	addi	a0, a0, 683
   11004: 13 15 c5 00  	slli	a0, a0, 12
   11008: 13 05 55 ed  	addi	a0, a0, -299
   1100c: 23 34 a1 02  	sd	a0, 40(sp)
   11010: 37 05 c0 ff  	lui	a0, 1047552
   11014: 1b 05 d5 af  	addiw	a0, a0, -1283
   11018: 13 15 c5 00  	slli	a0, a0, 12
   1101c: 13 05 55 22  	addi	a0, a0, 549
   11020: 13 15 f5 00  	slli	a0, a0, 15
   11024: 13 05 15 db  	addi	a0, a0, -591
   11028: 13 15 d5 00  	slli	a0, a0, 13
   1102c: 13 05 95 0b  	addi	a0, a0, 185
   11030: 23 38 a1 02  	sd	a0, 48(sp)
   11034: 13 05 00 01  	addi	a0, zero, 16
   11038: 93 05 81 00  	addi	a1, sp, 8
   1103c: 77 b0 a5 00  	mlce64.m	tr0, (a1), a0, m1
   11040: 93 05 81 04  	addi	a1, sp, 72
   11044: f7 b0 a5 04  	mlae64.m	tr1, (a1), a0, m1
   11048: 97 15 01 00  	auipc	a1, 17
   1104c: 93 85 85 94  	addi	a1, a1, -1720
   11050: 77 b1 a5 08  	mlbe64.m	tr2, (a1), a0, m1
   11054: 77 e0 20 02  	mfma.mm	tr0, tr1, tr2
   11058: 17 5a 01 00  	auipc	s4, 21
   1105c: 13 0a 8a ea  	addi	s4, s4, -344
   11060: 77 30 aa 02  	msce64.m	tr0, (s4), a0, m1
   11064: 07 30 0a 00  	fld	ft0, 0(s4)
   11068: 13 85 81 8c  	addi	a0, gp, -1848
   1106c: 87 30 05 00  	fld	ft1, 0(a0)
   11070: 13 07 00 00  	mv	a4, zero
   11074: 03 a5 09 00  	lw	a0, 0(s3)
   11078: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   1107c: d3 15 24 a2  	flt.d	a1, fs0, ft2
   11080: 53 16 91 a2  	flt.d	a2, ft2, fs1
   11084: b3 e5 c5 00  	or	a1, a1, a2
   11088: 1b 05 15 00  	addiw	a0, a0, 1
   1108c: 23 a0 a9 00  	sw	a0, 0(s3)
   11090: 63 9e 05 0a  	bnez	a1, 0x1114c <main+0xf6c>
   11094: 07 30 8a 00  	fld	ft0, 8(s4)
   11098: 13 85 01 8d  	addi	a0, gp, -1840
   1109c: 87 30 05 00  	fld	ft1, 0(a0)
   110a0: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   110a4: 53 15 24 a2  	flt.d	a0, fs0, ft2
   110a8: d3 15 91 a2  	flt.d	a1, ft2, fs1
   110ac: 33 65 b5 00  	or	a0, a0, a1
   110b0: 13 07 10 00  	addi	a4, zero, 1
   110b4: 63 1c 05 08  	bnez	a0, 0x1114c <main+0xf6c>
   110b8: 07 30 0a 01  	fld	ft0, 16(s4)
   110bc: 13 85 81 8d  	addi	a0, gp, -1832
   110c0: 87 30 05 00  	fld	ft1, 0(a0)
   110c4: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   110c8: 53 15 24 a2  	flt.d	a0, fs0, ft2
   110cc: d3 15 91 a2  	flt.d	a1, ft2, fs1
   110d0: 33 65 b5 00  	or	a0, a0, a1
   110d4: 13 07 20 00  	addi	a4, zero, 2
   110d8: 63 1a 05 06  	bnez	a0, 0x1114c <main+0xf6c>
   110dc: 07 30 8a 01  	fld	ft0, 24(s4)
   110e0: 13 85 01 8e  	addi	a0, gp, -1824
   110e4: 87 30 05 00  	fld	ft1, 0(a0)
   110e8: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   110ec: 53 15 24 a2  	flt.d	a0, fs0, ft2
   110f0: d3 15 91 a2  	flt.d	a1, ft2, fs1
   110f4: 33 65 b5 00  	or	a0, a0, a1
   110f8: 13 07 30 00  	addi	a4, zero, 3
   110fc: 63 18 05 04  	bnez	a0, 0x1114c <main+0xf6c>
   11100: 07 30 0a 02  	fld	ft0, 32(s4)
   11104: 13 85 81 8e  	addi	a0, gp, -1816
   11108: 87 30 05 00  	fld	ft1, 0(a0)
   1110c: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   11110: 53 15 24 a2  	flt.d	a0, fs0, ft2
   11114: d3 15 91 a2  	flt.d	a1, ft2, fs1
   11118: 33 65 b5 00  	or	a0, a0, a1
   1111c: 13 07 40 00  	addi	a4, zero, 4
   11120: 63 16 05 02  	bnez	a0, 0x1114c <main+0xf6c>
   11124: 07 30 8a 02  	fld	ft0, 40(s4)
   11128: 13 85 01 8f  	addi	a0, gp, -1808
   1112c: 87 30 05 00  	fld	ft1, 0(a0)
   11130: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   11134: 53 15 24 a2  	flt.d	a0, fs0, ft2
   11138: d3 15 91 a2  	flt.d	a1, ft2, fs1
   1113c: 33 65 b5 00  	or	a0, a0, a1
   11140: 13 07 50 00  	addi	a4, zero, 5
   11144: 63 14 05 00  	bnez	a0, 0x1114c <main+0xf6c>
   11148: 6f 20 80 0d  	j	0x13220 <main+0x3040>
   1114c: d3 87 00 e2  	fmv.x.d	a5, ft1
   11150: 53 08 00 e2  	fmv.x.d	a6, ft0
   11154: 17 05 01 00  	auipc	a0, 16
   11158: 13 05 95 4b  	addi	a0, a0, 1209
   1115c: 97 05 01 00  	auipc	a1, 16
   11160: 93 85 55 32  	addi	a1, a1, 805
   11164: 97 06 01 00  	auipc	a3, 16
   11168: 93 86 26 51  	addi	a3, a3, 1298
   1116c: 13 06 50 0c  	addi	a2, zero, 197
   11170: ef 20 20 35  	jal	0x134c2 <printf>
   11174: 77 75 10 00  	msettypei	a0, e8, m1, ba
   11178: 77 75 01 20  	msettilemi	a0, 2
   1117c: 77 f5 02 40  	msettileki	a0, 5
   11180: 77 75 08 60  	msettileni	a0, 16
   11184: 17 f5 00 00  	auipc	a0, 15
   11188: 13 05 05 14  	addi	a0, a0, 320
   1118c: 93 05 50 00  	addi	a1, zero, 5
   11190: 77 00 b5 04  	mlae8.m	tr0, (a0), a1, m1
   11194: 17 f5 00 00  	auipc	a0, 15
   11198: 13 05 a5 13  	addi	a0, a0, 314
   1119c: 93 05 00 01  	addi	a1, zero, 16
   111a0: f7 00 b5 08  	mlbe8.m	tr1, (a0), a1, m1
   111a4: 17 15 01 00  	auipc	a0, 17
   111a8: 13 05 c5 80  	addi	a0, a0, -2036
   111ac: 77 01 b5 00  	mlce8.m	tr2, (a0), a1, m1
   111b0: 77 61 1c 00  	msma.mm	tr2, tr0, tr1
   111b4: 77 81 bc 02  	msce8.m	tr2, (s9), a1, m1
   111b8: 83 a5 09 00  	lw	a1, 0(s3)
   111bc: 13 05 00 00  	mv	a0, zero
   111c0: 9b 85 15 00  	addiw	a1, a1, 1
   111c4: 23 a0 b9 00  	sw	a1, 0(s3)
   111c8: 97 15 01 00  	auipc	a1, 17
   111cc: 93 85 85 80  	addi	a1, a1, -2040
   111d0: 13 06 00 02  	addi	a2, zero, 32
   111d4: b3 06 b5 00  	add	a3, a0, a1
   111d8: 83 c6 06 00  	lbu	a3, 0(a3)
   111dc: 33 07 95 01  	add	a4, a0, s9
   111e0: 83 44 07 00  	lbu	s1, 0(a4)
   111e4: 63 90 96 04  	bne	a3, s1, 0x11224 <main+0x1044>
   111e8: 13 05 15 00  	addi	a0, a0, 1
   111ec: e3 14 c5 fe  	bne	a0, a2, 0x111d4 <main+0xff4>
   111f0: 13 85 41 95  	addi	a0, gp, -1708
   111f4: 83 25 05 00  	lw	a1, 0(a0)
   111f8: 9b 85 15 00  	addiw	a1, a1, 1
   111fc: 23 20 b5 00  	sw	a1, 0(a0)
   11200: 17 05 01 00  	auipc	a0, 16
   11204: 13 05 55 2a  	addi	a0, a0, 677
   11208: 97 05 01 00  	auipc	a1, 16
   1120c: 93 85 95 27  	addi	a1, a1, 633
   11210: 97 06 01 00  	auipc	a3, 16
   11214: 93 86 a6 47  	addi	a3, a3, 1146
   11218: 13 06 40 0e  	addi	a2, zero, 228
   1121c: ef 20 60 2a  	jal	0x134c2 <printf>
   11220: 6f 00 80 03  	j	0x11258 <main+0x1078>
   11224: 1b 07 05 00  	sext.w	a4, a0
   11228: 13 95 86 03  	slli	a0, a3, 56
   1122c: 93 57 85 43  	srai	a5, a0, 56
   11230: 13 95 84 03  	slli	a0, s1, 56
   11234: 13 58 85 43  	srai	a6, a0, 56
   11238: 17 05 01 00  	auipc	a0, 16
   1123c: 13 05 85 20  	addi	a0, a0, 520
   11240: 97 05 01 00  	auipc	a1, 16
   11244: 93 85 15 24  	addi	a1, a1, 577
   11248: 97 06 01 00  	auipc	a3, 16
   1124c: 93 86 26 44  	addi	a3, a3, 1090
   11250: 13 06 40 0e  	addi	a2, zero, 228
   11254: ef 20 e0 26  	jal	0x134c2 <printf>
   11258: 77 75 12 00  	msettypei	a0, e16, m1, ba
   1125c: 77 f5 01 20  	msettilemi	a0, 3
   11260: 77 75 02 40  	msettileki	a0, 4
   11264: 77 75 04 60  	msettileni	a0, 8
   11268: 17 f5 00 00  	auipc	a0, 15
   1126c: 13 05 65 0b  	addi	a0, a0, 182
   11270: 93 05 80 00  	addi	a1, zero, 8
   11274: 77 10 b5 04  	mlae16.m	tr0, (a0), a1, m1
   11278: 17 f5 00 00  	auipc	a0, 15
   1127c: 13 05 e5 0b  	addi	a0, a0, 190
   11280: 93 05 00 01  	addi	a1, zero, 16
   11284: f7 10 b5 08  	mlbe16.m	tr1, (a0), a1, m1
   11288: 17 f5 00 00  	auipc	a0, 15
   1128c: 13 05 e5 0e  	addi	a0, a0, 238
   11290: 77 11 b5 00  	mlce16.m	tr2, (a0), a1, m1
   11294: 77 61 1c 00  	msma.mm	tr2, tr0, tr1
   11298: 77 11 b4 02  	msce16.m	tr2, (s0), a1, m1
   1129c: 83 a5 09 00  	lw	a1, 0(s3)
   112a0: 13 05 00 00  	mv	a0, zero
   112a4: 9b 85 15 00  	addiw	a1, a1, 1
   112a8: 23 a0 b9 00  	sw	a1, 0(s3)
   112ac: 97 f5 00 00  	auipc	a1, 15
   112b0: 93 85 a5 0f  	addi	a1, a1, 250
   112b4: 13 06 80 01  	addi	a2, zero, 24
   112b8: 83 d7 05 00  	lhu	a5, 0(a1)
   112bc: 83 56 04 00  	lhu	a3, 0(s0)
   112c0: 63 94 d7 04  	bne	a5, a3, 0x11308 <main+0x1128>
   112c4: 13 05 15 00  	addi	a0, a0, 1
   112c8: 13 04 24 00  	addi	s0, s0, 2
   112cc: 93 85 25 00  	addi	a1, a1, 2
   112d0: e3 14 c5 fe  	bne	a0, a2, 0x112b8 <main+0x10d8>
   112d4: 13 85 41 95  	addi	a0, gp, -1708
   112d8: 83 25 05 00  	lw	a1, 0(a0)
   112dc: 9b 85 15 00  	addiw	a1, a1, 1
   112e0: 23 20 b5 00  	sw	a1, 0(a0)
   112e4: 17 05 01 00  	auipc	a0, 16
   112e8: 13 05 15 1c  	addi	a0, a0, 449
   112ec: 97 05 01 00  	auipc	a1, 16
   112f0: 93 85 55 19  	addi	a1, a1, 405
   112f4: 97 06 01 00  	auipc	a3, 16
   112f8: 93 86 a6 3a  	addi	a3, a3, 938
   112fc: 13 06 e0 0f  	addi	a2, zero, 254
   11300: ef 20 20 1c  	jal	0x134c2 <printf>
   11304: 6f 00 80 03  	j	0x1133c <main+0x115c>
   11308: 1b 07 05 00  	sext.w	a4, a0
   1130c: 13 95 07 03  	slli	a0, a5, 48
   11310: 93 57 05 43  	srai	a5, a0, 48
   11314: 13 95 06 03  	slli	a0, a3, 48
   11318: 13 58 05 43  	srai	a6, a0, 48
   1131c: 17 05 01 00  	auipc	a0, 16
   11320: 13 05 45 12  	addi	a0, a0, 292
   11324: 97 05 01 00  	auipc	a1, 16
   11328: 93 85 d5 15  	addi	a1, a1, 349
   1132c: 97 06 01 00  	auipc	a3, 16
   11330: 93 86 26 37  	addi	a3, a3, 882
   11334: 13 06 e0 0f  	addi	a2, zero, 254
   11338: ef 20 a0 18  	jal	0x134c2 <printf>
   1133c: 77 75 14 00  	msettypei	a0, e32, m1, ba
   11340: 77 75 02 20  	msettilemi	a0, 4
   11344: 77 f5 01 40  	msettileki	a0, 3
   11348: 77 75 02 60  	msettileni	a0, 4
   1134c: 23 30 01 08  	sd	zero, 128(sp)
   11350: 23 3c 01 06  	sd	zero, 120(sp)
   11354: 23 38 01 06  	sd	zero, 112(sp)
   11358: 23 34 01 06  	sd	zero, 104(sp)
   1135c: 23 30 01 06  	sd	zero, 96(sp)
   11360: 23 3c 01 04  	sd	zero, 88(sp)
   11364: 23 38 01 04  	sd	zero, 80(sp)
   11368: 23 34 01 04  	sd	zero, 72(sp)
   1136c: 17 f5 00 00  	auipc	a0, 15
   11370: 13 05 c5 06  	addi	a0, a0, 108
   11374: 93 05 c0 00  	addi	a1, zero, 12
   11378: 77 20 b5 04  	mlae32.m	tr0, (a0), a1, m1
   1137c: 17 f5 00 00  	auipc	a0, 15
   11380: 13 05 c5 08  	addi	a0, a0, 140
   11384: 93 05 00 01  	addi	a1, zero, 16
   11388: f7 20 b5 08  	mlbe32.m	tr1, (a0), a1, m1
   1138c: 13 05 81 04  	addi	a0, sp, 72
   11390: 77 21 b5 00  	mlce32.m	tr2, (a0), a1, m1
   11394: 77 61 1c 00  	msma.mm	tr2, tr0, tr1
   11398: 77 21 bc 02  	msce32.m	tr2, (s8), a1, m1
   1139c: 03 a5 09 00  	lw	a0, 0(s3)
   113a0: 9b 05 15 00  	addiw	a1, a0, 1
   113a4: 03 65 0c 00  	lwu	a0, 0(s8)
   113a8: 23 a0 b9 00  	sw	a1, 0(s3)
   113ac: 13 07 10 00  	addi	a4, zero, 1
   113b0: 93 16 f7 01  	slli	a3, a4, 31
   113b4: b7 05 00 80  	lui	a1, 524288
   113b8: 63 1e d5 12  	bne	a0, a3, 0x114f4 <main+0x1314>
   113bc: 13 86 81 c1  	addi	a2, gp, -1000
   113c0: 03 65 46 00  	lwu	a0, 4(a2)
   113c4: 63 1a d5 12  	bne	a0, a3, 0x114f8 <main+0x1318>
   113c8: 03 65 86 00  	lwu	a0, 8(a2)
   113cc: b7 06 00 80  	lui	a3, 524288
   113d0: 9b 85 f6 ff  	addiw	a1, a3, -1
   113d4: 13 07 20 00  	addi	a4, zero, 2
   113d8: 63 10 b5 12  	bne	a0, a1, 0x114f8 <main+0x1318>
   113dc: 03 65 c6 00  	lwu	a0, 12(a2)
   113e0: 9b 85 f6 ff  	addiw	a1, a3, -1
   113e4: 13 07 30 00  	addi	a4, zero, 3
   113e8: 63 18 b5 10  	bne	a0, a1, 0x114f8 <main+0x1318>
   113ec: 03 65 06 01  	lwu	a0, 16(a2)
   113f0: b7 06 00 80  	lui	a3, 524288
   113f4: 9b 85 f6 ff  	addiw	a1, a3, -1
   113f8: 13 07 40 00  	addi	a4, zero, 4
   113fc: 63 1e b5 0e  	bne	a0, a1, 0x114f8 <main+0x1318>
   11400: 03 65 46 01  	lwu	a0, 20(a2)
   11404: 9b 85 f6 ff  	addiw	a1, a3, -1
   11408: 13 07 50 00  	addi	a4, zero, 5
   1140c: 63 16 b5 0e  	bne	a0, a1, 0x114f8 <main+0x1318>
   11410: 03 65 86 01  	lwu	a0, 24(a2)
   11414: b7 05 00 80  	lui	a1, 524288
   11418: 93 06 10 00  	addi	a3, zero, 1
   1141c: 93 96 f6 01  	slli	a3, a3, 31
   11420: 13 07 60 00  	addi	a4, zero, 6
   11424: 63 1a d5 0c  	bne	a0, a3, 0x114f8 <main+0x1318>
   11428: 03 65 c6 01  	lwu	a0, 28(a2)
   1142c: 13 07 70 00  	addi	a4, zero, 7
   11430: 63 14 d5 0c  	bne	a0, a3, 0x114f8 <main+0x1318>
   11434: 03 65 06 02  	lwu	a0, 32(a2)
   11438: b7 06 00 80  	lui	a3, 524288
   1143c: 9b 85 f6 ff  	addiw	a1, a3, -1
   11440: 13 07 80 00  	addi	a4, zero, 8
   11444: 63 1a b5 0a  	bne	a0, a1, 0x114f8 <main+0x1318>
   11448: 03 65 46 02  	lwu	a0, 36(a2)
   1144c: 9b 85 f6 ff  	addiw	a1, a3, -1
   11450: 13 07 90 00  	addi	a4, zero, 9
   11454: 63 12 b5 0a  	bne	a0, a1, 0x114f8 <main+0x1318>
   11458: 03 65 86 02  	lwu	a0, 40(a2)
   1145c: b7 05 00 80  	lui	a1, 524288
   11460: 93 06 10 00  	addi	a3, zero, 1
   11464: 93 96 f6 01  	slli	a3, a3, 31
   11468: 13 07 a0 00  	addi	a4, zero, 10
   1146c: 63 16 d5 08  	bne	a0, a3, 0x114f8 <main+0x1318>
   11470: 03 65 c6 02  	lwu	a0, 44(a2)
   11474: 13 07 b0 00  	addi	a4, zero, 11
   11478: 63 10 d5 08  	bne	a0, a3, 0x114f8 <main+0x1318>
   1147c: 03 65 06 03  	lwu	a0, 48(a2)
   11480: 93 06 10 00  	addi	a3, zero, 1
   11484: 93 96 f6 01  	slli	a3, a3, 31
   11488: 13 07 c0 00  	addi	a4, zero, 12
   1148c: 63 16 d5 06  	bne	a0, a3, 0x114f8 <main+0x1318>
   11490: 03 65 46 03  	lwu	a0, 52(a2)
   11494: 13 07 d0 00  	addi	a4, zero, 13
   11498: 63 10 d5 06  	bne	a0, a3, 0x114f8 <main+0x1318>
   1149c: 03 65 86 03  	lwu	a0, 56(a2)
   114a0: b7 06 00 80  	lui	a3, 524288
   114a4: 9b 85 f6 ff  	addiw	a1, a3, -1
   114a8: 13 07 e0 00  	addi	a4, zero, 14
   114ac: 63 16 b5 04  	bne	a0, a1, 0x114f8 <main+0x1318>
   114b0: 03 65 c6 03  	lwu	a0, 60(a2)
   114b4: 9b 85 f6 ff  	addiw	a1, a3, -1
   114b8: 13 07 f0 00  	addi	a4, zero, 15
   114bc: 63 1e b5 02  	bne	a0, a1, 0x114f8 <main+0x1318>
   114c0: 13 85 41 95  	addi	a0, gp, -1708
   114c4: 83 25 05 00  	lw	a1, 0(a0)
   114c8: 9b 85 15 00  	addiw	a1, a1, 1
   114cc: 23 20 b5 00  	sw	a1, 0(a0)
   114d0: 17 05 01 00  	auipc	a0, 16
   114d4: 13 05 55 fd  	addi	a0, a0, -43
   114d8: 97 05 01 00  	auipc	a1, 16
   114dc: 93 85 95 fa  	addi	a1, a1, -87
   114e0: 97 06 01 00  	auipc	a3, 16
   114e4: 93 86 26 1d  	addi	a3, a3, 466
   114e8: 13 06 50 11  	addi	a2, zero, 277
   114ec: ef 10 70 7d  	jal	0x134c2 <printf>
   114f0: 6f 00 00 03  	j	0x11520 <main+0x1340>
   114f4: 13 07 00 00  	mv	a4, zero
   114f8: 9b 87 05 00  	sext.w	a5, a1
   114fc: 1b 08 05 00  	sext.w	a6, a0
   11500: 17 05 01 00  	auipc	a0, 16
   11504: 13 05 05 f4  	addi	a0, a0, -192
   11508: 97 05 01 00  	auipc	a1, 16
   1150c: 93 85 95 f7  	addi	a1, a1, -135
   11510: 97 06 01 00  	auipc	a3, 16
   11514: 93 86 26 1a  	addi	a3, a3, 418
   11518: 13 06 50 11  	addi	a2, zero, 277
   1151c: ef 10 70 7a  	jal	0x134c2 <printf>
   11520: 77 75 16 00  	msettypei	a0, e64, m1, ba
   11524: 77 75 01 20  	msettilemi	a0, 2
   11528: 77 75 01 40  	msettileki	a0, 2
   1152c: 77 75 01 60  	msettileni	a0, 2
   11530: 23 30 01 02  	sd	zero, 32(sp)
   11534: 23 3c 01 00  	sd	zero, 24(sp)
   11538: 23 38 01 00  	sd	zero, 16(sp)
   1153c: 23 34 01 00  	sd	zero, 8(sp)
   11540: 17 05 01 00  	auipc	a0, 16
   11544: 13 05 05 4b  	addi	a0, a0, 1200
   11548: 93 05 00 01  	addi	a1, zero, 16
   1154c: 77 30 b5 04  	mlae64.m	tr0, (a0), a1, m1
   11550: 17 05 01 00  	auipc	a0, 16
   11554: 13 05 05 4c  	addi	a0, a0, 1216
   11558: f7 30 b5 08  	mlbe64.m	tr1, (a0), a1, m1
   1155c: 13 05 81 00  	addi	a0, sp, 8
   11560: 77 31 b5 00  	mlce64.m	tr2, (a0), a1, m1
   11564: 77 61 1c 00  	msma.mm	tr2, tr0, tr1
   11568: 77 31 b9 02  	msce64.m	tr2, (s2), a1, m1
   1156c: 03 a5 09 00  	lw	a0, 0(s3)
   11570: 13 07 00 00  	mv	a4, zero
   11574: 03 38 09 00  	ld	a6, 0(s2)
   11578: 1b 05 15 00  	addiw	a0, a0, 1
   1157c: 93 05 f0 ff  	addi	a1, zero, -1
   11580: 93 97 f5 03  	slli	a5, a1, 63
   11584: 23 a0 a9 00  	sw	a0, 0(s3)
   11588: 63 12 f8 06  	bne	a6, a5, 0x115ec <main+0x140c>
   1158c: 03 38 89 00  	ld	a6, 8(s2)
   11590: 13 07 10 00  	addi	a4, zero, 1
   11594: 63 1c f8 04  	bne	a6, a5, 0x115ec <main+0x140c>
   11598: 03 38 09 01  	ld	a6, 16(s2)
   1159c: 13 05 f0 ff  	addi	a0, zero, -1
   115a0: 13 15 f5 03  	slli	a0, a0, 63
   115a4: 13 07 20 00  	addi	a4, zero, 2
   115a8: 63 12 a8 04  	bne	a6, a0, 0x115ec <main+0x140c>
   115ac: 03 38 89 01  	ld	a6, 24(s2)
   115b0: 13 07 30 00  	addi	a4, zero, 3
   115b4: 63 1c a8 02  	bne	a6, a0, 0x115ec <main+0x140c>
   115b8: 13 85 41 95  	addi	a0, gp, -1708
   115bc: 83 25 05 00  	lw	a1, 0(a0)
   115c0: 9b 85 15 00  	addiw	a1, a1, 1
   115c4: 23 20 b5 00  	sw	a1, 0(a0)
   115c8: 17 05 01 00  	auipc	a0, 16
   115cc: 13 05 d5 ed  	addi	a0, a0, -291
   115d0: 97 05 01 00  	auipc	a1, 16
   115d4: 93 85 15 eb  	addi	a1, a1, -335
   115d8: 97 06 01 00  	auipc	a3, 16
   115dc: 93 86 e6 0e  	addi	a3, a3, 238
   115e0: 13 06 70 12  	addi	a2, zero, 295
   115e4: ef 10 f0 6d  	jal	0x134c2 <printf>
   115e8: 6f 00 40 02  	j	0x1160c <main+0x142c>
   115ec: 17 05 01 00  	auipc	a0, 16
   115f0: 13 05 a5 fc  	addi	a0, a0, -54
   115f4: 97 05 01 00  	auipc	a1, 16
   115f8: 93 85 d5 e8  	addi	a1, a1, -371
   115fc: 97 06 01 00  	auipc	a3, 16
   11600: 93 86 a6 0c  	addi	a3, a3, 202
   11604: 13 06 70 12  	addi	a2, zero, 295
   11608: ef 10 b0 6b  	jal	0x134c2 <printf>
   1160c: 77 75 10 00  	msettypei	a0, e8, m1, ba
   11610: 77 75 01 20  	msettilemi	a0, 2
   11614: 77 f5 02 40  	msettileki	a0, 5
   11618: 77 75 08 60  	msettileni	a0, 16
   1161c: 17 f5 00 00  	auipc	a0, 15
   11620: 13 05 c5 e1  	addi	a0, a0, -484
   11624: 93 05 50 00  	addi	a1, zero, 5
   11628: 77 00 b5 04  	mlae8.m	tr0, (a0), a1, m1
   1162c: 17 f5 00 00  	auipc	a0, 15
   11630: 13 05 65 e1  	addi	a0, a0, -490
   11634: 93 05 00 01  	addi	a1, zero, 16
   11638: f7 00 b5 08  	mlbe8.m	tr1, (a0), a1, m1
   1163c: 17 05 01 00  	auipc	a0, 16
   11640: 13 05 45 3f  	addi	a0, a0, 1012
   11644: 77 01 b5 00  	mlce8.m	tr2, (a0), a1, m1
   11648: 77 61 14 00  	msmau.mm	tr2, tr0, tr1
   1164c: 77 81 bb 02  	msce8.m	tr2, (s7), a1, m1
   11650: 83 a5 09 00  	lw	a1, 0(s3)
   11654: 13 05 00 00  	mv	a0, zero
   11658: 9b 85 15 00  	addiw	a1, a1, 1
   1165c: 23 a0 b9 00  	sw	a1, 0(s3)
   11660: 97 05 01 00  	auipc	a1, 16
   11664: 93 85 05 3f  	addi	a1, a1, 1008
   11668: 13 06 00 02  	addi	a2, zero, 32
   1166c: b3 06 b5 00  	add	a3, a0, a1
   11670: 83 c7 06 00  	lbu	a5, 0(a3)
   11674: b3 06 75 01  	add	a3, a0, s7
   11678: 03 c8 06 00  	lbu	a6, 0(a3)
   1167c: 63 90 07 05  	bne	a5, a6, 0x116bc <main+0x14dc>
   11680: 13 05 15 00  	addi	a0, a0, 1
   11684: e3 14 c5 fe  	bne	a0, a2, 0x1166c <main+0x148c>
   11688: 13 85 41 95  	addi	a0, gp, -1708
   1168c: 83 25 05 00  	lw	a1, 0(a0)
   11690: 9b 85 15 00  	addiw	a1, a1, 1
   11694: 23 20 b5 00  	sw	a1, 0(a0)
   11698: 17 05 01 00  	auipc	a0, 16
   1169c: 13 05 d5 e0  	addi	a0, a0, -499
   116a0: 97 05 01 00  	auipc	a1, 16
   116a4: 93 85 15 de  	addi	a1, a1, -543
   116a8: 97 06 01 00  	auipc	a3, 16
   116ac: 93 86 26 03  	addi	a3, a3, 50
   116b0: 13 06 60 14  	addi	a2, zero, 326
   116b4: ef 10 f0 60  	jal	0x134c2 <printf>
   116b8: 6f 00 80 02  	j	0x116e0 <main+0x1500>
   116bc: 1b 07 05 00  	sext.w	a4, a0
   116c0: 17 05 01 00  	auipc	a0, 16
   116c4: 13 05 05 d8  	addi	a0, a0, -640
   116c8: 97 05 01 00  	auipc	a1, 16
   116cc: 93 85 95 db  	addi	a1, a1, -583
   116d0: 97 06 01 00  	auipc	a3, 16
   116d4: 93 86 a6 00  	addi	a3, a3, 10
   116d8: 13 06 60 14  	addi	a2, zero, 326
   116dc: ef 10 70 5e  	jal	0x134c2 <printf>
   116e0: 77 75 12 00  	msettypei	a0, e16, m1, ba
   116e4: 77 f5 01 20  	msettilemi	a0, 3
   116e8: 77 75 02 40  	msettileki	a0, 4
   116ec: 77 75 04 60  	msettileni	a0, 8
   116f0: 13 05 f0 ff  	addi	a0, zero, -1
   116f4: 23 38 a1 06  	sd	a0, 112(sp)
   116f8: 23 34 a1 06  	sd	a0, 104(sp)
   116fc: 23 30 a1 06  	sd	a0, 96(sp)
   11700: 23 3c a1 04  	sd	a0, 88(sp)
   11704: 23 38 a1 04  	sd	a0, 80(sp)
   11708: 23 34 a1 04  	sd	a0, 72(sp)
   1170c: 17 f5 00 00  	auipc	a0, 15
   11710: 13 05 65 d8  	addi	a0, a0, -634
   11714: 93 05 80 00  	addi	a1, zero, 8
   11718: 77 10 b5 04  	mlae16.m	tr0, (a0), a1, m1
   1171c: 17 f5 00 00  	auipc	a0, 15
   11720: 13 05 e5 d8  	addi	a0, a0, -626
   11724: 93 05 00 01  	addi	a1, zero, 16
   11728: f7 10 b5 08  	mlbe16.m	tr1, (a0), a1, m1
   1172c: 17 f5 00 00  	auipc	a0, 15
   11730: 13 05 e5 db  	addi	a0, a0, -578
   11734: 77 11 b5 00  	mlce16.m	tr2, (a0), a1, m1
   11738: 77 61 14 00  	msmau.mm	tr2, tr0, tr1
   1173c: 77 91 bd 02  	msce16.m	tr2, (s11), a1, m1
   11740: 83 a5 09 00  	lw	a1, 0(s3)
   11744: 13 05 00 00  	mv	a0, zero
   11748: 9b 85 15 00  	addiw	a1, a1, 1
   1174c: 23 a0 b9 00  	sw	a1, 0(s3)
   11750: 93 05 81 04  	addi	a1, sp, 72
   11754: 13 06 80 01  	addi	a2, zero, 24
   11758: 83 d7 05 00  	lhu	a5, 0(a1)
   1175c: 03 d8 0d 00  	lhu	a6, 0(s11)
   11760: 63 94 07 05  	bne	a5, a6, 0x117a8 <main+0x15c8>
   11764: 13 05 15 00  	addi	a0, a0, 1
   11768: 93 8d 2d 00  	addi	s11, s11, 2
   1176c: 93 85 25 00  	addi	a1, a1, 2
   11770: e3 14 c5 fe  	bne	a0, a2, 0x11758 <main+0x1578>
   11774: 13 85 41 95  	addi	a0, gp, -1708
   11778: 83 25 05 00  	lw	a1, 0(a0)
   1177c: 9b 85 15 00  	addiw	a1, a1, 1
   11780: 23 20 b5 00  	sw	a1, 0(a0)
   11784: 17 05 01 00  	auipc	a0, 16
   11788: 13 05 15 d2  	addi	a0, a0, -735
   1178c: 97 05 01 00  	auipc	a1, 16
   11790: 93 85 55 cf  	addi	a1, a1, -779
   11794: 97 06 01 00  	auipc	a3, 16
   11798: 93 86 a6 f5  	addi	a3, a3, -166
   1179c: 13 06 00 16  	addi	a2, zero, 352
   117a0: ef 10 30 52  	jal	0x134c2 <printf>
   117a4: 6f 00 80 02  	j	0x117cc <main+0x15ec>
   117a8: 1b 07 05 00  	sext.w	a4, a0
   117ac: 17 05 01 00  	auipc	a0, 16
   117b0: 13 05 a5 d0  	addi	a0, a0, -758
   117b4: 97 05 01 00  	auipc	a1, 16
   117b8: 93 85 d5 cc  	addi	a1, a1, -819
   117bc: 97 06 01 00  	auipc	a3, 16
   117c0: 93 86 26 f3  	addi	a3, a3, -206
   117c4: 13 06 00 16  	addi	a2, zero, 352
   117c8: ef 10 b0 4f  	jal	0x134c2 <printf>
   117cc: 77 75 14 00  	msettypei	a0, e32, m1, ba
   117d0: 77 75 02 20  	msettilemi	a0, 4
   117d4: 77 f5 01 40  	msettileki	a0, 3
   117d8: 77 75 02 60  	msettileni	a0, 4
   117dc: 17 f5 00 00  	auipc	a0, 15
   117e0: 13 05 05 d4  	addi	a0, a0, -704
   117e4: 93 05 c0 00  	addi	a1, zero, 12
   117e8: 77 20 b5 04  	mlae32.m	tr0, (a0), a1, m1
   117ec: 17 f5 00 00  	auipc	a0, 15
   117f0: 13 05 05 d6  	addi	a0, a0, -672
   117f4: 93 05 00 01  	addi	a1, zero, 16
   117f8: f7 20 b5 08  	mlbe32.m	tr1, (a0), a1, m1
   117fc: 17 f5 00 00  	auipc	a0, 15
   11800: 13 05 05 d8  	addi	a0, a0, -640
   11804: 77 21 b5 00  	mlce32.m	tr2, (a0), a1, m1
   11808: 77 61 14 00  	msmau.mm	tr2, tr0, tr1
   1180c: 77 21 bd 02  	msce32.m	tr2, (s10), a1, m1
   11810: 83 a5 09 00  	lw	a1, 0(s3)
   11814: 03 65 0d 00  	lwu	a0, 0(s10)
   11818: 9b 85 15 00  	addiw	a1, a1, 1
   1181c: 13 06 f0 ff  	addi	a2, zero, -1
   11820: 13 56 06 02  	srli	a2, a2, 32
   11824: 23 a0 b9 00  	sw	a1, 0(s3)
   11828: 63 06 c5 00  	beq	a0, a2, 0x11834 <main+0x1654>
   1182c: 13 07 00 00  	mv	a4, zero
   11830: 6f 00 c0 0b  	j	0x118ec <main+0x170c>
   11834: 97 45 01 00  	auipc	a1, 20
   11838: 93 85 c5 8b  	addi	a1, a1, -1860
   1183c: 03 e5 45 00  	lwu	a0, 4(a1)
   11840: 13 07 10 00  	addi	a4, zero, 1
   11844: 63 14 c5 0a  	bne	a0, a2, 0x118ec <main+0x170c>
   11848: 03 e5 85 00  	lwu	a0, 8(a1)
   1184c: 13 06 f0 ff  	addi	a2, zero, -1
   11850: 13 56 06 02  	srli	a2, a2, 32
   11854: 13 07 20 00  	addi	a4, zero, 2
   11858: 63 1a c5 08  	bne	a0, a2, 0x118ec <main+0x170c>
   1185c: 03 e5 c5 00  	lwu	a0, 12(a1)
   11860: 13 07 30 00  	addi	a4, zero, 3
   11864: 63 14 c5 08  	bne	a0, a2, 0x118ec <main+0x170c>
   11868: 03 e5 05 01  	lwu	a0, 16(a1)
   1186c: 13 06 f0 ff  	addi	a2, zero, -1
   11870: 13 56 06 02  	srli	a2, a2, 32
   11874: 13 07 40 00  	addi	a4, zero, 4
   11878: 63 1a c5 06  	bne	a0, a2, 0x118ec <main+0x170c>
   1187c: 03 e5 45 01  	lwu	a0, 20(a1)
   11880: 13 07 50 00  	addi	a4, zero, 5
   11884: 63 14 c5 06  	bne	a0, a2, 0x118ec <main+0x170c>
   11888: 03 e5 85 01  	lwu	a0, 24(a1)
   1188c: 13 06 f0 ff  	addi	a2, zero, -1
   11890: 13 56 06 02  	srli	a2, a2, 32
   11894: 13 07 60 00  	addi	a4, zero, 6
   11898: 63 1a c5 04  	bne	a0, a2, 0x118ec <main+0x170c>
   1189c: 03 e5 c5 01  	lwu	a0, 28(a1)
   118a0: 13 07 70 00  	addi	a4, zero, 7
   118a4: 63 14 c5 04  	bne	a0, a2, 0x118ec <main+0x170c>
   118a8: 03 e5 05 02  	lwu	a0, 32(a1)
   118ac: 13 06 f0 ff  	addi	a2, zero, -1
   118b0: 13 56 06 02  	srli	a2, a2, 32
   118b4: 13 07 80 00  	addi	a4, zero, 8
   118b8: 63 1a c5 02  	bne	a0, a2, 0x118ec <main+0x170c>
   118bc: 03 e5 45 02  	lwu	a0, 36(a1)
   118c0: 13 07 90 00  	addi	a4, zero, 9
   118c4: 63 14 c5 02  	bne	a0, a2, 0x118ec <main+0x170c>
   118c8: 03 e5 85 02  	lwu	a0, 40(a1)
   118cc: 13 06 f0 ff  	addi	a2, zero, -1
   118d0: 13 56 06 02  	srli	a2, a2, 32
   118d4: 13 07 a0 00  	addi	a4, zero, 10
   118d8: 63 1a c5 00  	bne	a0, a2, 0x118ec <main+0x170c>
   118dc: 03 e5 c5 02  	lwu	a0, 44(a1)
   118e0: 13 07 b0 00  	addi	a4, zero, 11
   118e4: 63 14 c5 00  	bne	a0, a2, 0x118ec <main+0x170c>
   118e8: 6f 10 50 23  	j	0x1331c <main+0x313c>
   118ec: 1b 08 05 00  	sext.w	a6, a0
   118f0: 17 05 01 00  	auipc	a0, 16
   118f4: 13 05 65 bc  	addi	a0, a0, -1082
   118f8: 97 05 01 00  	auipc	a1, 16
   118fc: 93 85 95 b8  	addi	a1, a1, -1143
   11900: 97 06 01 00  	auipc	a3, 16
   11904: 93 86 26 e0  	addi	a3, a3, -510
   11908: 13 06 90 17  	addi	a2, zero, 377
   1190c: 93 07 f0 ff  	addi	a5, zero, -1
   11910: ef 10 30 3b  	jal	0x134c2 <printf>
   11914: 77 75 16 00  	msettypei	a0, e64, m1, ba
   11918: 77 75 01 20  	msettilemi	a0, 2
   1191c: 77 75 01 40  	msettileki	a0, 2
   11920: 77 75 01 60  	msettileni	a0, 2
   11924: 17 05 01 00  	auipc	a0, 16
   11928: 13 05 c5 14  	addi	a0, a0, 332
   1192c: 93 05 00 01  	addi	a1, zero, 16
   11930: 77 30 b5 04  	mlae64.m	tr0, (a0), a1, m1
   11934: 17 05 01 00  	auipc	a0, 16
   11938: 13 05 c5 15  	addi	a0, a0, 348
   1193c: f7 30 b5 08  	mlbe64.m	tr1, (a0), a1, m1
   11940: 17 05 01 00  	auipc	a0, 16
   11944: 13 05 05 17  	addi	a0, a0, 368
   11948: 77 31 b5 00  	mlce64.m	tr2, (a0), a1, m1
   1194c: 77 61 14 00  	msmau.mm	tr2, tr0, tr1
   11950: 77 b1 ba 02  	msce64.m	tr2, (s5), a1, m1
   11954: 03 a5 09 00  	lw	a0, 0(s3)
   11958: 03 b8 0a 00  	ld	a6, 0(s5)
   1195c: 9b 05 15 00  	addiw	a1, a0, 1
   11960: 13 05 f0 ff  	addi	a0, zero, -1
   11964: 23 a0 b9 00  	sw	a1, 0(s3)
   11968: 63 06 a8 00  	beq	a6, a0, 0x11974 <main+0x1794>
   1196c: 13 07 00 00  	mv	a4, zero
   11970: 6f 00 00 03  	j	0x119a0 <main+0x17c0>
   11974: 03 b8 8a 00  	ld	a6, 8(s5)
   11978: 13 07 10 00  	addi	a4, zero, 1
   1197c: 63 12 a8 02  	bne	a6, a0, 0x119a0 <main+0x17c0>
   11980: 03 b8 0a 01  	ld	a6, 16(s5)
   11984: 13 05 f0 ff  	addi	a0, zero, -1
   11988: 13 07 20 00  	addi	a4, zero, 2
   1198c: 63 1a a8 00  	bne	a6, a0, 0x119a0 <main+0x17c0>
   11990: 03 b8 8a 01  	ld	a6, 24(s5)
   11994: 13 07 30 00  	addi	a4, zero, 3
   11998: 63 14 a8 00  	bne	a6, a0, 0x119a0 <main+0x17c0>
   1199c: 6f 10 80 7e  	j	0x13184 <main+0x2fa4>
   119a0: 17 05 01 00  	auipc	a0, 16
   119a4: 13 05 15 b8  	addi	a0, a0, -1151
   119a8: 97 05 01 00  	auipc	a1, 16
   119ac: 93 85 95 ad  	addi	a1, a1, -1319
   119b0: 97 06 01 00  	auipc	a3, 16
   119b4: 93 86 66 d6  	addi	a3, a3, -666
   119b8: 13 06 c0 18  	addi	a2, zero, 396
   119bc: 93 07 f0 ff  	addi	a5, zero, -1
   119c0: ef 10 30 30  	jal	0x134c2 <printf>
   119c4: 77 75 10 00  	msettypei	a0, e8, m1, ba
   119c8: 77 75 01 20  	msettilemi	a0, 2
   119cc: 77 75 04 40  	msettileki	a0, 8
   119d0: 77 75 08 60  	msettileni	a0, 16
   119d4: 17 05 01 00  	auipc	a0, 16
   119d8: 13 05 85 e7  	addi	a0, a0, -392
   119dc: 93 05 80 00  	addi	a1, zero, 8
   119e0: 77 00 b5 04  	mlae8.m	tr0, (a0), a1, m1
   119e4: 17 f5 00 00  	auipc	a0, 15
   119e8: 13 05 85 bd  	addi	a0, a0, -1064
   119ec: 93 05 00 01  	addi	a1, zero, 16
   119f0: f7 00 b5 08  	mlbe8.m	tr1, (a0), a1, m1
   119f4: 77 f5 12 00  	msettypei	a0, e16, m2, ba
   119f8: 77 75 08 60  	msettileni	a0, 16
   119fc: 97 f5 00 00  	auipc	a1, 15
   11a00: 93 85 05 c4  	addi	a1, a1, -960
   11a04: 13 05 00 02  	addi	a0, zero, 32
   11a08: 77 95 a5 00  	mlce16.m	tr2, (a1), a0, m2
   11a0c: f7 75 10 00  	msettypei	a1, e8, m1, ba
   11a10: f7 75 08 60  	msettileni	a1, 16
   11a14: 77 61 10 04  	mwmau.mm	tr2, tr0, tr1
   11a18: f7 f5 12 00  	msettypei	a1, e16, m2, ba
   11a1c: f7 75 08 60  	msettileni	a1, 16
   11a20: 93 85 01 64  	addi	a1, gp, 1600
   11a24: 77 95 a5 02  	msce16.m	tr2, (a1), a0, m2
   11a28: 83 a6 09 00  	lw	a3, 0(s3)
   11a2c: 13 06 00 00  	mv	a2, zero
   11a30: 9b 86 16 00  	addiw	a3, a3, 1
   11a34: 23 a0 d9 00  	sw	a3, 0(s3)
   11a38: 97 f6 00 00  	auipc	a3, 15
   11a3c: 93 86 46 c4  	addi	a3, a3, -956
   11a40: 83 d7 06 00  	lhu	a5, 0(a3)
   11a44: 03 d8 05 00  	lhu	a6, 0(a1)
   11a48: 63 94 07 05  	bne	a5, a6, 0x11a90 <main+0x18b0>
   11a4c: 13 06 16 00  	addi	a2, a2, 1
   11a50: 93 85 25 00  	addi	a1, a1, 2
   11a54: 93 86 26 00  	addi	a3, a3, 2
   11a58: e3 14 a6 fe  	bne	a2, a0, 0x11a40 <main+0x1860>
   11a5c: 13 85 41 95  	addi	a0, gp, -1708
   11a60: 83 25 05 00  	lw	a1, 0(a0)
   11a64: 9b 85 15 00  	addiw	a1, a1, 1
   11a68: 23 20 b5 00  	sw	a1, 0(a0)
   11a6c: 17 05 01 00  	auipc	a0, 16
   11a70: 13 05 95 a3  	addi	a0, a0, -1479
   11a74: 97 05 01 00  	auipc	a1, 16
   11a78: 93 85 d5 a0  	addi	a1, a1, -1523
   11a7c: 97 06 01 00  	auipc	a3, 16
   11a80: 93 86 e6 ca  	addi	a3, a3, -850
   11a84: 13 06 10 1b  	addi	a2, zero, 433
   11a88: ef 10 b0 23  	jal	0x134c2 <printf>
   11a8c: 6f 00 80 02  	j	0x11ab4 <main+0x18d4>
   11a90: 1b 07 06 00  	sext.w	a4, a2
   11a94: 17 05 01 00  	auipc	a0, 16
   11a98: 13 05 25 a2  	addi	a0, a0, -1502
   11a9c: 97 05 01 00  	auipc	a1, 16
   11aa0: 93 85 55 9e  	addi	a1, a1, -1563
   11aa4: 97 06 01 00  	auipc	a3, 16
   11aa8: 93 86 66 c8  	addi	a3, a3, -890
   11aac: 13 06 10 1b  	addi	a2, zero, 433
   11ab0: ef 10 30 21  	jal	0x134c2 <printf>
   11ab4: 77 75 12 00  	msettypei	a0, e16, m1, ba
   11ab8: 77 75 01 20  	msettilemi	a0, 2
   11abc: 77 75 04 40  	msettileki	a0, 8
   11ac0: 77 75 04 60  	msettileni	a0, 8
   11ac4: 17 05 01 00  	auipc	a0, 16
   11ac8: 13 05 c5 00  	addi	a0, a0, 12
   11acc: 93 05 00 01  	addi	a1, zero, 16
   11ad0: 77 10 b5 04  	mlae16.m	tr0, (a0), a1, m1
   11ad4: 17 f5 00 00  	auipc	a0, 15
   11ad8: 13 05 85 be  	addi	a0, a0, -1048
   11adc: f7 10 b5 08  	mlbe16.m	tr1, (a0), a1, m1
   11ae0: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   11ae4: 77 75 04 60  	msettileni	a0, 8
   11ae8: 17 f5 00 00  	auipc	a0, 15
   11aec: 13 05 45 c5  	addi	a0, a0, -940
   11af0: 93 05 00 02  	addi	a1, zero, 32
   11af4: 77 25 b5 00  	mlce32.m	tr2, (a0), a1, m2
   11af8: 77 75 12 00  	msettypei	a0, e16, m1, ba
   11afc: 77 75 04 60  	msettileni	a0, 8
   11b00: 77 61 10 04  	mwmau.mm	tr2, tr0, tr1
   11b04: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   11b08: 77 75 04 60  	msettileni	a0, 8
   11b0c: 77 25 bd 02  	msce32.m	tr2, (s10), a1, m2
   11b10: 83 a5 09 00  	lw	a1, 0(s3)
   11b14: 03 65 0d 00  	lwu	a0, 0(s10)
   11b18: 9b 85 15 00  	addiw	a1, a1, 1
   11b1c: 37 46 0c 00  	lui	a2, 196
   11b20: 9b 07 a6 08  	addiw	a5, a2, 138
   11b24: 23 a0 b9 00  	sw	a1, 0(s3)
   11b28: 63 16 f5 16  	bne	a0, a5, 0x11c94 <main+0x1ab4>
   11b2c: 97 35 01 00  	auipc	a1, 19
   11b30: 93 85 45 5c  	addi	a1, a1, 1476
   11b34: 03 e5 45 00  	lwu	a0, 4(a1)
   11b38: 37 d6 0f 00  	lui	a2, 253
   11b3c: 9b 07 06 8c  	addiw	a5, a2, -1856
   11b40: 13 07 10 00  	addi	a4, zero, 1
   11b44: 63 1a f5 14  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11b48: 03 e5 85 00  	lwu	a0, 8(a1)
   11b4c: 37 96 12 00  	lui	a2, 297
   11b50: 9b 07 26 be  	addiw	a5, a2, -1054
   11b54: 13 07 20 00  	addi	a4, zero, 2
   11b58: 63 10 f5 14  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11b5c: 03 e5 c5 00  	lwu	a0, 12(a1)
   11b60: 37 a6 0c 00  	lui	a2, 202
   11b64: 9b 07 16 30  	addiw	a5, a2, 769
   11b68: 13 07 30 00  	addi	a4, zero, 3
   11b6c: 63 16 f5 12  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11b70: 03 e5 05 01  	lwu	a0, 16(a1)
   11b74: 37 56 14 00  	lui	a2, 325
   11b78: 9b 07 f6 f9  	addiw	a5, a2, -97
   11b7c: 13 07 40 00  	addi	a4, zero, 4
   11b80: 63 1c f5 10  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11b84: 03 e5 45 01  	lwu	a0, 20(a1)
   11b88: 37 76 12 00  	lui	a2, 295
   11b8c: 9b 07 66 a0  	addiw	a5, a2, -1530
   11b90: 13 07 50 00  	addi	a4, zero, 5
   11b94: 63 12 f5 10  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11b98: 03 e5 85 01  	lwu	a0, 24(a1)
   11b9c: 37 46 0d 00  	lui	a2, 212
   11ba0: 9b 07 26 54  	addiw	a5, a2, 1346
   11ba4: 13 07 60 00  	addi	a4, zero, 6
   11ba8: 63 18 f5 0e  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11bac: 03 e5 c5 01  	lwu	a0, 28(a1)
   11bb0: 37 96 06 00  	lui	a2, 105
   11bb4: 9b 07 d6 7c  	addiw	a5, a2, 1997
   11bb8: 13 07 70 00  	addi	a4, zero, 7
   11bbc: 63 1e f5 0c  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11bc0: 03 e5 05 02  	lwu	a0, 32(a1)
   11bc4: 37 c6 0f 00  	lui	a2, 252
   11bc8: 9b 07 26 2a  	addiw	a5, a2, 674
   11bcc: 13 07 80 00  	addi	a4, zero, 8
   11bd0: 63 14 f5 0c  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11bd4: 03 e5 45 02  	lwu	a0, 36(a1)
   11bd8: 37 d6 0f 00  	lui	a2, 253
   11bdc: 9b 07 c6 35  	addiw	a5, a2, 860
   11be0: 13 07 90 00  	addi	a4, zero, 9
   11be4: 63 1a f5 0a  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11be8: 03 e5 85 02  	lwu	a0, 40(a1)
   11bec: 37 c6 12 00  	lui	a2, 300
   11bf0: 9b 07 b6 cd  	addiw	a5, a2, -805
   11bf4: 13 07 a0 00  	addi	a4, zero, 10
   11bf8: 63 10 f5 0a  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11bfc: 03 e5 c5 02  	lwu	a0, 44(a1)
   11c00: 37 36 11 00  	lui	a2, 275
   11c04: 9b 07 26 83  	addiw	a5, a2, -1998
   11c08: 13 07 b0 00  	addi	a4, zero, 11
   11c0c: 63 16 f5 08  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11c10: 03 e5 05 03  	lwu	a0, 48(a1)
   11c14: 37 b6 16 00  	lui	a2, 363
   11c18: 9b 07 a6 9e  	addiw	a5, a2, -1558
   11c1c: 13 07 c0 00  	addi	a4, zero, 12
   11c20: 63 1c f5 06  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11c24: 03 e5 45 03  	lwu	a0, 52(a1)
   11c28: 37 36 12 00  	lui	a2, 291
   11c2c: 9b 07 b6 63  	addiw	a5, a2, 1595
   11c30: 13 07 d0 00  	addi	a4, zero, 13
   11c34: 63 12 f5 06  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11c38: 03 e5 85 03  	lwu	a0, 56(a1)
   11c3c: 37 76 0d 00  	lui	a2, 215
   11c40: 9b 07 06 6e  	addiw	a5, a2, 1760
   11c44: 13 07 e0 00  	addi	a4, zero, 14
   11c48: 63 18 f5 04  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11c4c: 03 e5 c5 03  	lwu	a0, 60(a1)
   11c50: b7 65 08 00  	lui	a1, 134
   11c54: 9b 87 e5 58  	addiw	a5, a1, 1422
   11c58: 13 07 f0 00  	addi	a4, zero, 15
   11c5c: 63 1e f5 02  	bne	a0, a5, 0x11c98 <main+0x1ab8>
   11c60: 13 85 41 95  	addi	a0, gp, -1708
   11c64: 83 25 05 00  	lw	a1, 0(a0)
   11c68: 9b 85 15 00  	addiw	a1, a1, 1
   11c6c: 23 20 b5 00  	sw	a1, 0(a0)
   11c70: 17 05 01 00  	auipc	a0, 16
   11c74: 13 05 55 83  	addi	a0, a0, -1995
   11c78: 97 05 01 00  	auipc	a1, 16
   11c7c: 93 85 95 80  	addi	a1, a1, -2039
   11c80: 97 06 01 00  	auipc	a3, 16
   11c84: 93 86 e6 ab  	addi	a3, a3, -1346
   11c88: 13 06 f0 1c  	addi	a2, zero, 463
   11c8c: ef 10 70 03  	jal	0x134c2 <printf>
   11c90: 6f 00 c0 02  	j	0x11cbc <main+0x1adc>
   11c94: 13 07 00 00  	mv	a4, zero
   11c98: 1b 08 05 00  	sext.w	a6, a0
   11c9c: 17 05 01 00  	auipc	a0, 16
   11ca0: 13 05 a5 81  	addi	a0, a0, -2022
   11ca4: 97 f5 00 00  	auipc	a1, 15
   11ca8: 93 85 d5 7d  	addi	a1, a1, 2013
   11cac: 97 06 01 00  	auipc	a3, 16
   11cb0: 93 86 26 a9  	addi	a3, a3, -1390
   11cb4: 13 06 f0 1c  	addi	a2, zero, 463
   11cb8: ef 10 b0 00  	jal	0x134c2 <printf>
   11cbc: 77 75 10 00  	msettypei	a0, e8, m1, ba
   11cc0: 77 75 01 20  	msettilemi	a0, 2
   11cc4: 77 75 04 40  	msettileki	a0, 8
   11cc8: 77 75 08 60  	msettileni	a0, 16
   11ccc: 17 05 01 00  	auipc	a0, 16
   11cd0: 13 05 05 b9  	addi	a0, a0, -1136
   11cd4: 93 05 80 00  	addi	a1, zero, 8
   11cd8: 77 02 b5 04  	mlae8.m	tr4, (a0), a1, m1
   11cdc: 17 f5 00 00  	auipc	a0, 15
   11ce0: 13 05 05 aa  	addi	a0, a0, -1376
   11ce4: 93 05 00 01  	addi	a1, zero, 16
   11ce8: f7 02 b5 08  	mlbe8.m	tr5, (a0), a1, m1
   11cec: 77 75 15 00  	msettypei	a0, e32, m4, ba
   11cf0: 77 75 08 60  	msettileni	a0, 16
   11cf4: 17 f5 00 00  	auipc	a0, 15
   11cf8: 13 05 85 b0  	addi	a0, a0, -1272
   11cfc: 93 05 00 04  	addi	a1, zero, 64
   11d00: 77 28 b5 00  	mlce32.m	tr0, (a0), a1, m4
   11d04: 77 75 10 00  	msettypei	a0, e8, m1, ba
   11d08: 77 75 08 60  	msettileni	a0, 16
   11d0c: 77 60 52 08  	mqmau.mm	tr0, tr4, tr5
   11d10: 77 75 15 00  	msettypei	a0, e32, m4, ba
   11d14: 77 75 08 60  	msettileni	a0, 16
   11d18: 77 28 bd 02  	msce32.m	tr0, (s10), a1, m4
   11d1c: 83 a5 09 00  	lw	a1, 0(s3)
   11d20: 13 05 00 00  	mv	a0, zero
   11d24: 9b 85 15 00  	addiw	a1, a1, 1
   11d28: 23 a0 b9 00  	sw	a1, 0(s3)
   11d2c: 97 f5 00 00  	auipc	a1, 15
   11d30: 93 85 05 b5  	addi	a1, a1, -1200
   11d34: 13 06 00 02  	addi	a2, zero, 32
   11d38: 03 a7 05 00  	lw	a4, 0(a1)
   11d3c: 83 26 0d 00  	lw	a3, 0(s10)
   11d40: 63 14 d7 04  	bne	a4, a3, 0x11d88 <main+0x1ba8>
   11d44: 13 05 15 00  	addi	a0, a0, 1
   11d48: 13 0d 4d 00  	addi	s10, s10, 4
   11d4c: 93 85 45 00  	addi	a1, a1, 4
   11d50: e3 14 c5 fe  	bne	a0, a2, 0x11d38 <main+0x1b58>
   11d54: 13 85 41 95  	addi	a0, gp, -1708
   11d58: 83 25 05 00  	lw	a1, 0(a0)
   11d5c: 9b 85 15 00  	addiw	a1, a1, 1
   11d60: 23 20 b5 00  	sw	a1, 0(a0)
   11d64: 17 f5 00 00  	auipc	a0, 15
   11d68: 13 05 15 74  	addi	a0, a0, 1857
   11d6c: 97 f5 00 00  	auipc	a1, 15
   11d70: 93 85 55 71  	addi	a1, a1, 1813
   11d74: 97 06 01 00  	auipc	a3, 16
   11d78: 93 86 e6 9d  	addi	a3, a3, -1570
   11d7c: 13 06 70 1f  	addi	a2, zero, 503
   11d80: ef 10 20 74  	jal	0x134c2 <printf>
   11d84: 6f 00 00 04  	j	0x11dc4 <main+0x1be4>
   11d88: 93 15 07 02  	slli	a1, a4, 32
   11d8c: 93 d5 05 02  	srli	a1, a1, 32
   11d90: 13 96 06 02  	slli	a2, a3, 32
   11d94: 13 56 06 02  	srli	a2, a2, 32
   11d98: 1b 07 05 00  	sext.w	a4, a0
   11d9c: 9b 87 05 00  	sext.w	a5, a1
   11da0: 1b 08 06 00  	sext.w	a6, a2
   11da4: 17 f5 00 00  	auipc	a0, 15
   11da8: 13 05 25 71  	addi	a0, a0, 1810
   11dac: 97 f5 00 00  	auipc	a1, 15
   11db0: 93 85 55 6d  	addi	a1, a1, 1749
   11db4: 97 06 01 00  	auipc	a3, 16
   11db8: 93 86 e6 99  	addi	a3, a3, -1634
   11dbc: 13 06 70 1f  	addi	a2, zero, 503
   11dc0: ef 10 20 70  	jal	0x134c2 <printf>
   11dc4: 77 75 12 00  	msettypei	a0, e16, m1, ba
   11dc8: 77 75 01 20  	msettilemi	a0, 2
   11dcc: 77 75 04 40  	msettileki	a0, 8
   11dd0: 77 75 04 60  	msettileni	a0, 8
   11dd4: 17 05 01 00  	auipc	a0, 16
   11dd8: 13 05 c5 d1  	addi	a0, a0, -740
   11ddc: 93 05 00 01  	addi	a1, zero, 16
   11de0: 77 12 b5 04  	mlae16.m	tr4, (a0), a1, m1
   11de4: 17 f5 00 00  	auipc	a0, 15
   11de8: 13 05 85 b1  	addi	a0, a0, -1256
   11dec: f7 12 b5 08  	mlbe16.m	tr5, (a0), a1, m1
   11df0: 77 75 17 00  	msettypei	a0, e64, m4, ba
   11df4: 77 75 04 60  	msettileni	a0, 8
   11df8: 17 f5 00 00  	auipc	a0, 15
   11dfc: 13 05 85 b8  	addi	a0, a0, -1144
   11e00: 93 05 00 04  	addi	a1, zero, 64
   11e04: 77 38 b5 00  	mlce64.m	tr0, (a0), a1, m4
   11e08: 77 75 12 00  	msettypei	a0, e16, m1, ba
   11e0c: 77 75 04 60  	msettileni	a0, 8
   11e10: 77 60 52 08  	mqmau.mm	tr0, tr4, tr5
   11e14: 77 75 17 00  	msettypei	a0, e64, m4, ba
   11e18: 77 75 04 60  	msettileni	a0, 8
   11e1c: 77 b8 ba 02  	msce64.m	tr0, (s5), a1, m4
   11e20: 03 a5 09 00  	lw	a0, 0(s3)
   11e24: 03 b8 0a 00  	ld	a6, 0(s5)
   11e28: 1b 05 15 00  	addiw	a0, a0, 1
   11e2c: b7 55 98 05  	lui	a1, 22917
   11e30: 9b 87 e5 1a  	addiw	a5, a1, 430
   11e34: 23 a0 a9 00  	sw	a0, 0(s3)
   11e38: 63 12 f8 16  	bne	a6, a5, 0x11f9c <main+0x1dbc>
   11e3c: 03 b8 8a 00  	ld	a6, 8(s5)
   11e40: 37 95 d3 03  	lui	a0, 15673
   11e44: 9b 07 65 a3  	addiw	a5, a0, -1482
   11e48: 13 07 10 00  	addi	a4, zero, 1
   11e4c: 63 1a f8 14  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11e50: 03 b8 0a 01  	ld	a6, 16(s5)
   11e54: 37 55 05 05  	lui	a0, 20565
   11e58: 9b 07 85 91  	addiw	a5, a0, -1768
   11e5c: 13 07 20 00  	addi	a4, zero, 2
   11e60: 63 10 f8 14  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11e64: 03 b8 8a 01  	ld	a6, 24(s5)
   11e68: 37 f5 d3 06  	lui	a0, 27967
   11e6c: 9b 07 e5 35  	addiw	a5, a0, 862
   11e70: 13 07 30 00  	addi	a4, zero, 3
   11e74: 63 16 f8 12  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11e78: 03 b8 0a 02  	ld	a6, 32(s5)
   11e7c: 37 45 b5 07  	lui	a0, 31572
   11e80: 9b 07 45 4f  	addiw	a5, a0, 1268
   11e84: 13 07 40 00  	addi	a4, zero, 4
   11e88: 63 1c f8 10  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11e8c: 03 b8 8a 02  	ld	a6, 40(s5)
   11e90: 37 75 24 04  	lui	a0, 16967
   11e94: 9b 07 c5 b8  	addiw	a5, a0, -1140
   11e98: 13 07 50 00  	addi	a4, zero, 5
   11e9c: 63 12 f8 10  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11ea0: 03 b8 0a 03  	ld	a6, 48(s5)
   11ea4: 37 c5 42 07  	lui	a0, 29740
   11ea8: 9b 07 f5 f4  	addiw	a5, a0, -177
   11eac: 13 07 60 00  	addi	a4, zero, 6
   11eb0: 63 18 f8 0e  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11eb4: 03 b8 8a 03  	ld	a6, 56(s5)
   11eb8: 37 45 d5 05  	lui	a0, 23892
   11ebc: 9b 07 25 15  	addiw	a5, a0, 338
   11ec0: 13 07 70 00  	addi	a4, zero, 7
   11ec4: 63 1e f8 0c  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11ec8: 03 b8 0a 04  	ld	a6, 64(s5)
   11ecc: 37 85 6c 04  	lui	a0, 18120
   11ed0: 9b 07 b5 f1  	addiw	a5, a0, -229
   11ed4: 13 07 80 00  	addi	a4, zero, 8
   11ed8: 63 14 f8 0c  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11edc: 03 b8 8a 04  	ld	a6, 72(s5)
   11ee0: 37 15 54 03  	lui	a0, 13633
   11ee4: 9b 07 65 74  	addiw	a5, a0, 1862
   11ee8: 13 07 90 00  	addi	a4, zero, 9
   11eec: 63 1a f8 0a  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11ef0: 03 b8 0a 05  	ld	a6, 80(s5)
   11ef4: 37 85 24 03  	lui	a0, 12872
   11ef8: 9b 07 e5 b2  	addiw	a5, a0, -1234
   11efc: 13 07 a0 00  	addi	a4, zero, 10
   11f00: 63 10 f8 0a  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f04: 03 b8 8a 05  	ld	a6, 88(s5)
   11f08: 37 25 6b 03  	lui	a0, 14002
   11f0c: 9b 07 95 d8  	addiw	a5, a0, -631
   11f10: 13 07 b0 00  	addi	a4, zero, 11
   11f14: 63 16 f8 08  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f18: 03 b8 0a 06  	ld	a6, 96(s5)
   11f1c: 37 65 71 03  	lui	a0, 14102
   11f20: 9b 07 95 4e  	addiw	a5, a0, 1257
   11f24: 13 07 c0 00  	addi	a4, zero, 12
   11f28: 63 1c f8 06  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f2c: 03 b8 8a 06  	ld	a6, 104(s5)
   11f30: 37 05 06 03  	lui	a0, 12384
   11f34: 9b 07 65 c4  	addiw	a5, a0, -954
   11f38: 13 07 d0 00  	addi	a4, zero, 13
   11f3c: 63 12 f8 06  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f40: 03 b8 0a 07  	ld	a6, 112(s5)
   11f44: 37 d5 a2 04  	lui	a0, 18989
   11f48: 9b 07 75 0b  	addiw	a5, a0, 183
   11f4c: 13 07 e0 00  	addi	a4, zero, 14
   11f50: 63 18 f8 04  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f54: 03 b8 8a 07  	ld	a6, 120(s5)
   11f58: 37 05 2b 03  	lui	a0, 12976
   11f5c: 9b 07 b5 33  	addiw	a5, a0, 827
   11f60: 13 07 f0 00  	addi	a4, zero, 15
   11f64: 63 1e f8 02  	bne	a6, a5, 0x11fa0 <main+0x1dc0>
   11f68: 13 85 41 95  	addi	a0, gp, -1708
   11f6c: 83 25 05 00  	lw	a1, 0(a0)
   11f70: 9b 85 15 00  	addiw	a1, a1, 1
   11f74: 23 20 b5 00  	sw	a1, 0(a0)
   11f78: 17 f5 00 00  	auipc	a0, 15
   11f7c: 13 05 d5 52  	addi	a0, a0, 1325
   11f80: 97 f5 00 00  	auipc	a1, 15
   11f84: 93 85 15 50  	addi	a1, a1, 1281
   11f88: 97 f6 00 00  	auipc	a3, 15
   11f8c: 93 86 e6 7d  	addi	a3, a3, 2014
   11f90: 13 06 90 21  	addi	a2, zero, 537
   11f94: ef 10 e0 52  	jal	0x134c2 <printf>
   11f98: 6f 00 80 02  	j	0x11fc0 <main+0x1de0>
   11f9c: 13 07 00 00  	mv	a4, zero
   11fa0: 17 f5 00 00  	auipc	a0, 15
   11fa4: 13 05 15 58  	addi	a0, a0, 1409
   11fa8: 97 f5 00 00  	auipc	a1, 15
   11fac: 93 85 95 4d  	addi	a1, a1, 1241
   11fb0: 97 f6 00 00  	auipc	a3, 15
   11fb4: 93 86 66 7b  	addi	a3, a3, 1974
   11fb8: 13 06 90 21  	addi	a2, zero, 537
   11fbc: ef 10 60 50  	jal	0x134c2 <printf>
   11fc0: 13 05 f0 ff  	addi	a0, zero, -1
   11fc4: 23 30 a1 08  	sd	a0, 128(sp)
   11fc8: 23 3c a1 06  	sd	a0, 120(sp)
   11fcc: 23 38 a1 06  	sd	a0, 112(sp)
   11fd0: 23 34 a1 06  	sd	a0, 104(sp)
   11fd4: 23 30 a1 06  	sd	a0, 96(sp)
   11fd8: 23 3c a1 04  	sd	a0, 88(sp)
   11fdc: 23 38 a1 04  	sd	a0, 80(sp)
   11fe0: 23 34 a1 04  	sd	a0, 72(sp)
   11fe4: 77 75 10 00  	msettypei	a0, e8, m1, ba
   11fe8: 77 75 01 20  	msettilemi	a0, 2
   11fec: 77 75 04 40  	msettileki	a0, 8
   11ff0: 77 75 08 60  	msettileni	a0, 16
   11ff4: 17 05 01 00  	auipc	a0, 16
   11ff8: 13 05 85 87  	addi	a0, a0, -1928
   11ffc: 93 05 80 00  	addi	a1, zero, 8
   12000: 77 00 b5 04  	mlae8.m	tr0, (a0), a1, m1
   12004: 17 f5 00 00  	auipc	a0, 15
   12008: 13 05 c5 9f  	addi	a0, a0, -1540
   1200c: 93 05 00 01  	addi	a1, zero, 16
   12010: f7 00 b5 08  	mlbe8.m	tr1, (a0), a1, m1
   12014: 77 f5 12 00  	msettypei	a0, e16, m2, ba
   12018: 77 75 08 60  	msettileni	a0, 16
   1201c: 97 f5 00 00  	auipc	a1, 15
   12020: 93 85 45 a6  	addi	a1, a1, -1436
   12024: 13 05 00 02  	addi	a0, zero, 32
   12028: 77 95 a5 00  	mlce16.m	tr2, (a1), a0, m2
   1202c: f7 75 10 00  	msettypei	a1, e8, m1, ba
   12030: f7 75 08 60  	msettileni	a1, 16
   12034: 77 61 14 04  	mswmau.mm	tr2, tr0, tr1
   12038: f7 f5 12 00  	msettypei	a1, e16, m2, ba
   1203c: f7 75 08 60  	msettileni	a1, 16
   12040: 93 85 01 64  	addi	a1, gp, 1600
   12044: 77 95 a5 02  	msce16.m	tr2, (a1), a0, m2
   12048: 83 a6 09 00  	lw	a3, 0(s3)
   1204c: 13 06 00 00  	mv	a2, zero
   12050: 9b 86 16 00  	addiw	a3, a3, 1
   12054: 23 a0 d9 00  	sw	a3, 0(s3)
   12058: 93 06 81 04  	addi	a3, sp, 72
   1205c: 83 d7 06 00  	lhu	a5, 0(a3)
   12060: 03 d8 05 00  	lhu	a6, 0(a1)
   12064: 63 94 07 05  	bne	a5, a6, 0x120ac <main+0x1ecc>
   12068: 13 06 16 00  	addi	a2, a2, 1
   1206c: 93 85 25 00  	addi	a1, a1, 2
   12070: 93 86 26 00  	addi	a3, a3, 2
   12074: e3 14 a6 fe  	bne	a2, a0, 0x1205c <main+0x1e7c>
   12078: 13 85 41 95  	addi	a0, gp, -1708
   1207c: 83 25 05 00  	lw	a1, 0(a0)
   12080: 9b 85 15 00  	addiw	a1, a1, 1
   12084: 23 20 b5 00  	sw	a1, 0(a0)
   12088: 17 f5 00 00  	auipc	a0, 15
   1208c: 13 05 d5 41  	addi	a0, a0, 1053
   12090: 97 f5 00 00  	auipc	a1, 15
   12094: 93 85 15 3f  	addi	a1, a1, 1009
   12098: 97 f6 00 00  	auipc	a3, 15
   1209c: 93 86 26 6e  	addi	a3, a3, 1762
   120a0: 13 06 80 24  	addi	a2, zero, 584
   120a4: ef 10 e0 41  	jal	0x134c2 <printf>
   120a8: 6f 00 80 02  	j	0x120d0 <main+0x1ef0>
   120ac: 1b 07 06 00  	sext.w	a4, a2
   120b0: 17 f5 00 00  	auipc	a0, 15
   120b4: 13 05 65 40  	addi	a0, a0, 1030
   120b8: 97 f5 00 00  	auipc	a1, 15
   120bc: 93 85 95 3c  	addi	a1, a1, 969
   120c0: 97 f6 00 00  	auipc	a3, 15
   120c4: 93 86 a6 6b  	addi	a3, a3, 1722
   120c8: 13 06 80 24  	addi	a2, zero, 584
   120cc: ef 10 60 3f  	jal	0x134c2 <printf>
   120d0: 77 75 12 00  	msettypei	a0, e16, m1, ba
   120d4: 77 75 01 20  	msettilemi	a0, 2
   120d8: 77 75 04 40  	msettileki	a0, 8
   120dc: 77 75 04 60  	msettileni	a0, 8
   120e0: 17 05 01 00  	auipc	a0, 16
   120e4: 13 05 05 a3  	addi	a0, a0, -1488
   120e8: 93 05 00 01  	addi	a1, zero, 16
   120ec: 77 10 b5 04  	mlae16.m	tr0, (a0), a1, m1
   120f0: 17 f5 00 00  	auipc	a0, 15
   120f4: 13 05 05 9d  	addi	a0, a0, -1584
   120f8: f7 10 b5 08  	mlbe16.m	tr1, (a0), a1, m1
   120fc: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   12100: 77 75 04 60  	msettileni	a0, 8
   12104: 17 f5 00 00  	auipc	a0, 15
   12108: 13 05 c5 a3  	addi	a0, a0, -1476
   1210c: 93 05 00 02  	addi	a1, zero, 32
   12110: 77 25 b5 00  	mlce32.m	tr2, (a0), a1, m2
   12114: 77 75 12 00  	msettypei	a0, e16, m1, ba
   12118: 77 75 04 60  	msettileni	a0, 8
   1211c: 77 61 14 04  	mswmau.mm	tr2, tr0, tr1
   12120: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   12124: 77 75 04 60  	msettileni	a0, 8
   12128: 97 34 01 00  	auipc	s1, 19
   1212c: 93 84 84 fc  	addi	s1, s1, -56
   12130: 77 a5 b4 02  	msce32.m	tr2, (s1), a1, m2
   12134: 83 a5 09 00  	lw	a1, 0(s3)
   12138: 03 e5 04 00  	lwu	a0, 0(s1)
   1213c: 9b 85 15 00  	addiw	a1, a1, 1
   12140: 13 06 f0 ff  	addi	a2, zero, -1
   12144: 13 56 06 02  	srli	a2, a2, 32
   12148: 23 a0 b9 00  	sw	a1, 0(s3)
   1214c: 63 06 c5 00  	beq	a0, a2, 0x12158 <main+0x1f78>
   12150: 13 07 00 00  	mv	a4, zero
   12154: 6f 00 c0 0f  	j	0x12250 <main+0x2070>
   12158: 97 35 01 00  	auipc	a1, 19
   1215c: 93 85 85 f9  	addi	a1, a1, -104
   12160: 03 e5 45 00  	lwu	a0, 4(a1)
   12164: 13 07 10 00  	addi	a4, zero, 1
   12168: 63 14 c5 0e  	bne	a0, a2, 0x12250 <main+0x2070>
   1216c: 03 e5 85 00  	lwu	a0, 8(a1)
   12170: 13 06 f0 ff  	addi	a2, zero, -1
   12174: 13 56 06 02  	srli	a2, a2, 32
   12178: 13 07 20 00  	addi	a4, zero, 2
   1217c: 63 1a c5 0c  	bne	a0, a2, 0x12250 <main+0x2070>
   12180: 03 e5 c5 00  	lwu	a0, 12(a1)
   12184: 13 07 30 00  	addi	a4, zero, 3
   12188: 63 14 c5 0c  	bne	a0, a2, 0x12250 <main+0x2070>
   1218c: 03 e5 05 01  	lwu	a0, 16(a1)
   12190: 13 06 f0 ff  	addi	a2, zero, -1
   12194: 13 56 06 02  	srli	a2, a2, 32
   12198: 13 07 40 00  	addi	a4, zero, 4
   1219c: 63 1a c5 0a  	bne	a0, a2, 0x12250 <main+0x2070>
   121a0: 03 e5 45 01  	lwu	a0, 20(a1)
   121a4: 13 07 50 00  	addi	a4, zero, 5
   121a8: 63 14 c5 0a  	bne	a0, a2, 0x12250 <main+0x2070>
   121ac: 03 e5 85 01  	lwu	a0, 24(a1)
   121b0: 13 06 f0 ff  	addi	a2, zero, -1
   121b4: 13 56 06 02  	srli	a2, a2, 32
   121b8: 13 07 60 00  	addi	a4, zero, 6
   121bc: 63 1a c5 08  	bne	a0, a2, 0x12250 <main+0x2070>
   121c0: 03 e5 c5 01  	lwu	a0, 28(a1)
   121c4: 13 07 70 00  	addi	a4, zero, 7
   121c8: 63 14 c5 08  	bne	a0, a2, 0x12250 <main+0x2070>
   121cc: 03 e5 05 02  	lwu	a0, 32(a1)
   121d0: 13 06 f0 ff  	addi	a2, zero, -1
   121d4: 13 56 06 02  	srli	a2, a2, 32
   121d8: 13 07 80 00  	addi	a4, zero, 8
   121dc: 63 1a c5 06  	bne	a0, a2, 0x12250 <main+0x2070>
   121e0: 03 e5 45 02  	lwu	a0, 36(a1)
   121e4: 13 07 90 00  	addi	a4, zero, 9
   121e8: 63 14 c5 06  	bne	a0, a2, 0x12250 <main+0x2070>
   121ec: 03 e5 85 02  	lwu	a0, 40(a1)
   121f0: 13 06 f0 ff  	addi	a2, zero, -1
   121f4: 13 56 06 02  	srli	a2, a2, 32
   121f8: 13 07 a0 00  	addi	a4, zero, 10
   121fc: 63 1a c5 04  	bne	a0, a2, 0x12250 <main+0x2070>
   12200: 03 e5 c5 02  	lwu	a0, 44(a1)
   12204: 13 07 b0 00  	addi	a4, zero, 11
   12208: 63 14 c5 04  	bne	a0, a2, 0x12250 <main+0x2070>
   1220c: 03 e5 05 03  	lwu	a0, 48(a1)
   12210: 13 06 f0 ff  	addi	a2, zero, -1
   12214: 13 56 06 02  	srli	a2, a2, 32
   12218: 13 07 c0 00  	addi	a4, zero, 12
   1221c: 63 1a c5 02  	bne	a0, a2, 0x12250 <main+0x2070>
   12220: 03 e5 45 03  	lwu	a0, 52(a1)
   12224: 13 07 d0 00  	addi	a4, zero, 13
   12228: 63 14 c5 02  	bne	a0, a2, 0x12250 <main+0x2070>
   1222c: 03 e5 85 03  	lwu	a0, 56(a1)
   12230: 13 06 f0 ff  	addi	a2, zero, -1
   12234: 13 56 06 02  	srli	a2, a2, 32
   12238: 13 07 e0 00  	addi	a4, zero, 14
   1223c: 63 1a c5 00  	bne	a0, a2, 0x12250 <main+0x2070>
   12240: 03 e5 c5 03  	lwu	a0, 60(a1)
   12244: 13 07 f0 00  	addi	a4, zero, 15
   12248: 63 14 c5 00  	bne	a0, a2, 0x12250 <main+0x2070>
   1224c: 6f 10 40 10  	j	0x13350 <main+0x3170>
   12250: 1b 08 05 00  	sext.w	a6, a0
   12254: 17 f5 00 00  	auipc	a0, 15
   12258: 13 05 25 26  	addi	a0, a0, 610
   1225c: 97 f5 00 00  	auipc	a1, 15
   12260: 93 85 55 22  	addi	a1, a1, 549
   12264: 97 f6 00 00  	auipc	a3, 15
   12268: 93 86 a6 52  	addi	a3, a3, 1322
   1226c: 13 06 b0 26  	addi	a2, zero, 619
   12270: 93 07 f0 ff  	addi	a5, zero, -1
   12274: ef 10 e0 24  	jal	0x134c2 <printf>
   12278: 13 05 f0 ff  	addi	a0, zero, -1
   1227c: 23 30 a1 04  	sd	a0, 64(sp)
   12280: 23 3c a1 02  	sd	a0, 56(sp)
   12284: 23 38 a1 02  	sd	a0, 48(sp)
   12288: 23 34 a1 02  	sd	a0, 40(sp)
   1228c: 23 30 a1 02  	sd	a0, 32(sp)
   12290: 23 3c a1 00  	sd	a0, 24(sp)
   12294: 23 38 a1 00  	sd	a0, 16(sp)
   12298: 23 34 a1 00  	sd	a0, 8(sp)
   1229c: f7 75 14 00  	msettypei	a1, e32, m1, ba
   122a0: f7 75 01 20  	msettilemi	a1, 2
   122a4: f7 75 02 40  	msettileki	a1, 4
   122a8: f7 75 02 60  	msettileni	a1, 4
   122ac: 17 f4 00 00  	auipc	s0, 15
   122b0: 13 04 44 7a  	addi	s0, s0, 1956
   122b4: 93 05 00 01  	addi	a1, zero, 16
   122b8: 77 20 b4 04  	mlae32.m	tr0, (s0), a1, m1
   122bc: 13 06 81 00  	addi	a2, sp, 8
   122c0: f7 20 b6 08  	mlbe32.m	tr1, (a2), a1, m1
   122c4: f7 f5 16 00  	msettypei	a1, e64, m2, ba
   122c8: f7 75 02 60  	msettileni	a1, 4
   122cc: 97 f5 00 00  	auipc	a1, 15
   122d0: 93 85 45 8b  	addi	a1, a1, -1868
   122d4: 13 06 00 02  	addi	a2, zero, 32
   122d8: 77 b5 c5 00  	mlce64.m	tr2, (a1), a2, m2
   122dc: f7 75 14 00  	msettypei	a1, e32, m1, ba
   122e0: f7 75 02 60  	msettileni	a1, 4
   122e4: 77 61 14 04  	mswmau.mm	tr2, tr0, tr1
   122e8: f7 f5 16 00  	msettypei	a1, e64, m2, ba
   122ec: f7 75 02 60  	msettileni	a1, 4
   122f0: 77 b5 ca 02  	msce64.m	tr2, (s5), a2, m2
   122f4: 83 a5 09 00  	lw	a1, 0(s3)
   122f8: 03 b8 0a 00  	ld	a6, 0(s5)
   122fc: 9b 85 15 00  	addiw	a1, a1, 1
   12300: 23 a0 b9 00  	sw	a1, 0(s3)
   12304: 63 06 a8 00  	beq	a6, a0, 0x12310 <main+0x2130>
   12308: 13 07 00 00  	mv	a4, zero
   1230c: 6f 00 80 06  	j	0x12374 <main+0x2194>
   12310: 03 b8 8a 00  	ld	a6, 8(s5)
   12314: 13 07 10 00  	addi	a4, zero, 1
   12318: 63 1e a8 04  	bne	a6, a0, 0x12374 <main+0x2194>
   1231c: 03 b8 0a 01  	ld	a6, 16(s5)
   12320: 13 05 f0 ff  	addi	a0, zero, -1
   12324: 13 07 20 00  	addi	a4, zero, 2
   12328: 63 16 a8 04  	bne	a6, a0, 0x12374 <main+0x2194>
   1232c: 03 b8 8a 01  	ld	a6, 24(s5)
   12330: 13 07 30 00  	addi	a4, zero, 3
   12334: 63 10 a8 04  	bne	a6, a0, 0x12374 <main+0x2194>
   12338: 03 b8 0a 02  	ld	a6, 32(s5)
   1233c: 13 05 f0 ff  	addi	a0, zero, -1
   12340: 13 07 40 00  	addi	a4, zero, 4
   12344: 63 18 a8 02  	bne	a6, a0, 0x12374 <main+0x2194>
   12348: 03 b8 8a 02  	ld	a6, 40(s5)
   1234c: 13 07 50 00  	addi	a4, zero, 5
   12350: 63 12 a8 02  	bne	a6, a0, 0x12374 <main+0x2194>
   12354: 03 b8 0a 03  	ld	a6, 48(s5)
   12358: 13 05 f0 ff  	addi	a0, zero, -1
   1235c: 13 07 60 00  	addi	a4, zero, 6
   12360: 63 1a a8 00  	bne	a6, a0, 0x12374 <main+0x2194>
   12364: 03 b8 8a 03  	ld	a6, 56(s5)
   12368: 13 07 70 00  	addi	a4, zero, 7
   1236c: 63 14 a8 00  	bne	a6, a0, 0x12374 <main+0x2194>
   12370: 6f 00 90 75  	j	0x132c8 <main+0x30e8>
   12374: 17 f5 00 00  	auipc	a0, 15
   12378: 13 05 d5 1a  	addi	a0, a0, 429
   1237c: 97 f5 00 00  	auipc	a1, 15
   12380: 93 85 55 10  	addi	a1, a1, 261
   12384: 97 f6 00 00  	auipc	a3, 15
   12388: 93 86 e6 41  	addi	a3, a3, 1054
   1238c: 13 06 a0 28  	addi	a2, zero, 650
   12390: 93 07 f0 ff  	addi	a5, zero, -1
   12394: ef 10 e0 12  	jal	0x134c2 <printf>
   12398: 77 75 10 00  	msettypei	a0, e8, m1, ba
   1239c: 77 75 01 20  	msettilemi	a0, 2
   123a0: 77 75 04 40  	msettileki	a0, 8
   123a4: 77 75 08 60  	msettileni	a0, 16
   123a8: 17 f5 00 00  	auipc	a0, 15
   123ac: 13 05 45 4d  	addi	a0, a0, 1236
   123b0: 93 05 80 00  	addi	a1, zero, 8
   123b4: 77 02 b5 04  	mlae8.m	tr4, (a0), a1, m1
   123b8: 17 f5 00 00  	auipc	a0, 15
   123bc: 13 05 85 80  	addi	a0, a0, -2040
   123c0: 93 05 00 01  	addi	a1, zero, 16
   123c4: f7 02 b5 08  	mlbe8.m	tr5, (a0), a1, m1
   123c8: 77 75 15 00  	msettypei	a0, e32, m4, ba
   123cc: 77 75 08 60  	msettileni	a0, 16
   123d0: 17 f5 00 00  	auipc	a0, 15
   123d4: 13 05 05 87  	addi	a0, a0, -1936
   123d8: 93 05 00 04  	addi	a1, zero, 64
   123dc: 77 28 b5 00  	mlce32.m	tr0, (a0), a1, m4
   123e0: 77 75 10 00  	msettypei	a0, e8, m1, ba
   123e4: 77 75 08 60  	msettileni	a0, 16
   123e8: 77 60 56 08  	msqmau.mm	tr0, tr4, tr5
   123ec: 77 75 15 00  	msettypei	a0, e32, m4, ba
   123f0: 77 75 08 60  	msettileni	a0, 16
   123f4: 77 a8 b4 02  	msce32.m	tr0, (s1), a1, m4
   123f8: 83 a5 09 00  	lw	a1, 0(s3)
   123fc: 13 05 00 00  	mv	a0, zero
   12400: 9b 85 15 00  	addiw	a1, a1, 1
   12404: 23 a0 b9 00  	sw	a1, 0(s3)
   12408: 97 f5 00 00  	auipc	a1, 15
   1240c: 93 85 85 8b  	addi	a1, a1, -1864
   12410: 13 06 00 02  	addi	a2, zero, 32
   12414: 03 a7 05 00  	lw	a4, 0(a1)
   12418: 83 a6 04 00  	lw	a3, 0(s1)
   1241c: 63 14 d7 04  	bne	a4, a3, 0x12464 <main+0x2284>
   12420: 13 05 15 00  	addi	a0, a0, 1
   12424: 93 84 44 00  	addi	s1, s1, 4
   12428: 93 85 45 00  	addi	a1, a1, 4
   1242c: e3 14 c5 fe  	bne	a0, a2, 0x12414 <main+0x2234>
   12430: 13 85 41 95  	addi	a0, gp, -1708
   12434: 83 25 05 00  	lw	a1, 0(a0)
   12438: 9b 85 15 00  	addiw	a1, a1, 1
   1243c: 23 20 b5 00  	sw	a1, 0(a0)
   12440: 17 f5 00 00  	auipc	a0, 15
   12444: 13 05 55 06  	addi	a0, a0, 101
   12448: 97 f5 00 00  	auipc	a1, 15
   1244c: 93 85 95 03  	addi	a1, a1, 57
   12450: 97 f6 00 00  	auipc	a3, 15
   12454: 93 86 a6 32  	addi	a3, a3, 810
   12458: 13 06 20 2b  	addi	a2, zero, 690
   1245c: ef 10 60 06  	jal	0x134c2 <printf>
   12460: 6f 00 00 04  	j	0x124a0 <main+0x22c0>
   12464: 93 15 07 02  	slli	a1, a4, 32
   12468: 93 d5 05 02  	srli	a1, a1, 32
   1246c: 13 96 06 02  	slli	a2, a3, 32
   12470: 13 56 06 02  	srli	a2, a2, 32
   12474: 1b 07 05 00  	sext.w	a4, a0
   12478: 9b 87 05 00  	sext.w	a5, a1
   1247c: 1b 08 06 00  	sext.w	a6, a2
   12480: 17 f5 00 00  	auipc	a0, 15
   12484: 13 05 65 03  	addi	a0, a0, 54
   12488: 97 f5 00 00  	auipc	a1, 15
   1248c: 93 85 95 ff  	addi	a1, a1, -7
   12490: 97 f6 00 00  	auipc	a3, 15
   12494: 93 86 a6 2e  	addi	a3, a3, 746
   12498: 13 06 20 2b  	addi	a2, zero, 690
   1249c: ef 10 60 02  	jal	0x134c2 <printf>
   124a0: 77 75 12 00  	msettypei	a0, e16, m1, ba
   124a4: 77 75 01 20  	msettilemi	a0, 2
   124a8: 77 75 04 40  	msettileki	a0, 8
   124ac: 77 75 04 60  	msettileni	a0, 8
   124b0: 17 f5 00 00  	auipc	a0, 15
   124b4: 13 05 05 68  	addi	a0, a0, 1664
   124b8: 93 05 00 01  	addi	a1, zero, 16
   124bc: 77 12 b5 04  	mlae16.m	tr4, (a0), a1, m1
   124c0: 17 f5 00 00  	auipc	a0, 15
   124c4: 13 05 05 88  	addi	a0, a0, -1920
   124c8: f7 12 b5 08  	mlbe16.m	tr5, (a0), a1, m1
   124cc: 77 75 17 00  	msettypei	a0, e64, m4, ba
   124d0: 77 75 04 60  	msettileni	a0, 8
   124d4: 17 f5 00 00  	auipc	a0, 15
   124d8: 13 05 c5 8e  	addi	a0, a0, -1812
   124dc: 93 05 00 04  	addi	a1, zero, 64
   124e0: 77 38 b5 00  	mlce64.m	tr0, (a0), a1, m4
   124e4: 77 75 12 00  	msettypei	a0, e16, m1, ba
   124e8: 77 75 04 60  	msettileni	a0, 8
   124ec: 77 60 56 08  	msqmau.mm	tr0, tr4, tr5
   124f0: 77 75 17 00  	msettypei	a0, e64, m4, ba
   124f4: 77 75 04 60  	msettileni	a0, 8
   124f8: 77 b8 ba 02  	msce64.m	tr0, (s5), a1, m4
   124fc: 03 a5 09 00  	lw	a0, 0(s3)
   12500: 1b 05 15 00  	addiw	a0, a0, 1
   12504: 03 b8 0a 00  	ld	a6, 0(s5)
   12508: b7 95 65 3b  	lui	a1, 243289
   1250c: 93 95 35 00  	slli	a1, a1, 3
   12510: 93 87 05 0b  	addi	a5, a1, 176
   12514: 23 a0 a9 00  	sw	a0, 0(s3)
   12518: 63 14 f8 1c  	bne	a6, a5, 0x126e0 <main+0x2500>
   1251c: 03 b8 8a 00  	ld	a6, 8(s5)
   12520: 37 35 08 00  	lui	a0, 131
   12524: 1b 05 d5 76  	addiw	a0, a0, 1901
   12528: 13 15 e5 00  	slli	a0, a0, 14
   1252c: 93 07 15 db  	addi	a5, a0, -591
   12530: 13 07 10 00  	addi	a4, zero, 1
   12534: 63 18 f8 1a  	bne	a6, a5, 0x126e4 <main+0x2504>
   12538: 03 b8 0a 01  	ld	a6, 16(s5)
   1253c: 37 f5 25 00  	lui	a0, 607
   12540: 1b 05 95 7b  	addiw	a0, a0, 1977
   12544: 13 15 c5 00  	slli	a0, a0, 12
   12548: 93 07 a5 cc  	addi	a5, a0, -822
   1254c: 13 07 20 00  	addi	a4, zero, 2
   12550: 63 1a f8 18  	bne	a6, a5, 0x126e4 <main+0x2504>
   12554: 03 b8 8a 01  	ld	a6, 24(s5)
   12558: 37 45 1f 00  	lui	a0, 500
   1255c: 1b 05 35 73  	addiw	a0, a0, 1843
   12560: 13 15 c5 00  	slli	a0, a0, 12
   12564: 93 07 95 74  	addi	a5, a0, 1865
   12568: 13 07 30 00  	addi	a4, zero, 3
   1256c: 63 1c f8 16  	bne	a6, a5, 0x126e4 <main+0x2504>
   12570: 03 b8 0a 02  	ld	a6, 32(s5)
   12574: 37 55 11 00  	lui	a0, 277
   12578: 1b 05 35 ed  	addiw	a0, a0, -301
   1257c: 13 15 d5 00  	slli	a0, a0, 13
   12580: 93 07 45 5a  	addi	a5, a0, 1444
   12584: 13 07 40 00  	addi	a4, zero, 4
   12588: 63 1e f8 14  	bne	a6, a5, 0x126e4 <main+0x2504>
   1258c: 03 b8 8a 02  	ld	a6, 40(s5)
   12590: 37 a5 15 00  	lui	a0, 346
   12594: 1b 05 55 f8  	addiw	a0, a0, -123
   12598: 13 15 c5 00  	slli	a0, a0, 12
   1259c: 93 07 65 84  	addi	a5, a0, -1978
   125a0: 13 07 50 00  	addi	a4, zero, 5
   125a4: 63 10 f8 14  	bne	a6, a5, 0x126e4 <main+0x2504>
   125a8: 03 b8 0a 03  	ld	a6, 48(s5)
   125ac: 37 15 12 00  	lui	a0, 289
   125b0: 1b 05 95 d0  	addiw	a0, a0, -759
   125b4: 13 15 c5 00  	slli	a0, a0, 12
   125b8: 93 07 65 a5  	addi	a5, a0, -1450
   125bc: 13 07 60 00  	addi	a4, zero, 6
   125c0: 63 12 f8 12  	bne	a6, a5, 0x126e4 <main+0x2504>
   125c4: 03 b8 8a 03  	ld	a6, 56(s5)
   125c8: 37 55 8f 66  	lui	a0, 420085
   125cc: 13 15 25 00  	slli	a0, a0, 2
   125d0: 93 07 85 11  	addi	a5, a0, 280
   125d4: 13 07 70 00  	addi	a4, zero, 7
   125d8: 63 16 f8 10  	bne	a6, a5, 0x126e4 <main+0x2504>
   125dc: 03 b8 0a 04  	ld	a6, 64(s5)
   125e0: 37 95 ac 12  	lui	a0, 76489
   125e4: 13 15 55 00  	slli	a0, a0, 5
   125e8: 93 07 85 8f  	addi	a5, a0, -1800
   125ec: 13 07 80 00  	addi	a4, zero, 8
   125f0: 63 1a f8 0e  	bne	a6, a5, 0x126e4 <main+0x2504>
   125f4: 03 b8 8a 04  	ld	a6, 72(s5)
   125f8: 37 45 2b 00  	lui	a0, 692
   125fc: 1b 05 15 b9  	addiw	a0, a0, -1135
   12600: 13 15 c5 00  	slli	a0, a0, 12
   12604: 93 07 55 5e  	addi	a5, a0, 1509
   12608: 13 07 90 00  	addi	a4, zero, 9
   1260c: 63 1c f8 0c  	bne	a6, a5, 0x126e4 <main+0x2504>
   12610: 03 b8 0a 05  	ld	a6, 80(s5)
   12614: 37 f5 e1 29  	lui	a0, 171551
   12618: 13 15 45 00  	slli	a0, a0, 4
   1261c: 93 07 85 f8  	addi	a5, a0, -120
   12620: 13 07 a0 00  	addi	a4, zero, 10
   12624: 63 10 f8 0c  	bne	a6, a5, 0x126e4 <main+0x2504>
   12628: 03 b8 8a 05  	ld	a6, 88(s5)
   1262c: 37 a5 23 00  	lui	a0, 570
   12630: 1b 05 35 2f  	addiw	a0, a0, 755
   12634: 13 15 c5 00  	slli	a0, a0, 12
   12638: 93 07 75 e7  	addi	a5, a0, -393
   1263c: 13 07 b0 00  	addi	a4, zero, 11
   12640: 63 12 f8 0a  	bne	a6, a5, 0x126e4 <main+0x2504>
   12644: 03 b8 0a 06  	ld	a6, 96(s5)
   12648: 37 75 14 2e  	lui	a0, 188743
   1264c: 13 15 45 00  	slli	a0, a0, 4
   12650: 93 07 a5 59  	addi	a5, a0, 1434
   12654: 13 07 c0 00  	addi	a4, zero, 12
   12658: 63 16 f8 08  	bne	a6, a5, 0x126e4 <main+0x2504>
   1265c: 03 b8 8a 06  	ld	a6, 104(s5)
   12660: 37 95 59 63  	lui	a0, 406937
   12664: 13 15 25 00  	slli	a0, a0, 2
   12668: 93 07 c5 82  	addi	a5, a0, -2004
   1266c: 13 07 d0 00  	addi	a4, zero, 13
   12670: 63 1a f8 06  	bne	a6, a5, 0x126e4 <main+0x2504>
   12674: 03 b8 0a 07  	ld	a6, 112(s5)
   12678: 37 85 19 00  	lui	a0, 408
   1267c: 1b 05 95 b5  	addiw	a0, a0, -1191
   12680: 13 15 c5 00  	slli	a0, a0, 12
   12684: 93 07 05 92  	addi	a5, a0, -1760
   12688: 13 07 e0 00  	addi	a4, zero, 14
   1268c: 63 1c f8 04  	bne	a6, a5, 0x126e4 <main+0x2504>
   12690: 03 b8 8a 07  	ld	a6, 120(s5)
   12694: 37 65 22 00  	lui	a0, 550
   12698: 1b 05 15 c9  	addiw	a0, a0, -879
   1269c: 13 15 c5 00  	slli	a0, a0, 12
   126a0: 93 07 15 66  	addi	a5, a0, 1633
   126a4: 13 07 f0 00  	addi	a4, zero, 15
   126a8: 63 1e f8 02  	bne	a6, a5, 0x126e4 <main+0x2504>
   126ac: 13 85 41 95  	addi	a0, gp, -1708
   126b0: 83 25 05 00  	lw	a1, 0(a0)
   126b4: 9b 85 15 00  	addiw	a1, a1, 1
   126b8: 23 20 b5 00  	sw	a1, 0(a0)
   126bc: 17 f5 00 00  	auipc	a0, 15
   126c0: 13 05 95 de  	addi	a0, a0, -535
   126c4: 97 f5 00 00  	auipc	a1, 15
   126c8: 93 85 d5 db  	addi	a1, a1, -579
   126cc: 97 f6 00 00  	auipc	a3, 15
   126d0: 93 86 26 0c  	addi	a3, a3, 194
   126d4: 13 06 60 2d  	addi	a2, zero, 726
   126d8: ef 00 b0 5e  	jal	0x134c2 <printf>
   126dc: 6f 00 80 02  	j	0x12704 <main+0x2524>
   126e0: 13 07 00 00  	mv	a4, zero
   126e4: 17 f5 00 00  	auipc	a0, 15
   126e8: 13 05 d5 e3  	addi	a0, a0, -451
   126ec: 97 f5 00 00  	auipc	a1, 15
   126f0: 93 85 55 d9  	addi	a1, a1, -619
   126f4: 97 f6 00 00  	auipc	a3, 15
   126f8: 93 86 a6 09  	addi	a3, a3, 154
   126fc: 13 06 60 2d  	addi	a2, zero, 726
   12700: ef 00 30 5c  	jal	0x134c2 <printf>
   12704: 77 75 10 00  	msettypei	a0, e8, m1, ba
   12708: 77 75 01 20  	msettilemi	a0, 2
   1270c: 77 75 04 40  	msettileki	a0, 8
   12710: 77 75 08 60  	msettileni	a0, 16
   12714: 17 f5 00 00  	auipc	a0, 15
   12718: 13 05 85 17  	addi	a0, a0, 376
   1271c: 93 05 80 00  	addi	a1, zero, 8
   12720: 77 00 b5 04  	mlae8.m	tr0, (a0), a1, m1
   12724: 17 e5 00 00  	auipc	a0, 14
   12728: 13 05 c5 71  	addi	a0, a0, 1820
   1272c: 93 05 00 01  	addi	a1, zero, 16
   12730: f7 00 b5 08  	mlbe8.m	tr1, (a0), a1, m1
   12734: 77 f5 12 00  	msettypei	a0, e16, m2, ba
   12738: 77 75 08 60  	msettileni	a0, 16
   1273c: 97 e5 00 00  	auipc	a1, 14
   12740: 93 85 45 78  	addi	a1, a1, 1924
   12744: 13 05 00 02  	addi	a0, zero, 32
   12748: 77 95 a5 00  	mlce16.m	tr2, (a1), a0, m2
   1274c: f7 75 10 00  	msettypei	a1, e8, m1, ba
   12750: f7 75 08 60  	msettileni	a1, 16
   12754: 77 61 1c 04  	mswma.mm	tr2, tr0, tr1
   12758: f7 f5 12 00  	msettypei	a1, e16, m2, ba
   1275c: f7 75 08 60  	msettileni	a1, 16
   12760: 93 85 81 a8  	addi	a1, gp, -1400
   12764: 77 95 a5 02  	msce16.m	tr2, (a1), a0, m2
   12768: 83 a6 09 00  	lw	a3, 0(s3)
   1276c: 13 06 00 00  	mv	a2, zero
   12770: 9b 86 16 00  	addiw	a3, a3, 1
   12774: 23 a0 d9 00  	sw	a3, 0(s3)
   12778: 97 e6 00 00  	auipc	a3, 14
   1277c: 93 86 86 78  	addi	a3, a3, 1928
   12780: 83 d7 06 00  	lhu	a5, 0(a3)
   12784: 83 d4 05 00  	lhu	s1, 0(a1)
   12788: 63 94 97 04  	bne	a5, s1, 0x127d0 <main+0x25f0>
   1278c: 13 06 16 00  	addi	a2, a2, 1
   12790: 93 85 25 00  	addi	a1, a1, 2
   12794: 93 86 26 00  	addi	a3, a3, 2
   12798: e3 14 a6 fe  	bne	a2, a0, 0x12780 <main+0x25a0>
   1279c: 13 85 41 95  	addi	a0, gp, -1708
   127a0: 83 25 05 00  	lw	a1, 0(a0)
   127a4: 9b 85 15 00  	addiw	a1, a1, 1
   127a8: 23 20 b5 00  	sw	a1, 0(a0)
   127ac: 17 f5 00 00  	auipc	a0, 15
   127b0: 13 05 95 cf  	addi	a0, a0, -775
   127b4: 97 f5 00 00  	auipc	a1, 15
   127b8: 93 85 d5 cc  	addi	a1, a1, -819
   127bc: 97 f6 00 00  	auipc	a3, 15
   127c0: 93 86 a6 ff  	addi	a3, a3, -6
   127c4: 13 06 00 30  	addi	a2, zero, 768
   127c8: ef 00 b0 4f  	jal	0x134c2 <printf>
   127cc: 6f 00 80 03  	j	0x12804 <main+0x2624>
   127d0: 1b 07 06 00  	sext.w	a4, a2
   127d4: 13 95 07 03  	slli	a0, a5, 48
   127d8: 93 57 05 43  	srai	a5, a0, 48
   127dc: 13 95 04 03  	slli	a0, s1, 48
   127e0: 13 58 05 43  	srai	a6, a0, 48
   127e4: 17 f5 00 00  	auipc	a0, 15
   127e8: 13 05 c5 c5  	addi	a0, a0, -932
   127ec: 97 f5 00 00  	auipc	a1, 15
   127f0: 93 85 55 c9  	addi	a1, a1, -875
   127f4: 97 f6 00 00  	auipc	a3, 15
   127f8: 93 86 26 fc  	addi	a3, a3, -62
   127fc: 13 06 00 30  	addi	a2, zero, 768
   12800: ef 00 30 4c  	jal	0x134c2 <printf>
   12804: 77 75 12 00  	msettypei	a0, e16, m1, ba
   12808: 77 75 01 20  	msettilemi	a0, 2
   1280c: 77 75 04 40  	msettileki	a0, 8
   12810: 77 75 04 60  	msettileni	a0, 8
   12814: 17 f5 00 00  	auipc	a0, 15
   12818: 13 05 c5 33  	addi	a0, a0, 828
   1281c: 93 05 00 01  	addi	a1, zero, 16
   12820: 77 10 b5 04  	mlae16.m	tr0, (a0), a1, m1
   12824: 17 e5 00 00  	auipc	a0, 14
   12828: 13 05 c5 71  	addi	a0, a0, 1820
   1282c: f7 10 b5 08  	mlbe16.m	tr1, (a0), a1, m1
   12830: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   12834: 77 75 04 60  	msettileni	a0, 8
   12838: 17 e5 00 00  	auipc	a0, 14
   1283c: 13 05 85 78  	addi	a0, a0, 1928
   12840: 93 05 00 02  	addi	a1, zero, 32
   12844: 77 25 b5 00  	mlce32.m	tr2, (a0), a1, m2
   12848: 77 75 12 00  	msettypei	a0, e16, m1, ba
   1284c: 77 75 04 60  	msettileni	a0, 8
   12850: 77 61 1c 04  	mswma.mm	tr2, tr0, tr1
   12854: 77 f5 14 00  	msettypei	a0, e32, m2, ba
   12858: 77 75 04 60  	msettileni	a0, 8
   1285c: 77 25 bc 02  	msce32.m	tr2, (s8), a1, m2
   12860: 83 a5 09 00  	lw	a1, 0(s3)
   12864: 03 65 0c 00  	lwu	a0, 0(s8)
   12868: 9b 85 15 00  	addiw	a1, a1, 1
   1286c: 37 56 b9 47  	lui	a2, 293781
   12870: 1b 06 a6 10  	addiw	a2, a2, 266
   12874: 23 a0 b9 00  	sw	a1, 0(s3)
   12878: 63 1c c5 16  	bne	a0, a2, 0x129f0 <main+0x2810>
   1287c: 93 85 81 c1  	addi	a1, gp, -1000
   12880: 03 e5 45 00  	lwu	a0, 4(a1)
   12884: 37 16 de 08  	lui	a2, 36321
   12888: 1b 06 36 19  	addiw	a2, a2, 403
   1288c: 13 07 10 00  	addi	a4, zero, 1
   12890: 63 12 c5 16  	bne	a0, a2, 0x129f4 <main+0x2814>
   12894: 03 e5 85 00  	lwu	a0, 8(a1)
   12898: 37 06 0e 19  	lui	a2, 102624
   1289c: 1b 06 56 68  	addiw	a2, a2, 1669
   128a0: 13 07 20 00  	addi	a4, zero, 2
   128a4: 63 18 c5 14  	bne	a0, a2, 0x129f4 <main+0x2814>
   128a8: 03 e5 c5 00  	lwu	a0, 12(a1)
   128ac: 37 66 86 03  	lui	a2, 14438
   128b0: 1b 06 d6 2b  	addiw	a2, a2, 701
   128b4: 13 07 30 00  	addi	a4, zero, 3
   128b8: 63 1e c5 12  	bne	a0, a2, 0x129f4 <main+0x2814>
   128bc: 03 e5 05 01  	lwu	a0, 16(a1)
   128c0: 37 96 0e 00  	lui	a2, 233
   128c4: 1b 06 96 1f  	addiw	a2, a2, 505
   128c8: 13 16 c6 00  	slli	a2, a2, 12
   128cc: 13 06 e6 26  	addi	a2, a2, 622
   128d0: e3 12 c5 0a  	bne	a0, a2, 0x13174 <main+0x2f94>
   128d4: 03 e5 45 01  	lwu	a0, 20(a1)
   128d8: 37 66 8d 36  	lui	a2, 223446
   128dc: 1b 06 26 a3  	addiw	a2, a2, -1486
   128e0: 13 07 50 00  	addi	a4, zero, 5
   128e4: 63 18 c5 10  	bne	a0, a2, 0x129f4 <main+0x2814>
   128e8: 03 e5 85 01  	lwu	a0, 24(a1)
   128ec: 37 d6 c7 09  	lui	a2, 40061
   128f0: 1b 06 96 c0  	addiw	a2, a2, -1015
   128f4: 13 07 60 00  	addi	a4, zero, 6
   128f8: 63 1e c5 0e  	bne	a0, a2, 0x129f4 <main+0x2814>
   128fc: 03 e5 c5 01  	lwu	a0, 28(a1)
   12900: 37 96 36 01  	lui	a2, 4969
   12904: 13 16 76 00  	slli	a2, a2, 7
   12908: 13 06 66 f5  	addi	a2, a2, -170
   1290c: e3 14 c5 14  	bne	a0, a2, 0x13254 <main+0x3074>
   12910: 03 e5 05 02  	lwu	a0, 32(a1)
   12914: 37 56 06 34  	lui	a2, 213093
   12918: 13 16 26 00  	slli	a2, a2, 2
   1291c: 13 06 16 d9  	addi	a2, a2, -623
   12920: e3 1c c5 16  	bne	a0, a2, 0x13298 <main+0x30b8>
   12924: 03 e5 45 02  	lwu	a0, 36(a1)
   12928: 37 96 0e 00  	lui	a2, 233
   1292c: 1b 06 76 b5  	addiw	a2, a2, -1193
   12930: 13 16 c6 00  	slli	a2, a2, 12
   12934: 13 06 b6 0c  	addi	a2, a2, 203
   12938: e3 18 c5 16  	bne	a0, a2, 0x132a8 <main+0x30c8>
   1293c: 03 e5 85 02  	lwu	a0, 40(a1)
   12940: 37 a6 0e 00  	lui	a2, 234
   12944: 1b 06 16 fe  	addiw	a2, a2, -31
   12948: 13 16 c6 00  	slli	a2, a2, 12
   1294c: 13 06 e6 df  	addi	a2, a2, -514
   12950: e3 14 c5 16  	bne	a0, a2, 0x132b8 <main+0x30d8>
   12954: 03 e5 c5 02  	lwu	a0, 44(a1)
   12958: 37 b6 ba 34  	lui	a2, 215979
   1295c: 1b 06 36 b2  	addiw	a2, a2, -1245
   12960: 13 07 b0 00  	addi	a4, zero, 11
   12964: 63 18 c5 08  	bne	a0, a2, 0x129f4 <main+0x2814>
   12968: 03 e5 05 03  	lwu	a0, 48(a1)
   1296c: 37 26 09 00  	lui	a2, 146
   12970: 1b 06 76 5b  	addiw	a2, a2, 1463
   12974: 13 16 c6 00  	slli	a2, a2, 12
   12978: 13 06 c6 dc  	addi	a2, a2, -564
   1297c: e3 10 c5 18  	bne	a0, a2, 0x132fc <main+0x311c>
   12980: 03 e5 45 03  	lwu	a0, 52(a1)
   12984: 37 76 a5 52  	lui	a2, 338519
   12988: 1b 06 e6 45  	addiw	a2, a2, 1118
   1298c: 13 07 d0 00  	addi	a4, zero, 13
   12990: 63 12 c5 06  	bne	a0, a2, 0x129f4 <main+0x2814>
   12994: 03 e5 85 03  	lwu	a0, 56(a1)
   12998: 37 56 28 71  	lui	a2, 463493
   1299c: 13 16 16 00  	slli	a2, a2, 1
   129a0: 13 06 46 8e  	addi	a2, a2, -1820
   129a4: e3 14 c5 16  	bne	a0, a2, 0x1330c <main+0x312c>
   129a8: 03 e5 c5 03  	lwu	a0, 60(a1)
   129ac: b7 95 82 41  	lui	a1, 268329
   129b0: 1b 86 b5 dc  	addiw	a2, a1, -565
   129b4: 13 07 f0 00  	addi	a4, zero, 15
   129b8: 63 1e c5 02  	bne	a0, a2, 0x129f4 <main+0x2814>
   129bc: 13 85 41 95  	addi	a0, gp, -1708
   129c0: 83 25 05 00  	lw	a1, 0(a0)
   129c4: 9b 85 15 00  	addiw	a1, a1, 1
   129c8: 23 20 b5 00  	sw	a1, 0(a0)
   129cc: 17 f5 00 00  	auipc	a0, 15
   129d0: 13 05 95 ad  	addi	a0, a0, -1319
   129d4: 97 f5 00 00  	auipc	a1, 15
   129d8: 93 85 d5 aa  	addi	a1, a1, -1363
   129dc: 97 f6 00 00  	auipc	a3, 15
   129e0: 93 86 e6 de  	addi	a3, a3, -530
   129e4: 13 06 40 32  	addi	a2, zero, 804
   129e8: ef 00 b0 2d  	jal	0x134c2 <printf>
   129ec: 6f 00 00 03  	j	0x12a1c <main+0x283c>
   129f0: 13 07 00 00  	mv	a4, zero
   129f4: 9b 07 06 00  	sext.w	a5, a2
   129f8: 1b 08 05 00  	sext.w	a6, a0
   129fc: 17 f5 00 00  	auipc	a0, 15
   12a00: 13 05 45 a4  	addi	a0, a0, -1468
   12a04: 97 f5 00 00  	auipc	a1, 15
   12a08: 93 85 d5 a7  	addi	a1, a1, -1411
   12a0c: 97 f6 00 00  	auipc	a3, 15
   12a10: 93 86 e6 db  	addi	a3, a3, -578
   12a14: 13 06 40 32  	addi	a2, zero, 804
   12a18: ef 00 b0 2a  	jal	0x134c2 <printf>
   12a1c: 13 05 f0 ff  	addi	a0, zero, -1
   12a20: 23 30 a1 08  	sd	a0, 128(sp)
   12a24: 23 3c a1 06  	sd	a0, 120(sp)
   12a28: 23 38 a1 06  	sd	a0, 112(sp)
   12a2c: 23 34 a1 06  	sd	a0, 104(sp)
   12a30: 23 30 a1 06  	sd	a0, 96(sp)
   12a34: 23 3c a1 04  	sd	a0, 88(sp)
   12a38: 23 38 a1 04  	sd	a0, 80(sp)
   12a3c: 23 34 a1 04  	sd	a0, 72(sp)
   12a40: f7 75 14 00  	msettypei	a1, e32, m1, ba
   12a44: f7 75 01 20  	msettilemi	a1, 2
   12a48: f7 75 02 40  	msettileki	a1, 4
   12a4c: f7 75 02 60  	msettileni	a1, 4
   12a50: 93 05 00 01  	addi	a1, zero, 16
   12a54: 77 20 b4 04  	mlae32.m	tr0, (s0), a1, m1
   12a58: 13 06 81 04  	addi	a2, sp, 72
   12a5c: f7 20 b6 08  	mlbe32.m	tr1, (a2), a1, m1
   12a60: f7 f5 16 00  	msettypei	a1, e64, m2, ba
   12a64: f7 75 02 60  	msettileni	a1, 4
   12a68: 97 e5 00 00  	auipc	a1, 14
   12a6c: 93 85 85 59  	addi	a1, a1, 1432
   12a70: 13 06 00 02  	addi	a2, zero, 32
   12a74: 77 b5 c5 00  	mlce64.m	tr2, (a1), a2, m2
   12a78: f7 75 14 00  	msettypei	a1, e32, m1, ba
   12a7c: f7 75 02 60  	msettileni	a1, 4
   12a80: 77 61 1c 04  	mswma.mm	tr2, tr0, tr1
   12a84: f7 f5 16 00  	msettypei	a1, e64, m2, ba
   12a88: f7 75 02 60  	msettileni	a1, 4
   12a8c: 77 35 c9 02  	msce64.m	tr2, (s2), a2, m2
   12a90: 83 a5 09 00  	lw	a1, 0(s3)
   12a94: 03 38 09 00  	ld	a6, 0(s2)
   12a98: 9b 85 15 00  	addiw	a1, a1, 1
   12a9c: 93 57 15 00  	srli	a5, a0, 1
   12aa0: 23 a0 b9 00  	sw	a1, 0(s3)
   12aa4: 63 12 f8 0a  	bne	a6, a5, 0x12b48 <main+0x2968>
   12aa8: 03 38 89 00  	ld	a6, 8(s2)
   12aac: 13 07 10 00  	addi	a4, zero, 1
   12ab0: 63 1e f8 08  	bne	a6, a5, 0x12b4c <main+0x296c>
   12ab4: 03 38 09 01  	ld	a6, 16(s2)
   12ab8: 13 05 f0 ff  	addi	a0, zero, -1
   12abc: 13 55 15 00  	srli	a0, a0, 1
   12ac0: 13 07 20 00  	addi	a4, zero, 2
   12ac4: 63 14 a8 08  	bne	a6, a0, 0x12b4c <main+0x296c>
   12ac8: 03 38 89 01  	ld	a6, 24(s2)
   12acc: 13 07 30 00  	addi	a4, zero, 3
   12ad0: 63 1e a8 06  	bne	a6, a0, 0x12b4c <main+0x296c>
   12ad4: 03 38 09 02  	ld	a6, 32(s2)
   12ad8: 13 05 f0 ff  	addi	a0, zero, -1
   12adc: 13 55 15 00  	srli	a0, a0, 1
   12ae0: 13 07 40 00  	addi	a4, zero, 4
   12ae4: 63 14 a8 06  	bne	a6, a0, 0x12b4c <main+0x296c>
   12ae8: 03 38 89 02  	ld	a6, 40(s2)
   12aec: 13 07 50 00  	addi	a4, zero, 5
   12af0: 63 1e a8 04  	bne	a6, a0, 0x12b4c <main+0x296c>
   12af4: 03 38 09 03  	ld	a6, 48(s2)
   12af8: 13 05 f0 ff  	addi	a0, zero, -1
   12afc: 13 55 15 00  	srli	a0, a0, 1
   12b00: 13 07 60 00  	addi	a4, zero, 6
   12b04: 63 14 a8 04  	bne	a6, a0, 0x12b4c <main+0x296c>
   12b08: 03 38 89 03  	ld	a6, 56(s2)
   12b0c: 13 07 70 00  	addi	a4, zero, 7
   12b10: 63 1e a8 02  	bne	a6, a0, 0x12b4c <main+0x296c>
   12b14: 13 85 41 95  	addi	a0, gp, -1708
   12b18: 83 25 05 00  	lw	a1, 0(a0)
   12b1c: 9b 85 15 00  	addiw	a1, a1, 1
   12b20: 23 20 b5 00  	sw	a1, 0(a0)
   12b24: 17 f5 00 00  	auipc	a0, 15
   12b28: 13 05 15 98  	addi	a0, a0, -1663
   12b2c: 97 f5 00 00  	auipc	a1, 15
   12b30: 93 85 55 95  	addi	a1, a1, -1707
   12b34: 97 f6 00 00  	auipc	a3, 15
   12b38: 93 86 a6 ca  	addi	a3, a3, -854
   12b3c: 13 06 80 34  	addi	a2, zero, 840
   12b40: ef 00 30 18  	jal	0x134c2 <printf>
   12b44: 6f 00 80 02  	j	0x12b6c <main+0x298c>
   12b48: 13 07 00 00  	mv	a4, zero
   12b4c: 17 f5 00 00  	auipc	a0, 15
   12b50: 13 05 a5 a6  	addi	a0, a0, -1430
   12b54: 97 f5 00 00  	auipc	a1, 15
   12b58: 93 85 d5 92  	addi	a1, a1, -1747
   12b5c: 97 f6 00 00  	auipc	a3, 15
   12b60: 93 86 26 c8  	addi	a3, a3, -894
   12b64: 13 06 80 34  	addi	a2, zero, 840
   12b68: ef 00 b0 15  	jal	0x134c2 <printf>
   12b6c: 77 75 10 00  	msettypei	a0, e8, m1, ba
   12b70: 77 75 01 20  	msettilemi	a0, 2
   12b74: 77 75 04 40  	msettileki	a0, 8
   12b78: 77 75 08 60  	msettileni	a0, 16
   12b7c: 17 f5 00 00  	auipc	a0, 15
   12b80: 13 05 05 d2  	addi	a0, a0, -736
   12b84: 93 05 80 00  	addi	a1, zero, 8
   12b88: 77 02 b5 04  	mlae8.m	tr4, (a0), a1, m1
   12b8c: 17 e5 00 00  	auipc	a0, 14
   12b90: 13 05 45 4b  	addi	a0, a0, 1204
   12b94: 93 05 00 01  	addi	a1, zero, 16
   12b98: f7 02 b5 08  	mlbe8.m	tr5, (a0), a1, m1
   12b9c: 77 75 15 00  	msettypei	a0, e32, m4, ba
   12ba0: 77 75 08 60  	msettileni	a0, 16
   12ba4: 17 e5 00 00  	auipc	a0, 14
   12ba8: 13 05 c5 51  	addi	a0, a0, 1308
   12bac: 93 05 00 04  	addi	a1, zero, 64
   12bb0: 77 28 b5 00  	mlce32.m	tr0, (a0), a1, m4
   12bb4: 77 75 10 00  	msettypei	a0, e8, m1, ba
   12bb8: 77 75 08 60  	msettileni	a0, 16
   12bbc: 77 60 5e 08  	msqma.mm	tr0, tr4, tr5
   12bc0: 77 75 15 00  	msettypei	a0, e32, m4, ba
   12bc4: 77 75 08 60  	msettileni	a0, 16
   12bc8: 77 28 bc 02  	msce32.m	tr0, (s8), a1, m4
   12bcc: 83 a5 09 00  	lw	a1, 0(s3)
   12bd0: 13 05 00 00  	mv	a0, zero
   12bd4: 9b 85 15 00  	addiw	a1, a1, 1
   12bd8: 23 a0 b9 00  	sw	a1, 0(s3)
   12bdc: 97 e5 00 00  	auipc	a1, 14
   12be0: 93 85 45 56  	addi	a1, a1, 1380
   12be4: 13 06 00 02  	addi	a2, zero, 32
   12be8: 03 a7 05 00  	lw	a4, 0(a1)
   12bec: 83 26 0c 00  	lw	a3, 0(s8)
   12bf0: 63 14 d7 04  	bne	a4, a3, 0x12c38 <main+0x2a58>
   12bf4: 13 05 15 00  	addi	a0, a0, 1
   12bf8: 13 0c 4c 00  	addi	s8, s8, 4
   12bfc: 93 85 45 00  	addi	a1, a1, 4
   12c00: e3 14 c5 fe  	bne	a0, a2, 0x12be8 <main+0x2a08>
   12c04: 13 85 41 95  	addi	a0, gp, -1708
   12c08: 83 25 05 00  	lw	a1, 0(a0)
   12c0c: 9b 85 15 00  	addiw	a1, a1, 1
   12c10: 23 20 b5 00  	sw	a1, 0(a0)
   12c14: 17 f5 00 00  	auipc	a0, 15
   12c18: 13 05 15 89  	addi	a0, a0, -1903
   12c1c: 97 f5 00 00  	auipc	a1, 15
   12c20: 93 85 55 86  	addi	a1, a1, -1947
   12c24: 97 f6 00 00  	auipc	a3, 15
   12c28: 93 86 26 b9  	addi	a3, a3, -1134
   12c2c: 13 06 10 37  	addi	a2, zero, 881
   12c30: ef 00 30 09  	jal	0x134c2 <printf>
   12c34: 6f 00 00 04  	j	0x12c74 <main+0x2a94>
   12c38: 93 15 07 02  	slli	a1, a4, 32
   12c3c: 93 d5 05 02  	srli	a1, a1, 32
   12c40: 13 96 06 02  	slli	a2, a3, 32
   12c44: 13 56 06 02  	srli	a2, a2, 32
   12c48: 1b 07 05 00  	sext.w	a4, a0
   12c4c: 9b 87 05 00  	sext.w	a5, a1
   12c50: 1b 08 06 00  	sext.w	a6, a2
   12c54: 17 e5 00 00  	auipc	a0, 14
   12c58: 13 05 c5 7e  	addi	a0, a0, 2028
   12c5c: 97 f5 00 00  	auipc	a1, 15
   12c60: 93 85 55 82  	addi	a1, a1, -2011
   12c64: 97 f6 00 00  	auipc	a3, 15
   12c68: 93 86 26 b5  	addi	a3, a3, -1198
   12c6c: 13 06 10 37  	addi	a2, zero, 881
   12c70: ef 00 30 05  	jal	0x134c2 <printf>
   12c74: 77 75 12 00  	msettypei	a0, e16, m1, ba
   12c78: 77 75 01 20  	msettilemi	a0, 2
   12c7c: 77 75 04 40  	msettileki	a0, 8
   12c80: 77 75 04 60  	msettileni	a0, 8
   12c84: 17 f5 00 00  	auipc	a0, 15
   12c88: 13 05 c5 ee  	addi	a0, a0, -276
   12c8c: 93 05 00 01  	addi	a1, zero, 16
   12c90: 77 12 b5 04  	mlae16.m	tr4, (a0), a1, m1
   12c94: 17 e5 00 00  	auipc	a0, 14
   12c98: 13 05 c5 52  	addi	a0, a0, 1324
   12c9c: f7 12 b5 08  	mlbe16.m	tr5, (a0), a1, m1
   12ca0: 77 75 17 00  	msettypei	a0, e64, m4, ba
   12ca4: 77 75 04 60  	msettileni	a0, 8
   12ca8: 17 e5 00 00  	auipc	a0, 14
   12cac: 13 05 85 59  	addi	a0, a0, 1432
   12cb0: 93 05 00 04  	addi	a1, zero, 64
   12cb4: 77 38 b5 00  	mlce64.m	tr0, (a0), a1, m4
   12cb8: 77 75 12 00  	msettypei	a0, e16, m1, ba
   12cbc: 77 75 04 60  	msettileni	a0, 8
   12cc0: 77 60 5e 08  	msqma.mm	tr0, tr4, tr5
   12cc4: 77 75 17 00  	msettypei	a0, e64, m4, ba
   12cc8: 77 75 04 60  	msettileni	a0, 8
   12ccc: 77 38 b9 02  	msce64.m	tr0, (s2), a1, m4
   12cd0: 03 a5 09 00  	lw	a0, 0(s3)
   12cd4: 03 38 09 00  	ld	a6, 0(s2)
   12cd8: 1b 05 15 00  	addiw	a0, a0, 1
   12cdc: b7 65 42 f3  	lui	a1, 996390
   12ce0: 9b 87 65 c8  	addiw	a5, a1, -890
   12ce4: 23 a0 a9 00  	sw	a0, 0(s3)
   12ce8: 63 12 f8 16  	bne	a6, a5, 0x12e4c <main+0x2c6c>
   12cec: 03 38 89 00  	ld	a6, 8(s2)
   12cf0: 37 d5 83 20  	lui	a0, 133181
   12cf4: 9b 07 d5 bb  	addiw	a5, a0, -1091
   12cf8: 13 07 10 00  	addi	a4, zero, 1
   12cfc: 63 1a f8 14  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d00: 03 38 09 01  	ld	a6, 16(s2)
   12d04: 37 f5 0d 19  	lui	a0, 102623
   12d08: 9b 07 d5 86  	addiw	a5, a0, -1939
   12d0c: 13 07 20 00  	addi	a4, zero, 2
   12d10: 63 10 f8 14  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d14: 03 38 89 01  	ld	a6, 24(s2)
   12d18: 37 b5 f7 3d  	lui	a0, 253819
   12d1c: 9b 07 f5 74  	addiw	a5, a0, 1871
   12d20: 13 07 30 00  	addi	a4, zero, 3
   12d24: 63 16 f8 12  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d28: 03 38 09 02  	ld	a6, 32(s2)
   12d2c: 37 25 2a 24  	lui	a0, 148130
   12d30: 9b 07 85 45  	addiw	a5, a0, 1112
   12d34: 13 07 40 00  	addi	a4, zero, 4
   12d38: 63 1c f8 10  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d3c: 03 38 89 02  	ld	a6, 40(s2)
   12d40: 37 65 9a dd  	lui	a0, 907686
   12d44: 9b 07 c5 50  	addiw	a5, a0, 1292
   12d48: 13 07 50 00  	addi	a4, zero, 5
   12d4c: 63 12 f8 10  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d50: 03 38 09 03  	ld	a6, 48(s2)
   12d54: 37 85 51 69  	lui	a0, 431384
   12d58: 9b 07 35 25  	addiw	a5, a0, 595
   12d5c: 13 07 60 00  	addi	a4, zero, 6
   12d60: 63 18 f8 0e  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d64: 03 38 89 03  	ld	a6, 56(s2)
   12d68: 37 35 87 de  	lui	a0, 911475
   12d6c: 9b 07 35 b2  	addiw	a5, a0, -1245
   12d70: 13 07 70 00  	addi	a4, zero, 7
   12d74: 63 1e f8 0c  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d78: 03 38 09 04  	ld	a6, 64(s2)
   12d7c: 37 05 f7 4d  	lui	a0, 319344
   12d80: 9b 07 05 fd  	addiw	a5, a0, -48
   12d84: 13 07 80 00  	addi	a4, zero, 8
   12d88: 63 14 f8 0c  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12d8c: 03 38 89 04  	ld	a6, 72(s2)
   12d90: 37 15 b2 c5  	lui	a0, 809761
   12d94: 9b 07 15 59  	addiw	a5, a0, 1425
   12d98: 13 07 90 00  	addi	a4, zero, 9
   12d9c: 63 1a f8 0a  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12da0: 03 38 09 05  	ld	a6, 80(s2)
   12da4: 37 75 8b ec  	lui	a0, 968887
   12da8: 9b 07 a5 66  	addiw	a5, a0, 1642
   12dac: 13 07 a0 00  	addi	a4, zero, 10
   12db0: 63 10 f8 0a  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12db4: 03 38 89 05  	ld	a6, 88(s2)
   12db8: 37 75 13 e5  	lui	a0, 938295
   12dbc: 9b 07 15 70  	addiw	a5, a0, 1793
   12dc0: 13 07 b0 00  	addi	a4, zero, 11
   12dc4: 63 16 f8 08  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12dc8: 03 38 09 06  	ld	a6, 96(s2)
   12dcc: 37 15 6a 92  	lui	a0, 599713
   12dd0: 9b 07 25 20  	addiw	a5, a0, 514
   12dd4: 13 07 c0 00  	addi	a4, zero, 12
   12dd8: 63 1c f8 06  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12ddc: 03 38 89 06  	ld	a6, 104(s2)
   12de0: 37 b5 1a 2d  	lui	a0, 184747
   12de4: 9b 07 55 bb  	addiw	a5, a0, -1099
   12de8: 13 07 d0 00  	addi	a4, zero, 13
   12dec: 63 12 f8 06  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12df0: 03 38 09 07  	ld	a6, 112(s2)
   12df4: 37 25 87 10  	lui	a0, 67698
   12df8: 9b 07 a5 fa  	addiw	a5, a0, -86
   12dfc: 13 07 e0 00  	addi	a4, zero, 14
   12e00: 63 18 f8 04  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12e04: 03 38 89 07  	ld	a6, 120(s2)
   12e08: 37 85 76 ac  	lui	a0, 706408
   12e0c: 9b 07 b5 83  	addiw	a5, a0, -1989
   12e10: 13 07 f0 00  	addi	a4, zero, 15
   12e14: 63 1e f8 02  	bne	a6, a5, 0x12e50 <main+0x2c70>
   12e18: 13 85 41 95  	addi	a0, gp, -1708
   12e1c: 83 25 05 00  	lw	a1, 0(a0)
   12e20: 9b 85 15 00  	addiw	a1, a1, 1
   12e24: 23 20 b5 00  	sw	a1, 0(a0)
   12e28: 17 e5 00 00  	auipc	a0, 14
   12e2c: 13 05 d5 67  	addi	a0, a0, 1661
   12e30: 97 e5 00 00  	auipc	a1, 14
   12e34: 93 85 15 65  	addi	a1, a1, 1617
   12e38: 97 f6 00 00  	auipc	a3, 15
   12e3c: 93 86 26 99  	addi	a3, a3, -1646
   12e40: 13 06 50 39  	addi	a2, zero, 917
   12e44: ef 00 e0 67  	jal	0x134c2 <printf>
   12e48: 6f 00 80 02  	j	0x12e70 <main+0x2c90>
   12e4c: 13 07 00 00  	mv	a4, zero
   12e50: 17 e5 00 00  	auipc	a0, 14
   12e54: 13 05 65 76  	addi	a0, a0, 1894
   12e58: 97 e5 00 00  	auipc	a1, 14
   12e5c: 93 85 95 62  	addi	a1, a1, 1577
   12e60: 97 f6 00 00  	auipc	a3, 15
   12e64: 93 86 a6 96  	addi	a3, a3, -1686
   12e68: 13 06 50 39  	addi	a2, zero, 917
   12e6c: ef 00 60 65  	jal	0x134c2 <printf>
   12e70: 77 75 12 00  	msettypei	a0, e16, m1, ba
   12e74: 77 75 01 20  	msettilemi	a0, 2
   12e78: 77 75 04 40  	msettileki	a0, 8
   12e7c: 77 75 04 60  	msettileni	a0, 8
   12e80: 97 f5 00 00  	auipc	a1, 15
   12e84: 93 85 05 d1  	addi	a1, a1, -752
   12e88: 13 05 00 01  	addi	a0, zero, 16
   12e8c: 77 90 a5 04  	mlae16.m	tr0, (a1), a0, m1
   12e90: 97 e5 00 00  	auipc	a1, 14
   12e94: 93 85 05 43  	addi	a1, a1, 1072
   12e98: f7 90 a5 08  	mlbe16.m	tr1, (a1), a0, m1
   12e9c: f7 f5 14 00  	msettypei	a1, e32, m2, ba
   12ea0: f7 75 04 60  	msettileni	a1, 8
   12ea4: 97 e5 00 00  	auipc	a1, 14
   12ea8: 93 85 c5 49  	addi	a1, a1, 1180
   12eac: 13 06 00 02  	addi	a2, zero, 32
   12eb0: 77 a5 c5 00  	mlce32.m	tr2, (a1), a2, m2
   12eb4: f7 75 12 00  	msettypei	a1, e16, m1, ba
   12eb8: f7 75 04 60  	msettileni	a1, 8
   12ebc: 77 61 10 06  	mfwma.mm	tr2, tr0, tr1
   12ec0: f7 f5 14 00  	msettypei	a1, e32, m2, ba
   12ec4: f7 75 04 60  	msettileni	a1, 8
   12ec8: 77 25 cb 02  	msce32.m	tr2, (s6), a2, m2
   12ecc: 03 a6 09 00  	lw	a2, 0(s3)
   12ed0: 93 05 00 00  	mv	a1, zero
   12ed4: 1b 06 16 00  	addiw	a2, a2, 1
   12ed8: 23 a0 c9 00  	sw	a2, 0(s3)
   12edc: 17 e6 00 00  	auipc	a2, 14
   12ee0: 13 06 46 4a  	addi	a2, a2, 1188
   12ee4: 07 20 06 00  	flw	ft0, 0(a2)
   12ee8: 87 20 0b 00  	flw	ft1, 0(s6)
   12eec: 53 71 10 08  	fsub.s	ft2, ft0, ft1
   12ef0: 53 01 01 42  	fcvt.d.s	ft2, ft2
   12ef4: d3 16 24 a2  	flt.d	a3, fs0, ft2
   12ef8: 53 17 91 a2  	flt.d	a4, ft2, fs1
   12efc: b3 e6 e6 00  	or	a3, a3, a4
   12f00: 63 94 06 04  	bnez	a3, 0x12f48 <main+0x2d68>
   12f04: 93 85 15 00  	addi	a1, a1, 1
   12f08: 13 0b 4b 00  	addi	s6, s6, 4
   12f0c: 13 06 46 00  	addi	a2, a2, 4
   12f10: e3 9a a5 fc  	bne	a1, a0, 0x12ee4 <main+0x2d04>
   12f14: 13 85 41 95  	addi	a0, gp, -1708
   12f18: 83 25 05 00  	lw	a1, 0(a0)
   12f1c: 9b 85 15 00  	addiw	a1, a1, 1
   12f20: 23 20 b5 00  	sw	a1, 0(a0)
   12f24: 17 e5 00 00  	auipc	a0, 14
   12f28: 13 05 15 58  	addi	a0, a0, 1409
   12f2c: 97 e5 00 00  	auipc	a1, 14
   12f30: 93 85 55 55  	addi	a1, a1, 1365
   12f34: 97 f6 00 00  	auipc	a3, 15
   12f38: 93 86 e6 8b  	addi	a3, a3, -1858
   12f3c: 13 06 e0 3b  	addi	a2, zero, 958
   12f40: ef 00 20 58  	jal	0x134c2 <printf>
   12f44: 6f 00 80 03  	j	0x12f7c <main+0x2d9c>
   12f48: 53 00 00 42  	fcvt.d.s	ft0, ft0
   12f4c: d3 80 00 42  	fcvt.d.s	ft1, ft1
   12f50: 1b 87 05 00  	sext.w	a4, a1
   12f54: d3 07 00 e2  	fmv.x.d	a5, ft0
   12f58: 53 88 00 e2  	fmv.x.d	a6, ft1
   12f5c: 17 e5 00 00  	auipc	a0, 14
   12f60: 13 05 15 6b  	addi	a0, a0, 1713
   12f64: 97 e5 00 00  	auipc	a1, 14
   12f68: 93 85 d5 51  	addi	a1, a1, 1309
   12f6c: 97 f6 00 00  	auipc	a3, 15
   12f70: 93 86 66 88  	addi	a3, a3, -1914
   12f74: 13 06 e0 3b  	addi	a2, zero, 958
   12f78: ef 00 a0 54  	jal	0x134c2 <printf>
   12f7c: 77 75 14 00  	msettypei	a0, e32, m1, ba
   12f80: 77 75 01 20  	msettilemi	a0, 2
   12f84: 77 75 02 40  	msettileki	a0, 4
   12f88: 77 75 02 60  	msettileni	a0, 4
   12f8c: 17 f5 00 00  	auipc	a0, 15
   12f90: 13 05 45 c2  	addi	a0, a0, -988
   12f94: 93 05 00 01  	addi	a1, zero, 16
   12f98: 77 20 b5 04  	mlae32.m	tr0, (a0), a1, m1
   12f9c: 17 e5 00 00  	auipc	a0, 14
   12fa0: 13 05 45 42  	addi	a0, a0, 1060
   12fa4: f7 20 b5 08  	mlbe32.m	tr1, (a0), a1, m1
   12fa8: 77 f5 16 00  	msettypei	a0, e64, m2, ba
   12fac: 77 75 02 60  	msettileni	a0, 4
   12fb0: 17 e5 00 00  	auipc	a0, 14
   12fb4: 13 05 05 45  	addi	a0, a0, 1104
   12fb8: 93 05 00 02  	addi	a1, zero, 32
   12fbc: 77 35 b5 00  	mlce64.m	tr2, (a0), a1, m2
   12fc0: 77 75 14 00  	msettypei	a0, e32, m1, ba
   12fc4: 77 75 02 60  	msettileni	a0, 4
   12fc8: 77 61 10 06  	mfwma.mm	tr2, tr0, tr1
   12fcc: 77 f5 16 00  	msettypei	a0, e64, m2, ba
   12fd0: 77 75 02 60  	msettileni	a0, 4
   12fd4: 77 35 ba 02  	msce64.m	tr2, (s4), a1, m2
   12fd8: 07 30 0a 00  	fld	ft0, 0(s4)
   12fdc: 13 85 81 8f  	addi	a0, gp, -1800
   12fe0: 87 30 05 00  	fld	ft1, 0(a0)
   12fe4: 13 07 00 00  	mv	a4, zero
   12fe8: 03 a5 09 00  	lw	a0, 0(s3)
   12fec: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   12ff0: d3 15 24 a2  	flt.d	a1, fs0, ft2
   12ff4: 53 16 91 a2  	flt.d	a2, ft2, fs1
   12ff8: b3 e5 c5 00  	or	a1, a1, a2
   12ffc: 1b 05 15 00  	addiw	a0, a0, 1
   13000: 23 a0 a9 00  	sw	a0, 0(s3)
   13004: 63 90 05 10  	bnez	a1, 0x13104 <main+0x2f24>
   13008: 07 30 8a 00  	fld	ft0, 8(s4)
   1300c: 13 85 01 90  	addi	a0, gp, -1792
   13010: 87 30 05 00  	fld	ft1, 0(a0)
   13014: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   13018: 53 15 24 a2  	flt.d	a0, fs0, ft2
   1301c: d3 15 91 a2  	flt.d	a1, ft2, fs1
   13020: 33 65 b5 00  	or	a0, a0, a1
   13024: 13 07 10 00  	addi	a4, zero, 1
   13028: 63 1e 05 0c  	bnez	a0, 0x13104 <main+0x2f24>
   1302c: 07 30 0a 01  	fld	ft0, 16(s4)
   13030: 13 85 81 90  	addi	a0, gp, -1784
   13034: 87 30 05 00  	fld	ft1, 0(a0)
   13038: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   1303c: 53 15 24 a2  	flt.d	a0, fs0, ft2
   13040: d3 15 91 a2  	flt.d	a1, ft2, fs1
   13044: 33 65 b5 00  	or	a0, a0, a1
   13048: 13 07 20 00  	addi	a4, zero, 2
   1304c: 63 1c 05 0a  	bnez	a0, 0x13104 <main+0x2f24>
   13050: 07 30 8a 01  	fld	ft0, 24(s4)
   13054: 13 85 01 91  	addi	a0, gp, -1776
   13058: 87 30 05 00  	fld	ft1, 0(a0)
   1305c: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   13060: 53 15 24 a2  	flt.d	a0, fs0, ft2
   13064: d3 15 91 a2  	flt.d	a1, ft2, fs1
   13068: 33 65 b5 00  	or	a0, a0, a1
   1306c: 13 07 30 00  	addi	a4, zero, 3
   13070: 63 1a 05 08  	bnez	a0, 0x13104 <main+0x2f24>
   13074: 07 30 0a 02  	fld	ft0, 32(s4)
   13078: 13 85 81 91  	addi	a0, gp, -1768
   1307c: 87 30 05 00  	fld	ft1, 0(a0)
   13080: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   13084: 53 15 24 a2  	flt.d	a0, fs0, ft2
   13088: d3 15 91 a2  	flt.d	a1, ft2, fs1
   1308c: 33 65 b5 00  	or	a0, a0, a1
   13090: 13 07 40 00  	addi	a4, zero, 4
   13094: 63 18 05 06  	bnez	a0, 0x13104 <main+0x2f24>
   13098: 07 30 8a 02  	fld	ft0, 40(s4)
   1309c: 13 85 01 92  	addi	a0, gp, -1760
   130a0: 87 30 05 00  	fld	ft1, 0(a0)
   130a4: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   130a8: 53 15 24 a2  	flt.d	a0, fs0, ft2
   130ac: d3 15 91 a2  	flt.d	a1, ft2, fs1
   130b0: 33 65 b5 00  	or	a0, a0, a1
   130b4: 13 07 50 00  	addi	a4, zero, 5
   130b8: 63 16 05 04  	bnez	a0, 0x13104 <main+0x2f24>
   130bc: 07 30 0a 03  	fld	ft0, 48(s4)
   130c0: 13 85 81 92  	addi	a0, gp, -1752
   130c4: 87 30 05 00  	fld	ft1, 0(a0)
   130c8: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   130cc: 53 15 24 a2  	flt.d	a0, fs0, ft2
   130d0: d3 15 91 a2  	flt.d	a1, ft2, fs1
   130d4: 33 65 b5 00  	or	a0, a0, a1
   130d8: 13 07 60 00  	addi	a4, zero, 6
   130dc: 63 14 05 02  	bnez	a0, 0x13104 <main+0x2f24>
   130e0: 07 30 8a 03  	fld	ft0, 56(s4)
   130e4: 13 85 01 93  	addi	a0, gp, -1744
   130e8: 87 30 05 00  	fld	ft1, 0(a0)
   130ec: 53 f1 00 0a  	fsub.d	ft2, ft1, ft0
   130f0: 53 15 24 a2  	flt.d	a0, fs0, ft2
   130f4: d3 15 91 a2  	flt.d	a1, ft2, fs1
   130f8: 33 65 b5 00  	or	a0, a0, a1
   130fc: 13 07 70 00  	addi	a4, zero, 7
   13100: 63 02 05 16  	beqz	a0, 0x13264 <main+0x3084>
   13104: d3 87 00 e2  	fmv.x.d	a5, ft1
   13108: 53 08 00 e2  	fmv.x.d	a6, ft0
   1310c: 17 e5 00 00  	auipc	a0, 14
   13110: 13 05 15 50  	addi	a0, a0, 1281
   13114: 97 e5 00 00  	auipc	a1, 14
   13118: 93 85 d5 36  	addi	a1, a1, 877
   1311c: 97 e6 00 00  	auipc	a3, 14
   13120: 93 86 a6 6e  	addi	a3, a3, 1770
   13124: 13 06 c0 3d  	addi	a2, zero, 988
   13128: ef 00 a0 39  	jal	0x134c2 <printf>
   1312c: 13 05 00 00  	mv	a0, zero
   13130: 87 34 81 08  	fld	fs1, 136(sp)
   13134: 07 34 01 09  	fld	fs0, 144(sp)
   13138: 83 3d 81 09  	ld	s11, 152(sp)
   1313c: 03 3d 01 0a  	ld	s10, 160(sp)
   13140: 83 3c 81 0a  	ld	s9, 168(sp)
   13144: 03 3c 01 0b  	ld	s8, 176(sp)
   13148: 83 3b 81 0b  	ld	s7, 184(sp)
   1314c: 03 3b 01 0c  	ld	s6, 192(sp)
   13150: 83 3a 81 0c  	ld	s5, 200(sp)
   13154: 03 3a 01 0d  	ld	s4, 208(sp)
   13158: 83 39 81 0d  	ld	s3, 216(sp)
   1315c: 03 39 01 0e  	ld	s2, 224(sp)
   13160: 83 34 81 0e  	ld	s1, 232(sp)
   13164: 03 34 01 0f  	ld	s0, 240(sp)
   13168: 83 30 81 0f  	ld	ra, 248(sp)
   1316c: 13 01 01 10  	addi	sp, sp, 256
   13170: 67 80 00 00  	ret
   13174: 13 07 40 00  	addi	a4, zero, 4
   13178: b7 95 1f e9  	lui	a1, 954873
   1317c: 1b 86 e5 26  	addiw	a2, a1, 622
   13180: 6f f0 5f 87  	j	0x129f4 <main+0x2814>
   13184: 13 85 41 95  	addi	a0, gp, -1708
   13188: 83 25 05 00  	lw	a1, 0(a0)
   1318c: 9b 85 15 00  	addiw	a1, a1, 1
   13190: 23 20 b5 00  	sw	a1, 0(a0)
   13194: 17 e5 00 00  	auipc	a0, 14
   13198: 13 05 15 31  	addi	a0, a0, 785
   1319c: 97 e5 00 00  	auipc	a1, 14
   131a0: 93 85 55 2e  	addi	a1, a1, 741
   131a4: 97 e6 00 00  	auipc	a3, 14
   131a8: 93 86 26 57  	addi	a3, a3, 1394
   131ac: 13 06 c0 18  	addi	a2, zero, 396
   131b0: ef 00 20 31  	jal	0x134c2 <printf>
   131b4: 6f e0 1f 81  	j	0x119c4 <main+0x17e4>
   131b8: 13 85 41 95  	addi	a0, gp, -1708
   131bc: 83 25 05 00  	lw	a1, 0(a0)
   131c0: 9b 85 15 00  	addiw	a1, a1, 1
   131c4: 23 20 b5 00  	sw	a1, 0(a0)
   131c8: 17 e5 00 00  	auipc	a0, 14
   131cc: 13 05 d5 2d  	addi	a0, a0, 733
   131d0: 97 e5 00 00  	auipc	a1, 14
   131d4: 93 85 15 2b  	addi	a1, a1, 689
   131d8: 97 e6 00 00  	auipc	a3, 14
   131dc: 93 86 66 47  	addi	a3, a3, 1142
   131e0: 13 06 f0 09  	addi	a2, zero, 159
   131e4: ef 00 e0 2d  	jal	0x134c2 <printf>
   131e8: 6f d0 9f ad  	j	0x10cc0 <main+0xae0>
   131ec: 13 85 41 95  	addi	a0, gp, -1708
   131f0: 83 25 05 00  	lw	a1, 0(a0)
   131f4: 9b 85 15 00  	addiw	a1, a1, 1
   131f8: 23 20 b5 00  	sw	a1, 0(a0)
   131fc: 17 e5 00 00  	auipc	a0, 14
   13200: 13 05 95 2a  	addi	a0, a0, 681
   13204: 97 e5 00 00  	auipc	a1, 14
   13208: 93 85 d5 27  	addi	a1, a1, 637
   1320c: 97 e6 00 00  	auipc	a3, 14
   13210: 93 86 66 45  	addi	a3, a3, 1110
   13214: 13 06 30 0b  	addi	a2, zero, 179
   13218: ef 00 a0 2a  	jal	0x134c2 <printf>
   1321c: 6f d0 1f c6  	j	0x10e7c <main+0xc9c>
   13220: 13 85 41 95  	addi	a0, gp, -1708
   13224: 83 25 05 00  	lw	a1, 0(a0)
   13228: 9b 85 15 00  	addiw	a1, a1, 1
   1322c: 23 20 b5 00  	sw	a1, 0(a0)
   13230: 17 e5 00 00  	auipc	a0, 14
   13234: 13 05 55 27  	addi	a0, a0, 629
   13238: 97 e5 00 00  	auipc	a1, 14
   1323c: 93 85 95 24  	addi	a1, a1, 585
   13240: 97 e6 00 00  	auipc	a3, 14
   13244: 93 86 66 43  	addi	a3, a3, 1078
   13248: 13 06 50 0c  	addi	a2, zero, 197
   1324c: ef 00 60 27  	jal	0x134c2 <printf>
   13250: 6f d0 5f f2  	j	0x11174 <main+0xf94>
   13254: 13 07 70 00  	addi	a4, zero, 7
   13258: b7 05 48 9b  	lui	a1, 636032
   1325c: 1b 86 65 f5  	addiw	a2, a1, -170
   13260: 6f f0 4f f9  	j	0x129f4 <main+0x2814>
   13264: 13 85 41 95  	addi	a0, gp, -1708
   13268: 83 25 05 00  	lw	a1, 0(a0)
   1326c: 9b 85 15 00  	addiw	a1, a1, 1
   13270: 23 20 b5 00  	sw	a1, 0(a0)
   13274: 17 e5 00 00  	auipc	a0, 14
   13278: 13 05 15 23  	addi	a0, a0, 561
   1327c: 97 e5 00 00  	auipc	a1, 14
   13280: 93 85 55 20  	addi	a1, a1, 517
   13284: 97 e6 00 00  	auipc	a3, 14
   13288: 93 86 26 58  	addi	a3, a3, 1410
   1328c: 13 06 c0 3d  	addi	a2, zero, 988
   13290: ef 00 20 23  	jal	0x134c2 <printf>
   13294: 6f f0 9f e9  	j	0x1312c <main+0x2f4c>
   13298: 13 07 80 00  	addi	a4, zero, 8
   1329c: b7 45 19 d0  	lui	a1, 852372
   132a0: 1b 86 15 d9  	addiw	a2, a1, -623
   132a4: 6f f0 0f f5  	j	0x129f4 <main+0x2814>
   132a8: 13 07 90 00  	addi	a4, zero, 9
   132ac: b7 75 b5 e8  	lui	a1, 953175
   132b0: 1b 86 b5 0c  	addiw	a2, a1, 203
   132b4: 6f f0 0f f4  	j	0x129f4 <main+0x2814>
   132b8: 13 07 a0 00  	addi	a4, zero, 10
   132bc: b7 15 fe e9  	lui	a1, 958433
   132c0: 1b 86 e5 df  	addiw	a2, a1, -514
   132c4: 6f f0 0f f3  	j	0x129f4 <main+0x2814>
   132c8: 13 85 41 95  	addi	a0, gp, -1708
   132cc: 83 25 05 00  	lw	a1, 0(a0)
   132d0: 9b 85 15 00  	addiw	a1, a1, 1
   132d4: 23 20 b5 00  	sw	a1, 0(a0)
   132d8: 17 e5 00 00  	auipc	a0, 14
   132dc: 13 05 d5 1c  	addi	a0, a0, 461
   132e0: 97 e5 00 00  	auipc	a1, 14
   132e4: 93 85 15 1a  	addi	a1, a1, 417
   132e8: 97 e6 00 00  	auipc	a3, 14
   132ec: 93 86 a6 4b  	addi	a3, a3, 1210
   132f0: 13 06 a0 28  	addi	a2, zero, 650
   132f4: ef 00 e0 1c  	jal	0x134c2 <printf>
   132f8: 6f f0 0f 8a  	j	0x12398 <main+0x21b8>
   132fc: 13 07 c0 00  	addi	a4, zero, 12
   13300: b7 75 5b 92  	lui	a1, 599479
   13304: 1b 86 c5 dc  	addiw	a2, a1, -564
   13308: 6f f0 cf ee  	j	0x129f4 <main+0x2814>
   1330c: 13 07 e0 00  	addi	a4, zero, 14
   13310: b7 a5 50 e2  	lui	a1, 926986
   13314: 1b 86 45 8e  	addiw	a2, a1, -1820
   13318: 6f f0 cf ed  	j	0x129f4 <main+0x2814>
   1331c: 13 85 41 95  	addi	a0, gp, -1708
   13320: 83 25 05 00  	lw	a1, 0(a0)
   13324: 9b 85 15 00  	addiw	a1, a1, 1
   13328: 23 20 b5 00  	sw	a1, 0(a0)
   1332c: 17 e5 00 00  	auipc	a0, 14
   13330: 13 05 95 17  	addi	a0, a0, 377
   13334: 97 e5 00 00  	auipc	a1, 14
   13338: 93 85 d5 14  	addi	a1, a1, 333
   1333c: 97 e6 00 00  	auipc	a3, 14
   13340: 93 86 66 3c  	addi	a3, a3, 966
   13344: 13 06 90 17  	addi	a2, zero, 377
   13348: ef 00 a0 17  	jal	0x134c2 <printf>
   1334c: 6f e0 8f dc  	j	0x11914 <main+0x1734>
   13350: 13 85 41 95  	addi	a0, gp, -1708
   13354: 83 25 05 00  	lw	a1, 0(a0)
   13358: 9b 85 15 00  	addiw	a1, a1, 1
   1335c: 23 20 b5 00  	sw	a1, 0(a0)
   13360: 17 e5 00 00  	auipc	a0, 14
   13364: 13 05 55 14  	addi	a0, a0, 325
   13368: 97 e5 00 00  	auipc	a1, 14
   1336c: 93 85 95 11  	addi	a1, a1, 281
   13370: 97 e6 00 00  	auipc	a3, 14
   13374: 93 86 e6 41  	addi	a3, a3, 1054
   13378: 13 06 b0 26  	addi	a2, zero, 619
   1337c: ef 00 60 14  	jal	0x134c2 <printf>
   13380: 6f e0 9f ef  	j	0x12278 <main+0x2098>

0000000000013384 <__libc_init_array>:
   13384: 01 11        	addi	sp, sp, -32
   13386: 22 e8        	sd	s0, 16(sp)
   13388: b7 37 02 00  	lui	a5, 35
   1338c: 37 34 02 00  	lui	s0, 35
   13390: 4a e0        	sd	s2, 0(sp)
   13392: 93 87 47 00  	addi	a5, a5, 4
   13396: 13 07 44 00  	addi	a4, s0, 4
   1339a: 06 ec        	sd	ra, 24(sp)
   1339c: 26 e4        	sd	s1, 8(sp)
   1339e: 33 89 e7 40  	sub	s2, a5, a4
   133a2: 63 8d e7 00  	beq	a5, a4, 0x133bc <__libc_init_array+0x38>
   133a6: 13 59 39 40  	srai	s2, s2, 3
   133aa: 13 04 44 00  	addi	s0, s0, 4
   133ae: 81 44        	mv	s1, zero
   133b0: 1c 60        	ld	a5, 0(s0)
   133b2: 85 04        	addi	s1, s1, 1
   133b4: 21 04        	addi	s0, s0, 8
   133b6: 82 97        	jalr	a5
   133b8: e3 ec 24 ff  	bltu	s1, s2, 0x133b0 <__libc_init_array+0x2c>
   133bc: 37 34 02 00  	lui	s0, 35
   133c0: b7 37 02 00  	lui	a5, 35
   133c4: 93 87 87 01  	addi	a5, a5, 24
   133c8: 13 07 84 00  	addi	a4, s0, 8
   133cc: 33 89 e7 40  	sub	s2, a5, a4
   133d0: 13 59 39 40  	srai	s2, s2, 3
   133d4: 63 8b e7 00  	beq	a5, a4, 0x133ea <__libc_init_array+0x66>
   133d8: 13 04 84 00  	addi	s0, s0, 8
   133dc: 81 44        	mv	s1, zero
   133de: 1c 60        	ld	a5, 0(s0)
   133e0: 85 04        	addi	s1, s1, 1
   133e2: 21 04        	addi	s0, s0, 8
   133e4: 82 97        	jalr	a5
   133e6: e3 ec 24 ff  	bltu	s1, s2, 0x133de <__libc_init_array+0x5a>
   133ea: e2 60        	ld	ra, 24(sp)
   133ec: 42 64        	ld	s0, 16(sp)
   133ee: a2 64        	ld	s1, 8(sp)
   133f0: 02 69        	ld	s2, 0(sp)
   133f2: 05 61        	addi	sp, sp, 32
   133f4: 82 80        	ret

00000000000133f6 <memset>:
   133f6: 3d 43        	addi	t1, zero, 15
   133f8: 2a 87        	mv	a4, a0
   133fa: 63 71 c3 02  	bgeu	t1, a2, 0x1341c <memset+0x26>
   133fe: 93 77 f7 00  	andi	a5, a4, 15
   13402: c1 e3        	bnez	a5, 0x13482 <memset+0x8c>
   13404: bd e1        	bnez	a1, 0x1346a <memset+0x74>
   13406: 93 76 06 ff  	andi	a3, a2, -16
   1340a: 3d 8a        	andi	a2, a2, 15
   1340c: ba 96        	add	a3, a3, a4
   1340e: 0c e3        	sd	a1, 0(a4)
   13410: 0c e7        	sd	a1, 8(a4)
   13412: 41 07        	addi	a4, a4, 16
   13414: e3 6d d7 fe  	bltu	a4, a3, 0x1340e <memset+0x18>
   13418: 11 e2        	bnez	a2, 0x1341c <memset+0x26>
   1341a: 82 80        	ret
   1341c: b3 06 c3 40  	sub	a3, t1, a2
   13420: 8a 06        	slli	a3, a3, 2
   13422: 97 02 00 00  	auipc	t0, 0
   13426: 96 96        	add	a3, a3, t0
   13428: 67 80 a6 00  	jr	10(a3)
   1342c: 23 07 b7 00  	sb	a1, 14(a4)
   13430: a3 06 b7 00  	sb	a1, 13(a4)
   13434: 23 06 b7 00  	sb	a1, 12(a4)
   13438: a3 05 b7 00  	sb	a1, 11(a4)
   1343c: 23 05 b7 00  	sb	a1, 10(a4)
   13440: a3 04 b7 00  	sb	a1, 9(a4)
   13444: 23 04 b7 00  	sb	a1, 8(a4)
   13448: a3 03 b7 00  	sb	a1, 7(a4)
   1344c: 23 03 b7 00  	sb	a1, 6(a4)
   13450: a3 02 b7 00  	sb	a1, 5(a4)
   13454: 23 02 b7 00  	sb	a1, 4(a4)
   13458: a3 01 b7 00  	sb	a1, 3(a4)
   1345c: 23 01 b7 00  	sb	a1, 2(a4)
   13460: a3 00 b7 00  	sb	a1, 1(a4)
   13464: 23 00 b7 00  	sb	a1, 0(a4)
   13468: 82 80        	ret
   1346a: 93 f5 f5 0f  	andi	a1, a1, 255
   1346e: 93 96 85 00  	slli	a3, a1, 8
   13472: d5 8d        	or	a1, a1, a3
   13474: 93 96 05 01  	slli	a3, a1, 16
   13478: d5 8d        	or	a1, a1, a3
   1347a: 93 96 05 02  	slli	a3, a1, 32
   1347e: d5 8d        	or	a1, a1, a3
   13480: 59 b7        	j	0x13406 <memset+0x10>
   13482: 93 96 27 00  	slli	a3, a5, 2
   13486: 97 02 00 00  	auipc	t0, 0
   1348a: 96 96        	add	a3, a3, t0
   1348c: 86 82        	mv	t0, ra
   1348e: e7 80 26 fa  	jalr	-94(a3)
   13492: 96 80        	mv	ra, t0
   13494: c1 17        	addi	a5, a5, -16
   13496: 1d 8f        	sub	a4, a4, a5
   13498: 3e 96        	add	a2, a2, a5
   1349a: e3 71 c3 f8  	bgeu	t1, a2, 0x1341c <memset+0x26>
   1349e: 9d b7        	j	0x13404 <memset+0xe>

00000000000134a0 <_printf_r>:
   134a0: 5d 71        	addi	sp, sp, -80
   134a2: 32 f0        	sd	a2, 32(sp)
   134a4: 36 f4        	sd	a3, 40(sp)
   134a6: 3a f8        	sd	a4, 48(sp)
   134a8: 3e fc        	sd	a5, 56(sp)
   134aa: c2 e0        	sd	a6, 64(sp)
   134ac: c6 e4        	sd	a7, 72(sp)
   134ae: 2e 86        	mv	a2, a1
   134b0: 0c 69        	ld	a1, 16(a0)
   134b2: 14 10        	addi	a3, sp, 32
   134b4: 06 ec        	sd	ra, 24(sp)
   134b6: 36 e4        	sd	a3, 8(sp)
   134b8: ef 00 60 03  	jal	0x134ee <_vfprintf_r>
   134bc: e2 60        	ld	ra, 24(sp)
   134be: 61 61        	addi	sp, sp, 80
   134c0: 82 80        	ret

00000000000134c2 <printf>:
   134c2: 03 b3 81 93  	ld	t1, -1736(gp)
   134c6: 1d 71        	addi	sp, sp, -96
   134c8: 32 f8        	sd	a2, 48(sp)
   134ca: 36 fc        	sd	a3, 56(sp)
   134cc: 2e f4        	sd	a1, 40(sp)
   134ce: ba e0        	sd	a4, 64(sp)
   134d0: be e4        	sd	a5, 72(sp)
   134d2: c2 e8        	sd	a6, 80(sp)
   134d4: c6 ec        	sd	a7, 88(sp)
   134d6: 83 35 03 01  	ld	a1, 16(t1)
   134da: 34 10        	addi	a3, sp, 40
   134dc: 2a 86        	mv	a2, a0
   134de: 1a 85        	mv	a0, t1
   134e0: 06 ec        	sd	ra, 24(sp)
   134e2: 36 e4        	sd	a3, 8(sp)
   134e4: ef 00 a0 00  	jal	0x134ee <_vfprintf_r>
   134e8: e2 60        	ld	ra, 24(sp)
   134ea: 25 61        	addi	sp, sp, 96
   134ec: 82 80        	ret

00000000000134ee <_vfprintf_r>:
   134ee: 13 01 01 d9  	addi	sp, sp, -624
   134f2: 23 34 11 26  	sd	ra, 616(sp)
   134f6: 23 30 81 26  	sd	s0, 608(sp)
   134fa: 23 38 21 25  	sd	s2, 592(sp)
   134fe: 2e e0        	sd	a1, 0(sp)
   13500: 32 89        	mv	s2, a2
   13502: 36 f0        	sd	a3, 32(sp)
   13504: 23 3c 91 24  	sd	s1, 600(sp)
   13508: 23 34 31 25  	sd	s3, 584(sp)
   1350c: 23 30 41 25  	sd	s4, 576(sp)
   13510: 23 3c 51 23  	sd	s5, 568(sp)
   13514: 23 38 61 23  	sd	s6, 560(sp)
   13518: 23 34 71 23  	sd	s7, 552(sp)
   1351c: 23 30 81 23  	sd	s8, 544(sp)
   13520: 23 3c 91 21  	sd	s9, 536(sp)
   13524: 23 38 a1 21  	sd	s10, 528(sp)
   13528: 23 34 b1 21  	sd	s11, 520(sp)
   1352c: 2a 84        	mv	s0, a0
   1352e: aa e8        	sd	a0, 80(sp)
   13530: ef 40 f0 59  	jal	0x182ce <_localeconv_r>
   13534: 1c 61        	ld	a5, 0(a0)
   13536: 3e 85        	mv	a0, a5
   13538: be f0        	sd	a5, 96(sp)
   1353a: ef 60 40 78  	jal	0x19cbe <strlen>
   1353e: aa ec        	sd	a0, 88(sp)
   13540: 82 f9        	sd	zero, 240(sp)
   13542: 82 fd        	sd	zero, 248(sp)
   13544: 01 c4        	beqz	s0, 0x1354c <_vfprintf_r+0x5e>
   13546: 38 48        	lw	a4, 80(s0)
   13548: 63 0f 07 46  	beqz	a4, 0x139c6 <_vfprintf_r+0x4d8>
   1354c: 82 67        	ld	a5, 0(sp)
   1354e: 03 97 07 01  	lh	a4, 16(a5)
   13552: 93 16 27 03  	slli	a3, a4, 50
   13556: 63 c1 06 02  	bltz	a3, 0x13578 <_vfprintf_r+0x8a>
   1355a: 83 a6 c7 0a  	lw	a3, 172(a5)
   1355e: 09 66        	lui	a2, 2
   13560: 51 8f        	or	a4, a4, a2
   13562: 79 76        	lui	a2, 1048574
   13564: 1b 17 07 01  	slliw	a4, a4, 16
   13568: 7d 16        	addi	a2, a2, -1
   1356a: 1b 57 07 41  	sraiw	a4, a4, 16
   1356e: f1 8e        	and	a3, a3, a2
   13570: 23 98 e7 00  	sh	a4, 16(a5)
   13574: 23 a6 d7 0a  	sw	a3, 172(a5)
   13578: 93 76 87 00  	andi	a3, a4, 8
   1357c: 63 87 06 26  	beqz	a3, 0x137ea <_vfprintf_r+0x2fc>
   13580: 82 67        	ld	a5, 0(sp)
   13582: 94 6f        	ld	a3, 24(a5)
   13584: 63 83 06 26  	beqz	a3, 0x137ea <_vfprintf_r+0x2fc>
   13588: 69 8b        	andi	a4, a4, 26
   1358a: a9 46        	addi	a3, zero, 10
   1358c: 63 0d d7 26  	beq	a4, a3, 0x13806 <_vfprintf_r+0x318>
   13590: 1c 03        	addi	a5, sp, 384
   13592: 37 27 02 00  	lui	a4, 34
   13596: 3e 83        	mv	t1, a5
   13598: 3e e2        	sd	a5, 256(sp)
   1359a: 4a 8b        	mv	s6, s2
   1359c: 93 07 c7 c2  	addi	a5, a4, -980
   135a0: 37 27 02 00  	lui	a4, 34
   135a4: 3e f4        	sd	a5, 40(sp)
   135a6: 93 07 07 e3  	addi	a5, a4, -464
   135aa: 03 47 0b 00  	lbu	a4, 0(s6)
   135ae: 02 ea        	sd	zero, 272(sp)
   135b0: 23 24 01 10  	sw	zero, 264(sp)
   135b4: 02 fc        	sd	zero, 56(sp)
   135b6: 82 f4        	sd	zero, 104(sp)
   135b8: 82 f8        	sd	zero, 112(sp)
   135ba: 02 e1        	sd	zero, 128(sp)
   135bc: 02 e5        	sd	zero, 136(sp)
   135be: 82 fc        	sd	zero, 120(sp)
   135c0: 02 e8        	sd	zero, 16(sp)
   135c2: 3e e4        	sd	a5, 8(sp)
   135c4: c6 69        	ld	s3, 80(sp)
   135c6: 1a 84        	mv	s0, t1
   135c8: 63 01 07 1c  	beqz	a4, 0x1378a <_vfprintf_r+0x29c>
   135cc: da 84        	mv	s1, s6
   135ce: 93 06 50 02  	addi	a3, zero, 37
   135d2: 63 07 d7 24  	beq	a4, a3, 0x13820 <_vfprintf_r+0x332>
   135d6: 03 c7 14 00  	lbu	a4, 1(s1)
   135da: 85 04        	addi	s1, s1, 1
   135dc: 7d fb        	bnez	a4, 0x135d2 <_vfprintf_r+0xe4>
   135de: 3b 89 64 41  	subw	s2, s1, s6
   135e2: 63 04 09 1a  	beqz	s2, 0x1378a <_vfprintf_r+0x29c>
   135e6: d2 6b        	ld	s7, 272(sp)
   135e8: 83 26 81 10  	lw	a3, 264(sp)
   135ec: 23 30 64 01  	sd	s6, 0(s0)
   135f0: 33 87 2b 01  	add	a4, s7, s2
   135f4: 1b 86 16 00  	addiw	a2, a3, 1
   135f8: 23 34 24 01  	sd	s2, 8(s0)
   135fc: 3a ea        	sd	a4, 272(sp)
   135fe: 23 24 c1 10  	sw	a2, 264(sp)
   13602: 1d 47        	addi	a4, zero, 7
   13604: 41 04        	addi	s0, s0, 16
   13606: 63 44 c7 22  	blt	a4, a2, 0x1382e <_vfprintf_r+0x340>
   1360a: c2 67        	ld	a5, 16(sp)
   1360c: 03 c7 04 00  	lbu	a4, 0(s1)
   13610: bb 87 27 01  	addw	a5, a5, s2
   13614: 3e e8        	sd	a5, 16(sp)
   13616: 63 0a 07 16  	beqz	a4, 0x1378a <_vfprintf_r+0x29c>
   1361a: 7d 53        	addi	t1, zero, -1
   1361c: 03 ce 14 00  	lbu	t3, 1(s1)
   13620: a3 07 01 0c  	sb	zero, 207(sp)
   13624: 85 04        	addi	s1, s1, 1
   13626: 01 4a        	mv	s4, zero
   13628: 01 49        	mv	s2, zero
   1362a: 93 0b a0 05  	addi	s7, zero, 90
   1362e: 9a 8a        	mv	s5, t1
   13630: 85 04        	addi	s1, s1, 1
   13632: 1b 0d 0e 00  	sext.w	s10, t3
   13636: 9b 07 0d fe  	addiw	a5, s10, -32
   1363a: 1b 87 07 00  	sext.w	a4, a5
   1363e: 63 e5 eb 04  	bltu	s7, a4, 0x13688 <_vfprintf_r+0x19a>
   13642: 13 97 07 02  	slli	a4, a5, 32
   13646: 93 57 e7 01  	srli	a5, a4, 30
   1364a: 22 77        	ld	a4, 40(sp)
   1364c: ba 97        	add	a5, a5, a4
   1364e: 9c 43        	lw	a5, 0(a5)
   13650: 82 87        	jr	a5
   13652: 01 4a        	mv	s4, zero
   13654: a5 46        	addi	a3, zero, 9
   13656: 03 c7 04 00  	lbu	a4, 0(s1)
   1365a: 9b 17 2a 00  	slliw	a5, s4, 2
   1365e: 3b 8a 47 01  	addw	s4, a5, s4
   13662: 1b 0d 0d fd  	addiw	s10, s10, -48
   13666: 1b 1a 1a 00  	slliw	s4, s4, 1
   1366a: 9b 07 07 fd  	addiw	a5, a4, -48
   1366e: 3b 0a 4d 01  	addw	s4, s10, s4
   13672: 85 04        	addi	s1, s1, 1
   13674: 1b 0d 07 00  	sext.w	s10, a4
   13678: e3 ff f6 fc  	bgeu	a3, a5, 0x13656 <_vfprintf_r+0x168>
   1367c: 9b 07 0d fe  	addiw	a5, s10, -32
   13680: 1b 87 07 00  	sext.w	a4, a5
   13684: e3 ff eb fa  	bgeu	s7, a4, 0x13642 <_vfprintf_r+0x154>
   13688: 63 01 0d 10  	beqz	s10, 0x1378a <_vfprintf_r+0x29c>
   1368c: 23 0c a1 11  	sb	s10, 280(sp)
   13690: a3 07 01 0c  	sb	zero, 207(sp)
   13694: 05 4c        	addi	s8, zero, 1
   13696: 85 4d        	addi	s11, zero, 1
   13698: 13 0b 81 11  	addi	s6, sp, 280
   1369c: 02 ec        	sd	zero, 24(sp)
   1369e: 01 43        	mv	t1, zero
   136a0: 82 e4        	sd	zero, 72(sp)
   136a2: 82 e0        	sd	zero, 64(sp)
   136a4: 02 f8        	sd	zero, 48(sp)
   136a6: 93 7c 29 00  	andi	s9, s2, 2
   136aa: 63 83 0c 00  	beqz	s9, 0x136b0 <_vfprintf_r+0x1c2>
   136ae: 09 2c        	addiw	s8, s8, 2
   136b0: 93 7a 49 08  	andi	s5, s2, 132
   136b4: 63 96 0a 00  	bnez	s5, 0x136c0 <_vfprintf_r+0x1d2>
   136b8: 3b 07 8a 41  	subw	a4, s4, s8
   136bc: e3 48 e0 04  	bgtz	a4, 0x13f0c <_vfprintf_r+0xa1e>
   136c0: 03 47 f1 0c  	lbu	a4, 207(sp)
   136c4: d2 6b        	ld	s7, 272(sp)
   136c6: 1d c3        	beqz	a4, 0x136ec <_vfprintf_r+0x1fe>
   136c8: 03 26 81 10  	lw	a2, 264(sp)
   136cc: 93 05 f1 0c  	addi	a1, sp, 207
   136d0: 0c e0        	sd	a1, 0(s0)
   136d2: 05 26        	addiw	a2, a2, 1
   136d4: 85 45        	addi	a1, zero, 1
   136d6: 85 0b        	addi	s7, s7, 1
   136d8: 0c e4        	sd	a1, 8(s0)
   136da: 23 24 c1 10  	sw	a2, 264(sp)
   136de: 9b 05 06 00  	sext.w	a1, a2
   136e2: 5e ea        	sd	s7, 272(sp)
   136e4: 1d 46        	addi	a2, zero, 7
   136e6: 41 04        	addi	s0, s0, 16
   136e8: 63 44 b6 20  	blt	a2, a1, 0x138f0 <_vfprintf_r+0x402>
   136ec: 63 83 0c 02  	beqz	s9, 0x13712 <_vfprintf_r+0x224>
   136f0: 03 26 81 10  	lw	a2, 264(sp)
   136f4: 89 45        	addi	a1, zero, 2
   136f6: 89 0b        	addi	s7, s7, 2
   136f8: 05 26        	addiw	a2, a2, 1
   136fa: 9c 09        	addi	a5, sp, 208
   136fc: 0c e4        	sd	a1, 8(s0)
   136fe: 1c e0        	sd	a5, 0(s0)
   13700: 9b 05 06 00  	sext.w	a1, a2
   13704: 23 24 c1 10  	sw	a2, 264(sp)
   13708: 5e ea        	sd	s7, 272(sp)
   1370a: 1d 46        	addi	a2, zero, 7
   1370c: 41 04        	addi	s0, s0, 16
   1370e: e3 4c b6 0a  	blt	a2, a1, 0x13fc6 <_vfprintf_r+0xad8>
   13712: 13 06 00 08  	addi	a2, zero, 128
   13716: 63 84 ca 6a  	beq	s5, a2, 0x13dbe <_vfprintf_r+0x8d0>
   1371a: bb 0c b3 41  	subw	s9, t1, s11
   1371e: 63 4f 90 73  	bgtz	s9, 0x13e5c <_vfprintf_r+0x96e>
   13722: 13 76 09 10  	andi	a2, s2, 256
   13726: 63 17 06 54  	bnez	a2, 0x13c74 <_vfprintf_r+0x786>
   1372a: 03 26 81 10  	lw	a2, 264(sp)
   1372e: 33 87 7d 01  	add	a4, s11, s7
   13732: 23 30 64 01  	sd	s6, 0(s0)
   13736: 9b 06 16 00  	addiw	a3, a2, 1
   1373a: 23 34 b4 01  	sd	s11, 8(s0)
   1373e: 3a ea        	sd	a4, 272(sp)
   13740: 23 24 d1 10  	sw	a3, 264(sp)
   13744: 9d 47        	addi	a5, zero, 7
   13746: 63 c6 d7 26  	blt	a5, a3, 0x139b2 <_vfprintf_r+0x4c4>
   1374a: 41 04        	addi	s0, s0, 16
   1374c: 13 79 49 00  	andi	s2, s2, 4
   13750: 63 06 09 00  	beqz	s2, 0x1375c <_vfprintf_r+0x26e>
   13754: 3b 09 8a 41  	subw	s2, s4, s8
   13758: e3 43 20 09  	bgtz	s2, 0x13fde <_vfprintf_r+0xaf0>
   1375c: d2 87        	mv	a5, s4
   1375e: 63 53 8a 01  	bge	s4, s8, 0x13764 <_vfprintf_r+0x276>
   13762: e2 87        	mv	a5, s8
   13764: c2 66        	ld	a3, 16(sp)
   13766: b5 9f        	addw	a5, a5, a3
   13768: 3e e8        	sd	a5, 16(sp)
   1376a: 63 13 07 78  	bnez	a4, 0x13ef0 <_vfprintf_r+0xa02>
   1376e: e2 67        	ld	a5, 24(sp)
   13770: 23 24 01 10  	sw	zero, 264(sp)
   13774: 89 c7        	beqz	a5, 0x1377e <_vfprintf_r+0x290>
   13776: e2 65        	ld	a1, 24(sp)
   13778: 4e 85        	mv	a0, s3
   1377a: ef 20 c0 0e  	jal	0x15866 <_free_r>
   1377e: 00 03        	addi	s0, sp, 384
   13780: 26 8b        	mv	s6, s1
   13782: 03 47 0b 00  	lbu	a4, 0(s6)
   13786: e3 13 07 e4  	bnez	a4, 0x135cc <_vfprintf_r+0xde>
   1378a: d2 67        	ld	a5, 272(sp)
   1378c: 99 c3        	beqz	a5, 0x13792 <_vfprintf_r+0x2a4>
   1378e: 6f 10 80 2d  	j	0x14a66 <_vfprintf_r+0x1578>
   13792: 82 67        	ld	a5, 0(sp)
   13794: 83 d7 07 01  	lhu	a5, 16(a5)
   13798: 93 f7 07 04  	andi	a5, a5, 64
   1379c: 99 c3        	beqz	a5, 0x137a2 <_vfprintf_r+0x2b4>
   1379e: 6f 10 00 72  	j	0x14ebe <_vfprintf_r+0x19d0>
   137a2: 83 30 81 26  	ld	ra, 616(sp)
   137a6: 03 34 01 26  	ld	s0, 608(sp)
   137aa: 42 65        	ld	a0, 16(sp)
   137ac: 83 34 81 25  	ld	s1, 600(sp)
   137b0: 03 39 01 25  	ld	s2, 592(sp)
   137b4: 83 39 81 24  	ld	s3, 584(sp)
   137b8: 03 3a 01 24  	ld	s4, 576(sp)
   137bc: 83 3a 81 23  	ld	s5, 568(sp)
   137c0: 03 3b 01 23  	ld	s6, 560(sp)
   137c4: 83 3b 81 22  	ld	s7, 552(sp)
   137c8: 03 3c 01 22  	ld	s8, 544(sp)
   137cc: 83 3c 81 21  	ld	s9, 536(sp)
   137d0: 03 3d 01 21  	ld	s10, 528(sp)
   137d4: 83 3d 81 20  	ld	s11, 520(sp)
   137d8: 13 01 01 27  	addi	sp, sp, 624
   137dc: 82 80        	ret
   137de: 13 69 09 01  	ori	s2, s2, 16
   137e2: 03 ce 04 00  	lbu	t3, 0(s1)
   137e6: 01 29        	sext.w	s2, s2
   137e8: a1 b5        	j	0x13630 <_vfprintf_r+0x142>
   137ea: 82 65        	ld	a1, 0(sp)
   137ec: 46 65        	ld	a0, 80(sp)
   137ee: ef 10 30 15  	jal	0x15140 <__swsetup_r>
   137f2: 19 c1        	beqz	a0, 0x137f8 <_vfprintf_r+0x30a>
   137f4: 6f 10 a0 6c  	j	0x14ebe <_vfprintf_r+0x19d0>
   137f8: 82 67        	ld	a5, 0(sp)
   137fa: a9 46        	addi	a3, zero, 10
   137fc: 03 97 07 01  	lh	a4, 16(a5)
   13800: 69 8b        	andi	a4, a4, 26
   13802: e3 17 d7 d8  	bne	a4, a3, 0x13590 <_vfprintf_r+0xa2>
   13806: 82 67        	ld	a5, 0(sp)
   13808: 03 97 27 01  	lh	a4, 18(a5)
   1380c: e3 42 07 d8  	bltz	a4, 0x13590 <_vfprintf_r+0xa2>
   13810: 82 76        	ld	a3, 32(sp)
   13812: 46 65        	ld	a0, 80(sp)
   13814: 4a 86        	mv	a2, s2
   13816: be 85        	mv	a1, a5
   13818: ef 10 50 09  	jal	0x150ac <__sbprintf>
   1381c: 2a e8        	sd	a0, 16(sp)
   1381e: 51 b7        	j	0x137a2 <_vfprintf_r+0x2b4>
   13820: 3b 89 64 41  	subw	s2, s1, s6
   13824: e3 11 09 dc  	bnez	s2, 0x135e6 <_vfprintf_r+0xf8>
   13828: 03 c7 04 00  	lbu	a4, 0(s1)
   1382c: ed b3        	j	0x13616 <_vfprintf_r+0x128>
   1382e: 82 65        	ld	a1, 0(sp)
   13830: 10 02        	addi	a2, sp, 256
   13832: 4e 85        	mv	a0, s3
   13834: ef 80 60 1c  	jal	0x1b9fa <__sprint_r>
   13838: 29 fd        	bnez	a0, 0x13792 <_vfprintf_r+0x2a4>
   1383a: 00 03        	addi	s0, sp, 384
   1383c: f9 b3        	j	0x1360a <_vfprintf_r+0x11c>
   1383e: 13 77 89 00  	andi	a4, s2, 8
   13842: 56 83        	mv	t1, s5
   13844: e3 1b 07 2c  	bnez	a4, 0x1431a <_vfprintf_r+0xe2c>
   13848: 82 77        	ld	a5, 32(sp)
   1384a: 56 ec        	sd	s5, 24(sp)
   1384c: 88 23        	fld	fa0, 0(a5)
   1384e: 93 8a 87 00  	addi	s5, a5, 8
   13852: 56 f0        	sd	s5, 32(sp)
   13854: ef c0 60 52  	jal	0x1fd7a <__extenddftf2>
   13858: 62 63        	ld	t1, 24(sp)
   1385a: 2a 87        	mv	a4, a0
   1385c: 88 19        	addi	a0, sp, 240
   1385e: 1a ec        	sd	t1, 24(sp)
   13860: ba f9        	sd	a4, 240(sp)
   13862: ae fd        	sd	a1, 248(sp)
   13864: ef 40 30 22  	jal	0x18286 <_ldcheck>
   13868: aa cd        	sw	a0, 216(sp)
   1386a: 09 47        	addi	a4, zero, 2
   1386c: 62 63        	ld	t1, 24(sp)
   1386e: e3 0c e5 52  	beq	a0, a4, 0x145a6 <_vfprintf_r+0x10b8>
   13872: 05 47        	addi	a4, zero, 1
   13874: e3 04 e5 62  	beq	a0, a4, 0x1469c <_vfprintf_r+0x11ae>
   13878: 13 07 10 06  	addi	a4, zero, 97
   1387c: 63 14 ed 00  	bne	s10, a4, 0x13884 <_vfprintf_r+0x396>
   13880: 6f 10 c0 50  	j	0x14d8c <_vfprintf_r+0x189e>
   13884: 13 07 10 04  	addi	a4, zero, 65
   13888: 63 14 ed 00  	bne	s10, a4, 0x13890 <_vfprintf_r+0x3a2>
   1388c: 6f 10 00 21  	j	0x14a9c <_vfprintf_r+0x15ae>
   13890: 7d 57        	addi	a4, zero, -1
   13892: 63 14 e3 00  	bne	t1, a4, 0x1389a <_vfprintf_r+0x3ac>
   13896: 6f 10 a0 70  	j	0x14fa0 <_vfprintf_r+0x1ab2>
   1389a: 13 77 fd fd  	andi	a4, s10, -33
   1389e: 93 06 70 04  	addi	a3, zero, 71
   138a2: 63 14 d7 00  	bne	a4, a3, 0x138aa <_vfprintf_r+0x3bc>
   138a6: 6f 10 c0 6e  	j	0x14f92 <_vfprintf_r+0x1aa4>
   138aa: 13 67 09 10  	ori	a4, s2, 256
   138ae: ee 7c        	ld	s9, 248(sp)
   138b0: 9b 07 07 00  	sext.w	a5, a4
   138b4: 3e e9        	sd	a5, 144(sp)
   138b6: ca e0        	sd	s2, 64(sp)
   138b8: 4e 7c        	ld	s8, 240(sp)
   138ba: 63 d4 0c 00  	bgez	s9, 0x138c2 <_vfprintf_r+0x3d4>
   138be: 6f 10 00 4a  	j	0x14d5e <_vfprintf_r+0x1870>
   138c2: 02 ed        	sd	zero, 152(sp)
   138c4: 02 ec        	sd	zero, 24(sp)
   138c6: 1b 07 fd fb  	addiw	a4, s10, -65
   138ca: 1b 06 07 00  	sext.w	a2, a4
   138ce: 93 06 50 02  	addi	a3, zero, 37
   138d2: 63 f4 c6 00  	bgeu	a3, a2, 0x138da <_vfprintf_r+0x3ec>
   138d6: 6f 00 f0 7b  	j	0x14894 <_vfprintf_r+0x13a6>
   138da: 93 16 07 02  	slli	a3, a4, 32
   138de: 13 d7 e6 01  	srli	a4, a3, 30
   138e2: b7 26 02 00  	lui	a3, 34
   138e6: 93 86 86 d9  	addi	a3, a3, -616
   138ea: 36 97        	add	a4, a4, a3
   138ec: 18 43        	lw	a4, 0(a4)
   138ee: 02 87        	jr	a4
   138f0: 82 65        	ld	a1, 0(sp)
   138f2: 10 02        	addi	a2, sp, 256
   138f4: 4e 85        	mv	a0, s3
   138f6: 1a e9        	sd	t1, 144(sp)
   138f8: ef 80 20 10  	jal	0x1b9fa <__sprint_r>
   138fc: 63 11 05 60  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13900: d2 6b        	ld	s7, 272(sp)
   13902: 4a 63        	ld	t1, 144(sp)
   13904: 00 03        	addi	s0, sp, 384
   13906: dd b3        	j	0x136ec <_vfprintf_r+0x1fe>
   13908: e2 77        	ld	a5, 56(sp)
   1390a: 85 46        	addi	a3, zero, 1
   1390c: 93 0c 04 01  	addi	s9, s0, 16
   13910: e3 de f6 06  	bge	a3, a5, 0x1418c <_vfprintf_r+0xc9e>
   13914: 03 26 81 10  	lw	a2, 264(sp)
   13918: 85 46        	addi	a3, zero, 1
   1391a: 13 87 1b 00  	addi	a4, s7, 1
   1391e: 05 26        	addiw	a2, a2, 1
   13920: 14 e4        	sd	a3, 8(s0)
   13922: 23 24 c1 10  	sw	a2, 264(sp)
   13926: 9b 06 06 00  	sext.w	a3, a2
   1392a: 23 30 64 01  	sd	s6, 0(s0)
   1392e: 3a ea        	sd	a4, 272(sp)
   13930: 1d 46        	addi	a2, zero, 7
   13932: e3 4f d6 32  	blt	a2, a3, 0x14470 <_vfprintf_r+0xf82>
   13936: e6 67        	ld	a5, 88(sp)
   13938: 1b 84 16 00  	addiw	s0, a3, 1
   1393c: 23 24 81 10  	sw	s0, 264(sp)
   13940: 33 8d e7 00  	add	s10, a5, a4
   13944: 06 77        	ld	a4, 96(sp)
   13946: 23 b4 fc 00  	sd	a5, 8(s9)
   1394a: 6a ea        	sd	s10, 272(sp)
   1394c: 23 b0 ec 00  	sd	a4, 0(s9)
   13950: 1d 47        	addi	a4, zero, 7
   13952: c1 0c        	addi	s9, s9, 16
   13954: e3 4c 87 32  	blt	a4, s0, 0x1448c <_vfprintf_r+0xf9e>
   13958: 4e 75        	ld	a0, 240(sp)
   1395a: ee 75        	ld	a1, 248(sp)
   1395c: 01 46        	mv	a2, zero
   1395e: 81 46        	mv	a3, zero
   13960: ef b0 00 1c  	jal	0x1eb20 <__netf2>
   13964: e2 77        	ld	a5, 56(sp)
   13966: 9b 8d f7 ff  	addiw	s11, a5, -1
   1396a: e3 04 05 06  	beqz	a0, 0x141d2 <_vfprintf_r+0xce4>
   1396e: 93 07 1b 00  	addi	a5, s6, 1
   13972: 05 24        	addiw	s0, s0, 1
   13974: 6e 9d        	add	s10, s10, s11
   13976: 23 b0 fc 00  	sd	a5, 0(s9)
   1397a: 23 b4 bc 01  	sd	s11, 8(s9)
   1397e: 6a ea        	sd	s10, 272(sp)
   13980: 23 24 81 10  	sw	s0, 264(sp)
   13984: 9d 47        	addi	a5, zero, 7
   13986: e3 c8 87 02  	blt	a5, s0, 0x141b6 <_vfprintf_r+0xcc8>
   1398a: c1 0c        	addi	s9, s9, 16
   1398c: c6 76        	ld	a3, 112(sp)
   1398e: 05 24        	addiw	s0, s0, 1
   13990: 9c 11        	addi	a5, sp, 224
   13992: 33 87 a6 01  	add	a4, a3, s10
   13996: 23 b0 fc 00  	sd	a5, 0(s9)
   1399a: 23 b4 dc 00  	sd	a3, 8(s9)
   1399e: 23 24 81 10  	sw	s0, 264(sp)
   139a2: 9b 06 04 00  	sext.w	a3, s0
   139a6: 3a ea        	sd	a4, 272(sp)
   139a8: 9d 47        	addi	a5, zero, 7
   139aa: 13 84 0c 01  	addi	s0, s9, 16
   139ae: e3 df d7 d8  	bge	a5, a3, 0x1374c <_vfprintf_r+0x25e>
   139b2: 82 65        	ld	a1, 0(sp)
   139b4: 10 02        	addi	a2, sp, 256
   139b6: 4e 85        	mv	a0, s3
   139b8: ef 80 20 04  	jal	0x1b9fa <__sprint_r>
   139bc: 63 11 05 54  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   139c0: 52 67        	ld	a4, 272(sp)
   139c2: 00 03        	addi	s0, sp, 384
   139c4: 61 b3        	j	0x1374c <_vfprintf_r+0x25e>
   139c6: 46 65        	ld	a0, 80(sp)
   139c8: ef 10 b0 56  	jal	0x15732 <__sinit>
   139cc: 41 b6        	j	0x1354c <_vfprintf_r+0x5e>
   139ce: 02 77        	ld	a4, 32(sp)
   139d0: a3 07 01 0c  	sb	zero, 207(sp)
   139d4: 05 4c        	addi	s8, zero, 1
   139d6: 1c 43        	lw	a5, 0(a4)
   139d8: 21 07        	addi	a4, a4, 8
   139da: 3a f0        	sd	a4, 32(sp)
   139dc: 23 0c f1 10  	sb	a5, 280(sp)
   139e0: 85 4d        	addi	s11, zero, 1
   139e2: 13 0b 81 11  	addi	s6, sp, 280
   139e6: 5d b9        	j	0x1369c <_vfprintf_r+0x1ae>
   139e8: 82 77        	ld	a5, 32(sp)
   139ea: a3 07 01 0c  	sb	zero, 207(sp)
   139ee: 56 83        	mv	t1, s5
   139f0: 03 bb 07 00  	ld	s6, 0(a5)
   139f4: 93 8a 87 00  	addi	s5, a5, 8
   139f8: e3 0d 0b 1e  	beqz	s6, 0x143f2 <_vfprintf_r+0xf04>
   139fc: 7d 57        	addi	a4, zero, -1
   139fe: e3 04 e3 44  	beq	t1, a4, 0x14646 <_vfprintf_r+0x1158>
   13a02: 1a 86        	mv	a2, t1
   13a04: 81 45        	mv	a1, zero
   13a06: 5a 85        	mv	a0, s6
   13a08: 1a f0        	sd	t1, 32(sp)
   13a0a: ef 50 20 16  	jal	0x18b6c <memchr>
   13a0e: 2a ec        	sd	a0, 24(sp)
   13a10: 02 73        	ld	t1, 32(sp)
   13a12: 19 e1        	bnez	a0, 0x13a18 <_vfprintf_r+0x52a>
   13a14: 6f 10 60 3d  	j	0x14dea <_vfprintf_r+0x18fc>
   13a18: e2 67        	ld	a5, 24(sp)
   13a1a: bb 8d 67 41  	subw	s11, a5, s6
   13a1e: 03 47 f1 0c  	lbu	a4, 207(sp)
   13a22: 13 c8 fd ff  	not	a6, s11
   13a26: 13 58 f8 43  	srai	a6, a6, 63
   13a2a: 56 f0        	sd	s5, 32(sp)
   13a2c: 02 ec        	sd	zero, 24(sp)
   13a2e: 82 e4        	sd	zero, 72(sp)
   13a30: 82 e0        	sd	zero, 64(sp)
   13a32: 02 f8        	sd	zero, 48(sp)
   13a34: 33 7c b8 01  	and	s8, a6, s11
   13a38: 01 43        	mv	t1, zero
   13a3a: e3 06 07 c6  	beqz	a4, 0x136a6 <_vfprintf_r+0x1b8>
   13a3e: 05 2c        	addiw	s8, s8, 1
   13a40: 9d b1        	j	0x136a6 <_vfprintf_r+0x1b8>
   13a42: 82 77        	ld	a5, 32(sp)
   13a44: 03 aa 07 00  	lw	s4, 0(a5)
   13a48: a1 07        	addi	a5, a5, 8
   13a4a: e3 5e 0a 06  	bgez	s4, 0x142c6 <_vfprintf_r+0xdd8>
   13a4e: 3b 0a 40 41  	negw	s4, s4
   13a52: 3e f0        	sd	a5, 32(sp)
   13a54: 13 69 49 00  	ori	s2, s2, 4
   13a58: 03 ce 04 00  	lbu	t3, 0(s1)
   13a5c: 01 29        	sext.w	s2, s2
   13a5e: c9 be        	j	0x13630 <_vfprintf_r+0x142>
   13a60: 93 07 b0 02  	addi	a5, zero, 43
   13a64: 03 ce 04 00  	lbu	t3, 0(s1)
   13a68: a3 07 f1 0c  	sb	a5, 207(sp)
   13a6c: d1 b6        	j	0x13630 <_vfprintf_r+0x142>
   13a6e: 13 69 09 08  	ori	s2, s2, 128
   13a72: 03 ce 04 00  	lbu	t3, 0(s1)
   13a76: 01 29        	sext.w	s2, s2
   13a78: 65 be        	j	0x13630 <_vfprintf_r+0x142>
   13a7a: 03 cd 04 00  	lbu	s10, 0(s1)
   13a7e: 93 06 a0 02  	addi	a3, zero, 42
   13a82: 93 87 14 00  	addi	a5, s1, 1
   13a86: 63 14 dd 00  	bne	s10, a3, 0x13a8e <_vfprintf_r+0x5a0>
   13a8a: 6f 10 c0 5d  	j	0x15066 <_vfprintf_r+0x1b78>
   13a8e: 1b 07 0d fd  	addiw	a4, s10, -48
   13a92: 25 46        	addi	a2, zero, 9
   13a94: be 84        	mv	s1, a5
   13a96: 81 4a        	mv	s5, zero
   13a98: a5 46        	addi	a3, zero, 9
   13a9a: e3 6e e6 b8  	bltu	a2, a4, 0x13636 <_vfprintf_r+0x148>
   13a9e: 03 cd 04 00  	lbu	s10, 0(s1)
   13aa2: 9b 97 2a 00  	slliw	a5, s5, 2
   13aa6: 3b 83 57 01  	addw	t1, a5, s5
   13aaa: 1b 13 13 00  	slliw	t1, t1, 1
   13aae: bb 0a e3 00  	addw	s5, t1, a4
   13ab2: 1b 07 0d fd  	addiw	a4, s10, -48
   13ab6: 85 04        	addi	s1, s1, 1
   13ab8: e3 f3 e6 fe  	bgeu	a3, a4, 0x13a9e <_vfprintf_r+0x5b0>
   13abc: ad be        	j	0x13636 <_vfprintf_r+0x148>
   13abe: 4e 85        	mv	a0, s3
   13ac0: ef 40 f0 00  	jal	0x182ce <_localeconv_r>
   13ac4: 1c 65        	ld	a5, 8(a0)
   13ac6: 3e 85        	mv	a0, a5
   13ac8: be fc        	sd	a5, 120(sp)
   13aca: ef 60 40 1f  	jal	0x19cbe <strlen>
   13ace: aa 87        	mv	a5, a0
   13ad0: 4e 85        	mv	a0, s3
   13ad2: 3e 8c        	mv	s8, a5
   13ad4: 3e e5        	sd	a5, 136(sp)
   13ad6: ef 40 80 7f  	jal	0x182ce <_localeconv_r>
   13ada: 1c 69        	ld	a5, 16(a0)
   13adc: 03 ce 04 00  	lbu	t3, 0(s1)
   13ae0: 3e e1        	sd	a5, 128(sp)
   13ae2: e3 07 0c b4  	beqz	s8, 0x13630 <_vfprintf_r+0x142>
   13ae6: e3 85 07 b4  	beqz	a5, 0x13630 <_vfprintf_r+0x142>
   13aea: 83 c7 07 00  	lbu	a5, 0(a5)
   13aee: e3 81 07 b4  	beqz	a5, 0x13630 <_vfprintf_r+0x142>
   13af2: 13 69 09 40  	ori	s2, s2, 1024
   13af6: 01 29        	sext.w	s2, s2
   13af8: 25 be        	j	0x13630 <_vfprintf_r+0x142>
   13afa: 13 69 19 00  	ori	s2, s2, 1
   13afe: 03 ce 04 00  	lbu	t3, 0(s1)
   13b02: 01 29        	sext.w	s2, s2
   13b04: 35 b6        	j	0x13630 <_vfprintf_r+0x142>
   13b06: 83 47 f1 0c  	lbu	a5, 207(sp)
   13b0a: 03 ce 04 00  	lbu	t3, 0(s1)
   13b0e: e3 91 07 b2  	bnez	a5, 0x13630 <_vfprintf_r+0x142>
   13b12: 93 07 00 02  	addi	a5, zero, 32
   13b16: a3 07 f1 0c  	sb	a5, 207(sp)
   13b1a: 19 be        	j	0x13630 <_vfprintf_r+0x142>
   13b1c: 93 6c 09 01  	ori	s9, s2, 16
   13b20: 56 83        	mv	t1, s5
   13b22: 81 2c        	sext.w	s9, s9
   13b24: 82 76        	ld	a3, 32(sp)
   13b26: 93 f7 0c 02  	andi	a5, s9, 32
   13b2a: 13 87 86 00  	addi	a4, a3, 8
   13b2e: 63 80 07 76  	beqz	a5, 0x1428e <_vfprintf_r+0xda0>
   13b32: 83 bb 06 00  	ld	s7, 0(a3)
   13b36: 85 47        	addi	a5, zero, 1
   13b38: 3a f0        	sd	a4, 32(sp)
   13b3a: 01 47        	mv	a4, zero
   13b3c: a3 07 e1 0c  	sb	a4, 207(sp)
   13b40: 7d 57        	addi	a4, zero, -1
   13b42: 63 0d e3 1e  	beq	t1, a4, 0x13d3c <_vfprintf_r+0x84e>
   13b46: 13 f9 fc f7  	andi	s2, s9, -129
   13b4a: 01 29        	sext.w	s2, s2
   13b4c: 63 97 0b 1e  	bnez	s7, 0x13d3a <_vfprintf_r+0x84c>
   13b50: 63 1a 03 22  	bnez	t1, 0x13d84 <_vfprintf_r+0x896>
   13b54: 63 9d 07 74  	bnez	a5, 0x142ae <_vfprintf_r+0xdc0>
   13b58: 93 fd 1c 00  	andi	s11, s9, 1
   13b5c: 13 0b c1 17  	addi	s6, sp, 380
   13b60: 63 98 0d 76  	bnez	s11, 0x142d0 <_vfprintf_r+0xde2>
   13b64: 03 47 f1 0c  	lbu	a4, 207(sp)
   13b68: 1a 88        	mv	a6, t1
   13b6a: 63 53 b3 01  	bge	t1, s11, 0x13b70 <_vfprintf_r+0x682>
   13b6e: 6e 88        	mv	a6, s11
   13b70: 02 ec        	sd	zero, 24(sp)
   13b72: 82 e4        	sd	zero, 72(sp)
   13b74: 82 e0        	sd	zero, 64(sp)
   13b76: 02 f8        	sd	zero, 48(sp)
   13b78: 1b 0c 08 00  	sext.w	s8, a6
   13b7c: e3 11 07 ec  	bnez	a4, 0x13a3e <_vfprintf_r+0x550>
   13b80: 1d b6        	j	0x136a6 <_vfprintf_r+0x1b8>
   13b82: 13 69 89 00  	ori	s2, s2, 8
   13b86: 03 ce 04 00  	lbu	t3, 0(s1)
   13b8a: 01 29        	sext.w	s2, s2
   13b8c: 55 b4        	j	0x13630 <_vfprintf_r+0x142>
   13b8e: 13 69 09 01  	ori	s2, s2, 16
   13b92: 56 83        	mv	t1, s5
   13b94: 01 29        	sext.w	s2, s2
   13b96: 82 76        	ld	a3, 32(sp)
   13b98: 93 77 09 02  	andi	a5, s2, 32
   13b9c: 13 87 86 00  	addi	a4, a3, 8
   13ba0: 63 80 07 6a  	beqz	a5, 0x14240 <_vfprintf_r+0xd52>
   13ba4: 9c 62        	ld	a5, 0(a3)
   13ba6: be 8b        	mv	s7, a5
   13ba8: 63 cd 07 6a  	bltz	a5, 0x14262 <_vfprintf_r+0xd74>
   13bac: fd 57        	addi	a5, zero, -1
   13bae: 63 0e f3 3e  	beq	t1, a5, 0x13faa <_vfprintf_r+0xabc>
   13bb2: 93 7c f9 f7  	andi	s9, s2, -129
   13bb6: 81 2c        	sext.w	s9, s9
   13bb8: e3 95 0b 1e  	bnez	s7, 0x145a2 <_vfprintf_r+0x10b4>
   13bbc: 63 04 03 00  	beqz	t1, 0x13bc4 <_vfprintf_r+0x6d6>
   13bc0: 6f 10 60 3a  	j	0x14f66 <_vfprintf_r+0x1a78>
   13bc4: 81 4d        	mv	s11, zero
   13bc6: 3a f0        	sd	a4, 32(sp)
   13bc8: 66 89        	mv	s2, s9
   13bca: 13 0b c1 17  	addi	s6, sp, 380
   13bce: 59 bf        	j	0x13b64 <_vfprintf_r+0x676>
   13bd0: 13 69 09 01  	ori	s2, s2, 16
   13bd4: 56 83        	mv	t1, s5
   13bd6: 01 29        	sext.w	s2, s2
   13bd8: 82 76        	ld	a3, 32(sp)
   13bda: 93 77 09 02  	andi	a5, s2, 32
   13bde: 13 87 86 00  	addi	a4, a3, 8
   13be2: 63 88 07 68  	beqz	a5, 0x14272 <_vfprintf_r+0xd84>
   13be6: 83 bb 06 00  	ld	s7, 0(a3)
   13bea: 93 7c f9 bf  	andi	s9, s2, -1025
   13bee: 81 2c        	sext.w	s9, s9
   13bf0: 3a f0        	sd	a4, 32(sp)
   13bf2: 81 47        	mv	a5, zero
   13bf4: 99 b7        	j	0x13b3a <_vfprintf_r+0x64c>
   13bf6: 13 69 09 02  	ori	s2, s2, 32
   13bfa: 03 ce 04 00  	lbu	t3, 0(s1)
   13bfe: 01 29        	sext.w	s2, s2
   13c00: 05 bc        	j	0x13630 <_vfprintf_r+0x142>
   13c02: 02 77        	ld	a4, 32(sp)
   13c04: 93 77 09 02  	andi	a5, s2, 32
   13c08: 18 63        	ld	a4, 0(a4)
   13c0a: 63 88 07 6e  	beqz	a5, 0x142fa <_vfprintf_r+0xe0c>
   13c0e: c2 67        	ld	a5, 16(sp)
   13c10: 1c e3        	sd	a5, 0(a4)
   13c12: 82 77        	ld	a5, 32(sp)
   13c14: 26 8b        	mv	s6, s1
   13c16: a1 07        	addi	a5, a5, 8
   13c18: 3e f0        	sd	a5, 32(sp)
   13c1a: a5 b6        	j	0x13782 <_vfprintf_r+0x294>
   13c1c: 03 ce 04 00  	lbu	t3, 0(s1)
   13c20: 93 07 c0 06  	addi	a5, zero, 108
   13c24: 63 0f fe 7a  	beq	t3, a5, 0x143e2 <_vfprintf_r+0xef4>
   13c28: 13 69 09 01  	ori	s2, s2, 16
   13c2c: 01 29        	sext.w	s2, s2
   13c2e: 09 b4        	j	0x13630 <_vfprintf_r+0x142>
   13c30: 03 ce 04 00  	lbu	t3, 0(s1)
   13c34: 93 07 80 06  	addi	a5, zero, 104
   13c38: 63 0a fe 78  	beq	t3, a5, 0x143cc <_vfprintf_r+0xede>
   13c3c: 13 69 09 04  	ori	s2, s2, 64
   13c40: 01 29        	sext.w	s2, s2
   13c42: fd b2        	j	0x13630 <_vfprintf_r+0x142>
   13c44: 02 77        	ld	a4, 32(sp)
   13c46: e1 77        	lui	a5, 1048568
   13c48: 93 c7 07 83  	xori	a5, a5, -2000
   13c4c: 23 18 f1 0c  	sh	a5, 208(sp)
   13c50: 93 07 87 00  	addi	a5, a4, 8
   13c54: 3e f0        	sd	a5, 32(sp)
   13c56: b7 27 02 00  	lui	a5, 34
   13c5a: 93 87 87 c0  	addi	a5, a5, -1016
   13c5e: 93 6c 29 00  	ori	s9, s2, 2
   13c62: be f4        	sd	a5, 104(sp)
   13c64: 83 3b 07 00  	ld	s7, 0(a4)
   13c68: 56 83        	mv	t1, s5
   13c6a: 81 2c        	sext.w	s9, s9
   13c6c: 89 47        	addi	a5, zero, 2
   13c6e: 13 0d 80 07  	addi	s10, zero, 120
   13c72: e1 b5        	j	0x13b3a <_vfprintf_r+0x64c>
   13c74: 93 06 50 06  	addi	a3, zero, 101
   13c78: e3 d8 a6 c9  	bge	a3, s10, 0x13908 <_vfprintf_r+0x41a>
   13c7c: 4e 75        	ld	a0, 240(sp)
   13c7e: ee 75        	ld	a1, 248(sp)
   13c80: 01 46        	mv	a2, zero
   13c82: 81 46        	mv	a3, zero
   13c84: ef a0 d0 69  	jal	0x1eb20 <__netf2>
   13c88: 63 1c 05 3c  	bnez	a0, 0x14060 <_vfprintf_r+0xb72>
   13c8c: 83 27 81 10  	lw	a5, 264(sp)
   13c90: b7 26 02 00  	lui	a3, 34
   13c94: 93 86 86 c2  	addi	a3, a3, -984
   13c98: 85 27        	addiw	a5, a5, 1
   13c9a: 14 e0        	sd	a3, 0(s0)
   13c9c: 85 0b        	addi	s7, s7, 1
   13c9e: 85 46        	addi	a3, zero, 1
   13ca0: 14 e4        	sd	a3, 8(s0)
   13ca2: 23 24 f1 10  	sw	a5, 264(sp)
   13ca6: 9b 86 07 00  	sext.w	a3, a5
   13caa: 5e ea        	sd	s7, 272(sp)
   13cac: 9d 47        	addi	a5, zero, 7
   13cae: 41 04        	addi	s0, s0, 16
   13cb0: e3 cf d7 0a  	blt	a5, a3, 0x1456e <_vfprintf_r+0x1080>
   13cb4: ee 47        	lw	a5, 216(sp)
   13cb6: 62 77        	ld	a4, 56(sp)
   13cb8: 63 df e7 5e  	bge	a5, a4, 0x142b6 <_vfprintf_r+0xdc8>
   13cbc: e6 66        	ld	a3, 88(sp)
   13cbe: 83 27 81 10  	lw	a5, 264(sp)
   13cc2: 06 76        	ld	a2, 96(sp)
   13cc4: 33 87 76 01  	add	a4, a3, s7
   13cc8: 85 27        	addiw	a5, a5, 1
   13cca: 14 e4        	sd	a3, 8(s0)
   13ccc: 10 e0        	sd	a2, 0(s0)
   13cce: 9b 86 07 00  	sext.w	a3, a5
   13cd2: 23 24 f1 10  	sw	a5, 264(sp)
   13cd6: 3a ea        	sd	a4, 272(sp)
   13cd8: 9d 47        	addi	a5, zero, 7
   13cda: 41 04        	addi	s0, s0, 16
   13cdc: 63 ce d7 6c  	blt	a5, a3, 0x143b8 <_vfprintf_r+0xeca>
   13ce0: e2 77        	ld	a5, 56(sp)
   13ce2: 9b 8c f7 ff  	addiw	s9, a5, -1
   13ce6: e3 53 90 a7  	blez	s9, 0x1374c <_vfprintf_r+0x25e>
   13cea: 37 2f 02 00  	lui	t5, 34
   13cee: c1 46        	addi	a3, zero, 16
   13cf0: 83 27 81 10  	lw	a5, 264(sp)
   13cf4: 93 0b 0f e3  	addi	s7, t5, -464
   13cf8: e3 d6 96 09  	bge	a3, s9, 0x14584 <_vfprintf_r+0x1096>
   13cfc: 82 6a        	ld	s5, 0(sp)
   13cfe: 41 4d        	addi	s10, zero, 16
   13d00: 9d 4d        	addi	s11, zero, 7
   13d02: 21 a0        	j	0x13d0a <_vfprintf_r+0x81c>
   13d04: c1 3c        	addiw	s9, s9, -16
   13d06: e3 5f 9d 07  	bge	s10, s9, 0x14584 <_vfprintf_r+0x1096>
   13d0a: 85 27        	addiw	a5, a5, 1
   13d0c: 41 07        	addi	a4, a4, 16
   13d0e: 23 30 74 01  	sd	s7, 0(s0)
   13d12: 23 34 a4 01  	sd	s10, 8(s0)
   13d16: 3a ea        	sd	a4, 272(sp)
   13d18: 23 24 f1 10  	sw	a5, 264(sp)
   13d1c: 41 04        	addi	s0, s0, 16
   13d1e: e3 d3 fd fe  	bge	s11, a5, 0x13d04 <_vfprintf_r+0x816>
   13d22: 10 02        	addi	a2, sp, 256
   13d24: d6 85        	mv	a1, s5
   13d26: 4e 85        	mv	a0, s3
   13d28: ef 70 30 4d  	jal	0x1b9fa <__sprint_r>
   13d2c: 63 19 05 1c  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13d30: 52 67        	ld	a4, 272(sp)
   13d32: 83 27 81 10  	lw	a5, 264(sp)
   13d36: 00 03        	addi	s0, sp, 384
   13d38: f1 b7        	j	0x13d04 <_vfprintf_r+0x816>
   13d3a: ca 8c        	mv	s9, s2
   13d3c: 05 47        	addi	a4, zero, 1
   13d3e: 63 88 e7 26  	beq	a5, a4, 0x13fae <_vfprintf_r+0xac0>
   13d42: 09 47        	addi	a4, zero, 2
   13d44: 63 89 e7 04  	beq	a5, a4, 0x13d96 <_vfprintf_r+0x8a8>
   13d48: 93 0a c1 17  	addi	s5, sp, 380
   13d4c: 56 8b        	mv	s6, s5
   13d4e: 13 f7 7b 00  	andi	a4, s7, 7
   13d52: 13 07 07 03  	addi	a4, a4, 48
   13d56: a3 0f eb fe  	sb	a4, -1(s6)
   13d5a: 93 db 3b 00  	srli	s7, s7, 3
   13d5e: 5a 86        	mv	a2, s6
   13d60: 7d 1b        	addi	s6, s6, -1
   13d62: e3 96 0b fe  	bnez	s7, 0x13d4e <_vfprintf_r+0x860>
   13d66: 93 f5 1c 00  	andi	a1, s9, 1
   13d6a: b1 c5        	beqz	a1, 0x13db6 <_vfprintf_r+0x8c8>
   13d6c: 93 05 00 03  	addi	a1, zero, 48
   13d70: 63 03 b7 04  	beq	a4, a1, 0x13db6 <_vfprintf_r+0x8c8>
   13d74: 79 16        	addi	a2, a2, -2
   13d76: a3 0f bb fe  	sb	a1, -1(s6)
   13d7a: bb 8d ca 40  	subw	s11, s5, a2
   13d7e: 66 89        	mv	s2, s9
   13d80: 32 8b        	mv	s6, a2
   13d82: cd b3        	j	0x13b64 <_vfprintf_r+0x676>
   13d84: 05 47        	addi	a4, zero, 1
   13d86: 63 94 e7 00  	bne	a5, a4, 0x13d8e <_vfprintf_r+0x8a0>
   13d8a: 6f 10 60 0a  	j	0x14e30 <_vfprintf_r+0x1942>
   13d8e: 09 47        	addi	a4, zero, 2
   13d90: ca 8c        	mv	s9, s2
   13d92: e3 9b e7 fa  	bne	a5, a4, 0x13d48 <_vfprintf_r+0x85a>
   13d96: 26 77        	ld	a4, 104(sp)
   13d98: 93 0a c1 17  	addi	s5, sp, 380
   13d9c: 56 8b        	mv	s6, s5
   13d9e: 93 f7 fb 00  	andi	a5, s7, 15
   13da2: ba 97        	add	a5, a5, a4
   13da4: 83 c7 07 00  	lbu	a5, 0(a5)
   13da8: 93 db 4b 00  	srli	s7, s7, 4
   13dac: 7d 1b        	addi	s6, s6, -1
   13dae: 23 00 fb 00  	sb	a5, 0(s6)
   13db2: e3 96 0b fe  	bnez	s7, 0x13d9e <_vfprintf_r+0x8b0>
   13db6: bb 8d 6a 41  	subw	s11, s5, s6
   13dba: 66 89        	mv	s2, s9
   13dbc: 65 b3        	j	0x13b64 <_vfprintf_r+0x676>
   13dbe: bb 0a 8a 41  	subw	s5, s4, s8
   13dc2: e3 5c 50 95  	blez	s5, 0x1371a <_vfprintf_r+0x22c>
   13dc6: 37 2f 02 00  	lui	t5, 34
   13dca: c1 45        	addi	a1, zero, 16
   13dcc: 5e 87        	mv	a4, s7
   13dce: 03 26 81 10  	lw	a2, 264(sp)
   13dd2: 93 0b 0f e3  	addi	s7, t5, -464
   13dd6: 63 d0 55 07  	bge	a1, s5, 0x13e36 <_vfprintf_r+0x948>
   13dda: 26 e9        	sd	s1, 144(sp)
   13ddc: c1 4f        	addi	t6, zero, 16
   13dde: d6 84        	mv	s1, s5
   13de0: 9d 4c        	addi	s9, zero, 7
   13de2: d2 8a        	mv	s5, s4
   13de4: 4a 8a        	mv	s4, s2
   13de6: 5e 89        	mv	s2, s7
   13de8: da 8b        	mv	s7, s6
   13dea: 1a 8b        	mv	s6, t1
   13dec: 21 a0        	j	0x13df4 <_vfprintf_r+0x906>
   13dee: c1 34        	addiw	s1, s1, -16
   13df0: 63 dc 9f 02  	bge	t6, s1, 0x13e28 <_vfprintf_r+0x93a>
   13df4: 05 26        	addiw	a2, a2, 1
   13df6: 41 07        	addi	a4, a4, 16
   13df8: 23 30 24 01  	sd	s2, 0(s0)
   13dfc: 23 34 f4 01  	sd	t6, 8(s0)
   13e00: 3a ea        	sd	a4, 272(sp)
   13e02: 23 24 c1 10  	sw	a2, 264(sp)
   13e06: 41 04        	addi	s0, s0, 16
   13e08: e3 d3 cc fe  	bge	s9, a2, 0x13dee <_vfprintf_r+0x900>
   13e0c: 82 65        	ld	a1, 0(sp)
   13e0e: 10 02        	addi	a2, sp, 256
   13e10: 4e 85        	mv	a0, s3
   13e12: ef 70 90 3e  	jal	0x1b9fa <__sprint_r>
   13e16: 65 e5        	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13e18: c1 4f        	addi	t6, zero, 16
   13e1a: c1 34        	addiw	s1, s1, -16
   13e1c: 52 67        	ld	a4, 272(sp)
   13e1e: 03 26 81 10  	lw	a2, 264(sp)
   13e22: 00 03        	addi	s0, sp, 384
   13e24: e3 c8 9f fc  	blt	t6, s1, 0x13df4 <_vfprintf_r+0x906>
   13e28: 5a 83        	mv	t1, s6
   13e2a: 5e 8b        	mv	s6, s7
   13e2c: ca 8b        	mv	s7, s2
   13e2e: 52 89        	mv	s2, s4
   13e30: 56 8a        	mv	s4, s5
   13e32: a6 8a        	mv	s5, s1
   13e34: ca 64        	ld	s1, 144(sp)
   13e36: 9b 05 16 00  	addiw	a1, a2, 1
   13e3a: 56 97        	add	a4, a4, s5
   13e3c: 23 30 74 01  	sd	s7, 0(s0)
   13e40: 23 34 54 01  	sd	s5, 8(s0)
   13e44: 3a ea        	sd	a4, 272(sp)
   13e46: 23 24 b1 10  	sw	a1, 264(sp)
   13e4a: 1d 46        	addi	a2, zero, 7
   13e4c: 63 4b b6 4e  	blt	a2, a1, 0x14342 <_vfprintf_r+0xe54>
   13e50: bb 0c b3 41  	subw	s9, t1, s11
   13e54: 41 04        	addi	s0, s0, 16
   13e56: ba 8b        	mv	s7, a4
   13e58: e3 55 90 8d  	blez	s9, 0x13722 <_vfprintf_r+0x234>
   13e5c: 37 2f 02 00  	lui	t5, 34
   13e60: c1 45        	addi	a1, zero, 16
   13e62: 5e 87        	mv	a4, s7
   13e64: 03 26 81 10  	lw	a2, 264(sp)
   13e68: 93 0b 0f e3  	addi	s7, t5, -464
   13e6c: 63 d1 95 07  	bge	a1, s9, 0x13ece <_vfprintf_r+0x9e0>
   13e70: a2 6b        	ld	s7, 8(sp)
   13e72: c1 48        	addi	a7, zero, 16
   13e74: 9d 4a        	addi	s5, zero, 7
   13e76: de 87        	mv	a5, s7
   13e78: da 8b        	mv	s7, s6
   13e7a: 52 8b        	mv	s6, s4
   13e7c: 4a 8a        	mv	s4, s2
   13e7e: 26 89        	mv	s2, s1
   13e80: e6 84        	mv	s1, s9
   13e82: be 8c        	mv	s9, a5
   13e84: 21 a0        	j	0x13e8c <_vfprintf_r+0x99e>
   13e86: c1 34        	addiw	s1, s1, -16
   13e88: 63 dc 98 02  	bge	a7, s1, 0x13ec0 <_vfprintf_r+0x9d2>
   13e8c: a2 66        	ld	a3, 8(sp)
   13e8e: 05 26        	addiw	a2, a2, 1
   13e90: 41 07        	addi	a4, a4, 16
   13e92: 14 e0        	sd	a3, 0(s0)
   13e94: 23 34 14 01  	sd	a7, 8(s0)
   13e98: 3a ea        	sd	a4, 272(sp)
   13e9a: 23 24 c1 10  	sw	a2, 264(sp)
   13e9e: 41 04        	addi	s0, s0, 16
   13ea0: e3 d3 ca fe  	bge	s5, a2, 0x13e86 <_vfprintf_r+0x998>
   13ea4: 82 65        	ld	a1, 0(sp)
   13ea6: 10 02        	addi	a2, sp, 256
   13ea8: 4e 85        	mv	a0, s3
   13eaa: ef 70 10 35  	jal	0x1b9fa <__sprint_r>
   13eae: 21 e9        	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13eb0: c1 48        	addi	a7, zero, 16
   13eb2: c1 34        	addiw	s1, s1, -16
   13eb4: 52 67        	ld	a4, 272(sp)
   13eb6: 03 26 81 10  	lw	a2, 264(sp)
   13eba: 00 03        	addi	s0, sp, 384
   13ebc: e3 c8 98 fc  	blt	a7, s1, 0x13e8c <_vfprintf_r+0x99e>
   13ec0: e6 87        	mv	a5, s9
   13ec2: a6 8c        	mv	s9, s1
   13ec4: ca 84        	mv	s1, s2
   13ec6: 52 89        	mv	s2, s4
   13ec8: 5a 8a        	mv	s4, s6
   13eca: 5e 8b        	mv	s6, s7
   13ecc: be 8b        	mv	s7, a5
   13ece: 9b 05 16 00  	addiw	a1, a2, 1
   13ed2: 66 97        	add	a4, a4, s9
   13ed4: 23 30 74 01  	sd	s7, 0(s0)
   13ed8: 23 34 94 01  	sd	s9, 8(s0)
   13edc: 3a ea        	sd	a4, 272(sp)
   13ede: 23 24 b1 10  	sw	a1, 264(sp)
   13ee2: 1d 46        	addi	a2, zero, 7
   13ee4: 63 43 b6 34  	blt	a2, a1, 0x1422a <_vfprintf_r+0xd3c>
   13ee8: 41 04        	addi	s0, s0, 16
   13eea: ba 8b        	mv	s7, a4
   13eec: 6f f0 7f 83  	j	0x13722 <_vfprintf_r+0x234>
   13ef0: 82 65        	ld	a1, 0(sp)
   13ef2: 10 02        	addi	a2, sp, 256
   13ef4: 4e 85        	mv	a0, s3
   13ef6: ef 70 50 30  	jal	0x1b9fa <__sprint_r>
   13efa: e3 0a 05 86  	beqz	a0, 0x1376e <_vfprintf_r+0x280>
   13efe: e2 65        	ld	a1, 24(sp)
   13f00: e3 89 05 88  	beqz	a1, 0x13792 <_vfprintf_r+0x2a4>
   13f04: 46 65        	ld	a0, 80(sp)
   13f06: ef 10 10 16  	jal	0x15866 <_free_r>
   13f0a: 61 b0        	j	0x13792 <_vfprintf_r+0x2a4>
   13f0c: b7 28 02 00  	lui	a7, 34
   13f10: 41 45        	addi	a0, zero, 16
   13f12: d2 65        	ld	a1, 272(sp)
   13f14: 03 26 81 10  	lw	a2, 264(sp)
   13f18: 93 8b 08 e4  	addi	s7, a7, -448
   13f1c: 63 5e e5 04  	bge	a0, a4, 0x13f78 <_vfprintf_r+0xa8a>
   13f20: 26 e9        	sd	s1, 144(sp)
   13f22: a2 87        	mv	a5, s0
   13f24: de 84        	mv	s1, s7
   13f26: 41 4f        	addi	t5, zero, 16
   13f28: da 8b        	mv	s7, s6
   13f2a: 9d 43        	addi	t2, zero, 7
   13f2c: 3a 84        	mv	s0, a4
   13f2e: 1a 8b        	mv	s6, t1
   13f30: 21 a0        	j	0x13f38 <_vfprintf_r+0xa4a>
   13f32: 41 34        	addiw	s0, s0, -16
   13f34: 63 5c 8f 02  	bge	t5, s0, 0x13f6c <_vfprintf_r+0xa7e>
   13f38: 05 26        	addiw	a2, a2, 1
   13f3a: c1 05        	addi	a1, a1, 16
   13f3c: 84 e3        	sd	s1, 0(a5)
   13f3e: 23 b4 e7 01  	sd	t5, 8(a5)
   13f42: 2e ea        	sd	a1, 272(sp)
   13f44: 23 24 c1 10  	sw	a2, 264(sp)
   13f48: c1 07        	addi	a5, a5, 16
   13f4a: e3 d4 c3 fe  	bge	t2, a2, 0x13f32 <_vfprintf_r+0xa44>
   13f4e: 82 65        	ld	a1, 0(sp)
   13f50: 10 02        	addi	a2, sp, 256
   13f52: 4e 85        	mv	a0, s3
   13f54: ef 70 70 2a  	jal	0x1b9fa <__sprint_r>
   13f58: 5d f1        	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13f5a: 41 4f        	addi	t5, zero, 16
   13f5c: 41 34        	addiw	s0, s0, -16
   13f5e: d2 65        	ld	a1, 272(sp)
   13f60: 03 26 81 10  	lw	a2, 264(sp)
   13f64: 1c 03        	addi	a5, sp, 384
   13f66: 9d 43        	addi	t2, zero, 7
   13f68: e3 48 8f fc  	blt	t5, s0, 0x13f38 <_vfprintf_r+0xa4a>
   13f6c: 5a 83        	mv	t1, s6
   13f6e: 5e 8b        	mv	s6, s7
   13f70: a6 8b        	mv	s7, s1
   13f72: ca 64        	ld	s1, 144(sp)
   13f74: 22 87        	mv	a4, s0
   13f76: 3e 84        	mv	s0, a5
   13f78: ba 95        	add	a1, a1, a4
   13f7a: 05 26        	addiw	a2, a2, 1
   13f7c: 18 e4        	sd	a4, 8(s0)
   13f7e: 23 30 74 01  	sd	s7, 0(s0)
   13f82: 2e ea        	sd	a1, 272(sp)
   13f84: 23 24 c1 10  	sw	a2, 264(sp)
   13f88: 9b 05 06 00  	sext.w	a1, a2
   13f8c: 1d 47        	addi	a4, zero, 7
   13f8e: 41 04        	addi	s0, s0, 16
   13f90: 63 58 b7 f2  	bge	a4, a1, 0x136c0 <_vfprintf_r+0x1d2>
   13f94: 82 65        	ld	a1, 0(sp)
   13f96: 10 02        	addi	a2, sp, 256
   13f98: 4e 85        	mv	a0, s3
   13f9a: 1a e9        	sd	t1, 144(sp)
   13f9c: ef 70 f0 25  	jal	0x1b9fa <__sprint_r>
   13fa0: 39 fd        	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13fa2: 4a 63        	ld	t1, 144(sp)
   13fa4: 00 03        	addi	s0, sp, 384
   13fa6: 6f f0 af f1  	j	0x136c0 <_vfprintf_r+0x1d2>
   13faa: ca 8c        	mv	s9, s2
   13fac: 3a f0        	sd	a4, 32(sp)
   13fae: a5 47        	addi	a5, zero, 9
   13fb0: 63 ec 77 4f  	bltu	a5, s7, 0x144a8 <_vfprintf_r+0xfba>
   13fb4: 9b 8b 0b 03  	addiw	s7, s7, 48
   13fb8: a3 0d 71 17  	sb	s7, 379(sp)
   13fbc: 66 89        	mv	s2, s9
   13fbe: 85 4d        	addi	s11, zero, 1
   13fc0: 13 0b b1 17  	addi	s6, sp, 379
   13fc4: 45 b6        	j	0x13b64 <_vfprintf_r+0x676>
   13fc6: 82 65        	ld	a1, 0(sp)
   13fc8: 10 02        	addi	a2, sp, 256
   13fca: 4e 85        	mv	a0, s3
   13fcc: 1a e9        	sd	t1, 144(sp)
   13fce: ef 70 d0 22  	jal	0x1b9fa <__sprint_r>
   13fd2: 15 f5        	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   13fd4: d2 6b        	ld	s7, 272(sp)
   13fd6: 4a 63        	ld	t1, 144(sp)
   13fd8: 00 03        	addi	s0, sp, 384
   13fda: 6f f0 8f f3  	j	0x13712 <_vfprintf_r+0x224>
   13fde: b7 28 02 00  	lui	a7, 34
   13fe2: c1 46        	addi	a3, zero, 16
   13fe4: 83 27 81 10  	lw	a5, 264(sp)
   13fe8: 93 8b 08 e4  	addi	s7, a7, -448
   13fec: 63 d3 26 05  	bge	a3, s2, 0x14032 <_vfprintf_r+0xb44>
   13ff0: 82 6a        	ld	s5, 0(sp)
   13ff2: c1 4c        	addi	s9, zero, 16
   13ff4: 1d 4d        	addi	s10, zero, 7
   13ff6: 21 a0        	j	0x13ffe <_vfprintf_r+0xb10>
   13ff8: 41 39        	addiw	s2, s2, -16
   13ffa: 63 dc 2c 03  	bge	s9, s2, 0x14032 <_vfprintf_r+0xb44>
   13ffe: 85 27        	addiw	a5, a5, 1
   14000: 41 07        	addi	a4, a4, 16
   14002: 23 30 74 01  	sd	s7, 0(s0)
   14006: 23 34 94 01  	sd	s9, 8(s0)
   1400a: 3a ea        	sd	a4, 272(sp)
   1400c: 23 24 f1 10  	sw	a5, 264(sp)
   14010: 41 04        	addi	s0, s0, 16
   14012: e3 53 fd fe  	bge	s10, a5, 0x13ff8 <_vfprintf_r+0xb0a>
   14016: 10 02        	addi	a2, sp, 256
   14018: d6 85        	mv	a1, s5
   1401a: 4e 85        	mv	a0, s3
   1401c: ef 70 f0 1d  	jal	0x1b9fa <__sprint_r>
   14020: e3 1f 05 ec  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14024: 41 39        	addiw	s2, s2, -16
   14026: 52 67        	ld	a4, 272(sp)
   14028: 83 27 81 10  	lw	a5, 264(sp)
   1402c: 00 03        	addi	s0, sp, 384
   1402e: e3 c8 2c fd  	blt	s9, s2, 0x13ffe <_vfprintf_r+0xb10>
   14032: 9b 86 17 00  	addiw	a3, a5, 1
   14036: 4a 97        	add	a4, a4, s2
   14038: 23 30 74 01  	sd	s7, 0(s0)
   1403c: 23 34 24 01  	sd	s2, 8(s0)
   14040: 3a ea        	sd	a4, 272(sp)
   14042: 23 24 d1 10  	sw	a3, 264(sp)
   14046: 9d 47        	addi	a5, zero, 7
   14048: 63 da d7 f0  	bge	a5, a3, 0x1375c <_vfprintf_r+0x26e>
   1404c: 82 65        	ld	a1, 0(sp)
   1404e: 10 02        	addi	a2, sp, 256
   14050: 4e 85        	mv	a0, s3
   14052: ef 70 90 1a  	jal	0x1b9fa <__sprint_r>
   14056: e3 14 05 ea  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1405a: 52 67        	ld	a4, 272(sp)
   1405c: 6f f0 0f f0  	j	0x1375c <_vfprintf_r+0x26e>
   14060: 6e 46        	lw	a2, 216(sp)
   14062: 63 5e c0 48  	blez	a2, 0x144fe <_vfprintf_r+0x1010>
   14066: e2 77        	ld	a5, 56(sp)
   14068: 42 77        	ld	a4, 48(sp)
   1406a: b3 0a fb 00  	add	s5, s6, a5
   1406e: ba 86        	mv	a3, a4
   14070: 63 c3 e7 2c  	blt	a5, a4, 0x14336 <_vfprintf_r+0xe48>
   14074: 1b 8d 06 00  	sext.w	s10, a3
   14078: 63 52 a0 03  	blez	s10, 0x1409c <_vfprintf_r+0xbae>
   1407c: 03 26 81 10  	lw	a2, 264(sp)
   14080: ea 9b        	add	s7, s7, s10
   14082: 23 30 64 01  	sd	s6, 0(s0)
   14086: 9b 05 16 00  	addiw	a1, a2, 1
   1408a: 23 34 a4 01  	sd	s10, 8(s0)
   1408e: 5e ea        	sd	s7, 272(sp)
   14090: 23 24 b1 10  	sw	a1, 264(sp)
   14094: 1d 46        	addi	a2, zero, 7
   14096: 41 04        	addi	s0, s0, 16
   14098: e3 45 b6 16  	blt	a2, a1, 0x14a02 <_vfprintf_r+0x1514>
   1409c: 13 46 fd ff  	not	a2, s10
   140a0: c2 77        	ld	a5, 48(sp)
   140a2: 7d 96        	srai	a2, a2, 63
   140a4: b3 76 cd 00  	and	a3, s10, a2
   140a8: 3b 8d d7 40  	subw	s10, a5, a3
   140ac: 63 41 a0 37  	bgtz	s10, 0x1440e <_vfprintf_r+0xf20>
   140b0: c2 77        	ld	a5, 48(sp)
   140b2: 93 76 09 40  	andi	a3, s2, 1024
   140b6: 3e 9b        	add	s6, s6, a5
   140b8: 63 90 06 62  	bnez	a3, 0x146d8 <_vfprintf_r+0x11ea>
   140bc: ee 46        	lw	a3, 216(sp)
   140be: e2 77        	ld	a5, 56(sp)
   140c0: 63 c6 f6 00  	blt	a3, a5, 0x140cc <_vfprintf_r+0xbde>
   140c4: 13 76 19 00  	andi	a2, s2, 1
   140c8: e3 0a 06 14  	beqz	a2, 0x14a1c <_vfprintf_r+0x152e>
   140cc: 03 26 81 10  	lw	a2, 264(sp)
   140d0: e6 67        	ld	a5, 88(sp)
   140d2: 86 75        	ld	a1, 96(sp)
   140d4: 41 04        	addi	s0, s0, 16
   140d6: 33 87 77 01  	add	a4, a5, s7
   140da: 23 38 b4 fe  	sd	a1, -16(s0)
   140de: 9b 05 16 00  	addiw	a1, a2, 1
   140e2: 23 3c f4 fe  	sd	a5, -8(s0)
   140e6: 3a ea        	sd	a4, 272(sp)
   140e8: 23 24 b1 10  	sw	a1, 264(sp)
   140ec: 1d 46        	addi	a2, zero, 7
   140ee: e3 4a b6 4a  	blt	a2, a1, 0x14da2 <_vfprintf_r+0x18b4>
   140f2: e2 77        	ld	a5, 56(sp)
   140f4: b3 8c 6a 41  	sub	s9, s5, s6
   140f8: 9b 85 0c 00  	sext.w	a1, s9
   140fc: 3b 86 d7 40  	subw	a2, a5, a3
   14100: 63 53 b6 00  	bge	a2, a1, 0x14106 <_vfprintf_r+0xc18>
   14104: b2 8c        	mv	s9, a2
   14106: 81 2c        	sext.w	s9, s9
   14108: 63 51 90 03  	blez	s9, 0x1412a <_vfprintf_r+0xc3c>
   1410c: 83 26 81 10  	lw	a3, 264(sp)
   14110: 66 97        	add	a4, a4, s9
   14112: 23 30 64 01  	sd	s6, 0(s0)
   14116: 85 26        	addiw	a3, a3, 1
   14118: 23 34 94 01  	sd	s9, 8(s0)
   1411c: 3a ea        	sd	a4, 272(sp)
   1411e: 23 24 d1 10  	sw	a3, 264(sp)
   14122: 9d 47        	addi	a5, zero, 7
   14124: 41 04        	addi	s0, s0, 16
   14126: e3 c6 d7 4e  	blt	a5, a3, 0x14e12 <_vfprintf_r+0x1924>
   1412a: 93 c7 fc ff  	not	a5, s9
   1412e: fd 97        	srai	a5, a5, 63
   14130: b3 fc fc 00  	and	s9, s9, a5
   14134: bb 0c 96 41  	subw	s9, a2, s9
   14138: 63 5a 90 e1  	blez	s9, 0x1374c <_vfprintf_r+0x25e>
   1413c: 37 2f 02 00  	lui	t5, 34
   14140: c1 46        	addi	a3, zero, 16
   14142: 83 27 81 10  	lw	a5, 264(sp)
   14146: 93 0b 0f e3  	addi	s7, t5, -464
   1414a: 63 dd 96 43  	bge	a3, s9, 0x14584 <_vfprintf_r+0x1096>
   1414e: 82 6a        	ld	s5, 0(sp)
   14150: 41 4d        	addi	s10, zero, 16
   14152: 9d 4d        	addi	s11, zero, 7
   14154: 21 a0        	j	0x1415c <_vfprintf_r+0xc6e>
   14156: c1 3c        	addiw	s9, s9, -16
   14158: 63 56 9d 43  	bge	s10, s9, 0x14584 <_vfprintf_r+0x1096>
   1415c: 85 27        	addiw	a5, a5, 1
   1415e: 41 07        	addi	a4, a4, 16
   14160: 23 30 74 01  	sd	s7, 0(s0)
   14164: 23 34 a4 01  	sd	s10, 8(s0)
   14168: 3a ea        	sd	a4, 272(sp)
   1416a: 23 24 f1 10  	sw	a5, 264(sp)
   1416e: 41 04        	addi	s0, s0, 16
   14170: e3 d3 fd fe  	bge	s11, a5, 0x14156 <_vfprintf_r+0xc68>
   14174: 10 02        	addi	a2, sp, 256
   14176: d6 85        	mv	a1, s5
   14178: 4e 85        	mv	a0, s3
   1417a: ef 70 10 08  	jal	0x1b9fa <__sprint_r>
   1417e: e3 10 05 d8  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14182: 52 67        	ld	a4, 272(sp)
   14184: 83 27 81 10  	lw	a5, 264(sp)
   14188: 00 03        	addi	s0, sp, 384
   1418a: f1 b7        	j	0x14156 <_vfprintf_r+0xc68>
   1418c: 13 76 19 00  	andi	a2, s2, 1
   14190: 63 12 06 f8  	bnez	a2, 0x13914 <_vfprintf_r+0x426>
   14194: 03 26 81 10  	lw	a2, 264(sp)
   14198: 13 8d 1b 00  	addi	s10, s7, 1
   1419c: 23 30 64 01  	sd	s6, 0(s0)
   141a0: 9b 07 16 00  	addiw	a5, a2, 1
   141a4: 14 e4        	sd	a3, 8(s0)
   141a6: 23 24 f1 10  	sw	a5, 264(sp)
   141aa: 1b 84 07 00  	sext.w	s0, a5
   141ae: 6a ea        	sd	s10, 272(sp)
   141b0: 9d 47        	addi	a5, zero, 7
   141b2: 63 dd 87 fc  	bge	a5, s0, 0x1398c <_vfprintf_r+0x49e>
   141b6: 82 65        	ld	a1, 0(sp)
   141b8: 10 02        	addi	a2, sp, 256
   141ba: 4e 85        	mv	a0, s3
   141bc: ef 70 f0 03  	jal	0x1b9fa <__sprint_r>
   141c0: e3 1f 05 d2  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   141c4: 52 6d        	ld	s10, 272(sp)
   141c6: 03 24 81 10  	lw	s0, 264(sp)
   141ca: 93 0c 01 18  	addi	s9, sp, 384
   141ce: 6f f0 ef fb  	j	0x1398c <_vfprintf_r+0x49e>
   141d2: 62 77        	ld	a4, 56(sp)
   141d4: 85 47        	addi	a5, zero, 1
   141d6: 63 db e7 fa  	bge	a5, a4, 0x1398c <_vfprintf_r+0x49e>
   141da: 37 2f 02 00  	lui	t5, 34
   141de: c5 47        	addi	a5, zero, 17
   141e0: 93 0b 0f e3  	addi	s7, t5, -464
   141e4: 63 d4 e7 44  	bge	a5, a4, 0x1462c <_vfprintf_r+0x113e>
   141e8: 82 6a        	ld	s5, 0(sp)
   141ea: 41 47        	addi	a4, zero, 16
   141ec: 1d 4b        	addi	s6, zero, 7
   141ee: 21 a0        	j	0x141f6 <_vfprintf_r+0xd08>
   141f0: c1 3d        	addiw	s11, s11, -16
   141f2: 63 5d b7 43  	bge	a4, s11, 0x1462c <_vfprintf_r+0x113e>
   141f6: 05 24        	addiw	s0, s0, 1
   141f8: 41 0d        	addi	s10, s10, 16
   141fa: 23 b0 7c 01  	sd	s7, 0(s9)
   141fe: 23 b4 ec 00  	sd	a4, 8(s9)
   14202: 6a ea        	sd	s10, 272(sp)
   14204: 23 24 81 10  	sw	s0, 264(sp)
   14208: c1 0c        	addi	s9, s9, 16
   1420a: e3 53 8b fe  	bge	s6, s0, 0x141f0 <_vfprintf_r+0xd02>
   1420e: 10 02        	addi	a2, sp, 256
   14210: d6 85        	mv	a1, s5
   14212: 4e 85        	mv	a0, s3
   14214: ef 70 60 7e  	jal	0x1b9fa <__sprint_r>
   14218: e3 13 05 ce  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1421c: 52 6d        	ld	s10, 272(sp)
   1421e: 03 24 81 10  	lw	s0, 264(sp)
   14222: 93 0c 01 18  	addi	s9, sp, 384
   14226: 41 47        	addi	a4, zero, 16
   14228: e1 b7        	j	0x141f0 <_vfprintf_r+0xd02>
   1422a: 82 65        	ld	a1, 0(sp)
   1422c: 10 02        	addi	a2, sp, 256
   1422e: 4e 85        	mv	a0, s3
   14230: ef 70 a0 7c  	jal	0x1b9fa <__sprint_r>
   14234: e3 15 05 cc  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14238: d2 6b        	ld	s7, 272(sp)
   1423a: 00 03        	addi	s0, sp, 384
   1423c: 6f f0 6f ce  	j	0x13722 <_vfprintf_r+0x234>
   14240: 93 77 09 01  	andi	a5, s2, 16
   14244: dd e7        	bnez	a5, 0x142f2 <_vfprintf_r+0xe04>
   14246: 82 76        	ld	a3, 32(sp)
   14248: 93 77 09 04  	andi	a5, s2, 64
   1424c: 83 ab 06 00  	lw	s7, 0(a3)
   14250: 63 82 07 40  	beqz	a5, 0x14654 <_vfprintf_r+0x1166>
   14254: 9b 9b 0b 01  	slliw	s7, s7, 16
   14258: 9b db 0b 41  	sraiw	s7, s7, 16
   1425c: de 87        	mv	a5, s7
   1425e: e3 d7 07 94  	bgez	a5, 0x13bac <_vfprintf_r+0x6be>
   14262: 3a f0        	sd	a4, 32(sp)
   14264: b3 0b 70 41  	neg	s7, s7
   14268: ca 8c        	mv	s9, s2
   1426a: 13 07 d0 02  	addi	a4, zero, 45
   1426e: 85 47        	addi	a5, zero, 1
   14270: f1 b0        	j	0x13b3c <_vfprintf_r+0x64e>
   14272: 93 77 09 01  	andi	a5, s2, 16
   14276: b5 eb        	bnez	a5, 0x142ea <_vfprintf_r+0xdfc>
   14278: 82 76        	ld	a3, 32(sp)
   1427a: 93 77 09 04  	andi	a5, s2, 64
   1427e: 83 ab 06 00  	lw	s7, 0(a3)
   14282: 63 8e 07 3e  	beqz	a5, 0x1467e <_vfprintf_r+0x1190>
   14286: c2 1b        	slli	s7, s7, 48
   14288: 93 db 0b 03  	srli	s7, s7, 48
   1428c: b9 ba        	j	0x13bea <_vfprintf_r+0x6fc>
   1428e: 93 f7 0c 01  	andi	a5, s9, 16
   14292: b1 e7        	bnez	a5, 0x142de <_vfprintf_r+0xdf0>
   14294: 82 76        	ld	a3, 32(sp)
   14296: 93 f7 0c 04  	andi	a5, s9, 64
   1429a: 83 ab 06 00  	lw	s7, 0(a3)
   1429e: 63 86 07 3c  	beqz	a5, 0x1466a <_vfprintf_r+0x117c>
   142a2: c2 1b        	slli	s7, s7, 48
   142a4: 93 db 0b 03  	srli	s7, s7, 48
   142a8: 3a f0        	sd	a4, 32(sp)
   142aa: 85 47        	addi	a5, zero, 1
   142ac: 79 b0        	j	0x13b3a <_vfprintf_r+0x64c>
   142ae: 81 4d        	mv	s11, zero
   142b0: 13 0b c1 17  	addi	s6, sp, 380
   142b4: 45 b8        	j	0x13b64 <_vfprintf_r+0x676>
   142b6: 93 77 19 00  	andi	a5, s2, 1
   142ba: 5e 87        	mv	a4, s7
   142bc: 63 88 07 c8  	beqz	a5, 0x1374c <_vfprintf_r+0x25e>
   142c0: f5 ba        	j	0x13cbc <_vfprintf_r+0x7ce>
   142c2: 56 83        	mv	t1, s5
   142c4: c9 b8        	j	0x13b96 <_vfprintf_r+0x6a8>
   142c6: 03 ce 04 00  	lbu	t3, 0(s1)
   142ca: 3e f0        	sd	a5, 32(sp)
   142cc: 6f f0 4f b6  	j	0x13630 <_vfprintf_r+0x142>
   142d0: 93 07 00 03  	addi	a5, zero, 48
   142d4: a3 0d f1 16  	sb	a5, 379(sp)
   142d8: 13 0b b1 17  	addi	s6, sp, 379
   142dc: 61 b0        	j	0x13b64 <_vfprintf_r+0x676>
   142de: 82 77        	ld	a5, 32(sp)
   142e0: 3a f0        	sd	a4, 32(sp)
   142e2: 83 bb 07 00  	ld	s7, 0(a5)
   142e6: 85 47        	addi	a5, zero, 1
   142e8: 89 b8        	j	0x13b3a <_vfprintf_r+0x64c>
   142ea: 82 77        	ld	a5, 32(sp)
   142ec: 83 bb 07 00  	ld	s7, 0(a5)
   142f0: ed b8        	j	0x13bea <_vfprintf_r+0x6fc>
   142f2: 82 77        	ld	a5, 32(sp)
   142f4: 9c 63        	ld	a5, 0(a5)
   142f6: be 8b        	mv	s7, a5
   142f8: 45 b8        	j	0x13ba8 <_vfprintf_r+0x6ba>
   142fa: 93 77 09 01  	andi	a5, s2, 16
   142fe: e3 98 07 90  	bnez	a5, 0x13c0e <_vfprintf_r+0x720>
   14302: 93 77 09 04  	andi	a5, s2, 64
   14306: 63 9c 07 72  	bnez	a5, 0x14a3e <_vfprintf_r+0x1550>
   1430a: 13 79 09 20  	andi	s2, s2, 512
   1430e: c2 67        	ld	a5, 16(sp)
   14310: e3 0e 09 46  	beqz	s2, 0x14f8c <_vfprintf_r+0x1a9e>
   14314: 23 00 f7 00  	sb	a5, 0(a4)
   14318: ed b8        	j	0x13c12 <_vfprintf_r+0x724>
   1431a: 82 77        	ld	a5, 32(sp)
   1431c: 93 8a f7 00  	addi	s5, a5, 15
   14320: 93 fa 0a ff  	andi	s5, s5, -16
   14324: 93 87 0a 01  	addi	a5, s5, 16
   14328: 03 b7 0a 00  	ld	a4, 0(s5)
   1432c: 83 b5 8a 00  	ld	a1, 8(s5)
   14330: 3e f0        	sd	a5, 32(sp)
   14332: 6f f0 af d2  	j	0x1385c <_vfprintf_r+0x36e>
   14336: be 86        	mv	a3, a5
   14338: 1b 8d 06 00  	sext.w	s10, a3
   1433c: e3 40 a0 d5  	bgtz	s10, 0x1407c <_vfprintf_r+0xb8e>
   14340: b1 bb        	j	0x1409c <_vfprintf_r+0xbae>
   14342: 82 65        	ld	a1, 0(sp)
   14344: 10 02        	addi	a2, sp, 256
   14346: 4e 85        	mv	a0, s3
   14348: 1a e9        	sd	t1, 144(sp)
   1434a: ef 70 00 6b  	jal	0x1b9fa <__sprint_r>
   1434e: e3 18 05 ba  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14352: d2 6b        	ld	s7, 272(sp)
   14354: 4a 63        	ld	t1, 144(sp)
   14356: 00 03        	addi	s0, sp, 384
   14358: 6f f0 2f bc  	j	0x1371a <_vfprintf_r+0x22c>
   1435c: b7 27 02 00  	lui	a5, 34
   14360: 93 87 07 bf  	addi	a5, a5, -1040
   14364: 56 83        	mv	t1, s5
   14366: be f4        	sd	a5, 104(sp)
   14368: 82 76        	ld	a3, 32(sp)
   1436a: 93 77 09 02  	andi	a5, s2, 32
   1436e: 13 87 86 00  	addi	a4, a3, 8
   14372: e5 cb        	beqz	a5, 0x14462 <_vfprintf_r+0xf74>
   14374: 83 bb 06 00  	ld	s7, 0(a3)
   14378: 93 77 19 00  	andi	a5, s2, 1
   1437c: 81 cf        	beqz	a5, 0x14394 <_vfprintf_r+0xea6>
   1437e: 63 8b 0b 00  	beqz	s7, 0x14394 <_vfprintf_r+0xea6>
   14382: 13 69 29 00  	ori	s2, s2, 2
   14386: 93 07 00 03  	addi	a5, zero, 48
   1438a: 23 08 f1 0c  	sb	a5, 208(sp)
   1438e: a3 08 a1 0d  	sb	s10, 209(sp)
   14392: 01 29        	sext.w	s2, s2
   14394: 93 7c f9 bf  	andi	s9, s2, -1025
   14398: 81 2c        	sext.w	s9, s9
   1439a: 3a f0        	sd	a4, 32(sp)
   1439c: 89 47        	addi	a5, zero, 2
   1439e: 6f f0 cf f9  	j	0x13b3a <_vfprintf_r+0x64c>
   143a2: 56 83        	mv	t1, s5
   143a4: ca 8c        	mv	s9, s2
   143a6: 6f f0 ef f7  	j	0x13b24 <_vfprintf_r+0x636>
   143aa: b7 27 02 00  	lui	a5, 34
   143ae: 93 87 87 c0  	addi	a5, a5, -1016
   143b2: 56 83        	mv	t1, s5
   143b4: be f4        	sd	a5, 104(sp)
   143b6: 4d bf        	j	0x14368 <_vfprintf_r+0xe7a>
   143b8: 82 65        	ld	a1, 0(sp)
   143ba: 10 02        	addi	a2, sp, 256
   143bc: 4e 85        	mv	a0, s3
   143be: ef 70 c0 63  	jal	0x1b9fa <__sprint_r>
   143c2: e3 1e 05 b2  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   143c6: 52 67        	ld	a4, 272(sp)
   143c8: 00 03        	addi	s0, sp, 384
   143ca: 19 ba        	j	0x13ce0 <_vfprintf_r+0x7f2>
   143cc: 13 69 09 20  	ori	s2, s2, 512
   143d0: 03 ce 14 00  	lbu	t3, 1(s1)
   143d4: 01 29        	sext.w	s2, s2
   143d6: 85 04        	addi	s1, s1, 1
   143d8: 6f f0 8f a5  	j	0x13630 <_vfprintf_r+0x142>
   143dc: 56 83        	mv	t1, s5
   143de: 6f f0 af ff  	j	0x13bd8 <_vfprintf_r+0x6ea>
   143e2: 13 69 09 02  	ori	s2, s2, 32
   143e6: 03 ce 14 00  	lbu	t3, 1(s1)
   143ea: 01 29        	sext.w	s2, s2
   143ec: 85 04        	addi	s1, s1, 1
   143ee: 6f f0 2f a4  	j	0x13630 <_vfprintf_r+0x142>
   143f2: 99 47        	addi	a5, zero, 6
   143f4: 9a 86        	mv	a3, t1
   143f6: 63 e1 67 62  	bltu	a5, t1, 0x14a18 <_vfprintf_r+0x152a>
   143fa: 9b 8d 06 00  	sext.w	s11, a3
   143fe: b7 27 02 00  	lui	a5, 34
   14402: 6e 8c        	mv	s8, s11
   14404: 56 f0        	sd	s5, 32(sp)
   14406: 13 8b 07 c2  	addi	s6, a5, -992
   1440a: 6f f0 2f a9  	j	0x1369c <_vfprintf_r+0x1ae>
   1440e: 37 2f 02 00  	lui	t5, 34
   14412: c1 45        	addi	a1, zero, 16
   14414: 5e 87        	mv	a4, s7
   14416: 03 26 81 10  	lw	a2, 264(sp)
   1441a: 93 0b 0f e3  	addi	s7, t5, -464
   1441e: 63 dd a5 29  	bge	a1, s10, 0x146b8 <_vfprintf_r+0x11ca>
   14422: 82 6c        	ld	s9, 0(sp)
   14424: c1 48        	addi	a7, zero, 16
   14426: 9d 4d        	addi	s11, zero, 7
   14428: 21 a0        	j	0x14430 <_vfprintf_r+0xf42>
   1442a: 41 3d        	addiw	s10, s10, -16
   1442c: 63 d6 a8 29  	bge	a7, s10, 0x146b8 <_vfprintf_r+0x11ca>
   14430: 05 26        	addiw	a2, a2, 1
   14432: 41 07        	addi	a4, a4, 16
   14434: 23 30 74 01  	sd	s7, 0(s0)
   14438: 23 34 14 01  	sd	a7, 8(s0)
   1443c: 3a ea        	sd	a4, 272(sp)
   1443e: 23 24 c1 10  	sw	a2, 264(sp)
   14442: 41 04        	addi	s0, s0, 16
   14444: e3 d3 cd fe  	bge	s11, a2, 0x1442a <_vfprintf_r+0xf3c>
   14448: 10 02        	addi	a2, sp, 256
   1444a: e6 85        	mv	a1, s9
   1444c: 4e 85        	mv	a0, s3
   1444e: ef 70 c0 5a  	jal	0x1b9fa <__sprint_r>
   14452: e3 16 05 aa  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14456: 52 67        	ld	a4, 272(sp)
   14458: 03 26 81 10  	lw	a2, 264(sp)
   1445c: 00 03        	addi	s0, sp, 384
   1445e: c1 48        	addi	a7, zero, 16
   14460: e9 b7        	j	0x1442a <_vfprintf_r+0xf3c>
   14462: 93 77 09 01  	andi	a5, s2, 16
   14466: ed cb        	beqz	a5, 0x14558 <_vfprintf_r+0x106a>
   14468: 82 77        	ld	a5, 32(sp)
   1446a: 83 bb 07 00  	ld	s7, 0(a5)
   1446e: 29 b7        	j	0x14378 <_vfprintf_r+0xe8a>
   14470: 82 65        	ld	a1, 0(sp)
   14472: 10 02        	addi	a2, sp, 256
   14474: 4e 85        	mv	a0, s3
   14476: ef 70 40 58  	jal	0x1b9fa <__sprint_r>
   1447a: e3 12 05 a8  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1447e: 52 67        	ld	a4, 272(sp)
   14480: 83 26 81 10  	lw	a3, 264(sp)
   14484: 93 0c 01 18  	addi	s9, sp, 384
   14488: 6f f0 ef ca  	j	0x13936 <_vfprintf_r+0x448>
   1448c: 82 65        	ld	a1, 0(sp)
   1448e: 10 02        	addi	a2, sp, 256
   14490: 4e 85        	mv	a0, s3
   14492: ef 70 80 56  	jal	0x1b9fa <__sprint_r>
   14496: e3 14 05 a6  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1449a: 52 6d        	ld	s10, 272(sp)
   1449c: 03 24 81 10  	lw	s0, 264(sp)
   144a0: 93 0c 01 18  	addi	s9, sp, 384
   144a4: 6f f0 4f cb  	j	0x13958 <_vfprintf_r+0x46a>
   144a8: 93 0a c1 17  	addi	s5, sp, 380
   144ac: 0a 68        	ld	a6, 128(sp)
   144ae: 93 f7 0c 40  	andi	a5, s9, 1024
   144b2: 81 46        	mv	a3, zero
   144b4: d6 85        	mv	a1, s5
   144b6: 29 49        	addi	s2, zero, 10
   144b8: a5 4d        	addi	s11, zero, 9
   144ba: 13 0c f0 0f  	addi	s8, zero, 255
   144be: 39 a0        	j	0x144cc <_vfprintf_r+0xfde>
   144c0: 33 d7 2b 03  	divu	a4, s7, s2
   144c4: 63 f6 7d 03  	bgeu	s11, s7, 0x144f0 <_vfprintf_r+0x1002>
   144c8: ba 8b        	mv	s7, a4
   144ca: da 85        	mv	a1, s6
   144cc: 13 8b f5 ff  	addi	s6, a1, -1
   144d0: 85 26        	addiw	a3, a3, 1
   144d2: 33 f7 2b 03  	remu	a4, s7, s2
   144d6: 1b 07 07 03  	addiw	a4, a4, 48
   144da: a3 8f e5 fe  	sb	a4, -1(a1)
   144de: ed d3        	beqz	a5, 0x144c0 <_vfprintf_r+0xfd2>
   144e0: 03 47 08 00  	lbu	a4, 0(a6)
   144e4: e3 9e e6 fc  	bne	a3, a4, 0x144c0 <_vfprintf_r+0xfd2>
   144e8: e3 0c 87 fd  	beq	a4, s8, 0x144c0 <_vfprintf_r+0xfd2>
   144ec: 63 ec 7d 35  	bltu	s11, s7, 0x14844 <_vfprintf_r+0x1356>
   144f0: 36 fc        	sd	a3, 56(sp)
   144f2: 42 e1        	sd	a6, 128(sp)
   144f4: bb 8d 6a 41  	subw	s11, s5, s6
   144f8: 66 89        	mv	s2, s9
   144fa: 6f f0 af e6  	j	0x13b64 <_vfprintf_r+0x676>
   144fe: 83 26 81 10  	lw	a3, 264(sp)
   14502: b7 25 02 00  	lui	a1, 34
   14506: 93 85 85 c2  	addi	a1, a1, -984
   1450a: 85 26        	addiw	a3, a3, 1
   1450c: 0c e0        	sd	a1, 0(s0)
   1450e: 13 87 1b 00  	addi	a4, s7, 1
   14512: 85 45        	addi	a1, zero, 1
   14514: 0c e4        	sd	a1, 8(s0)
   14516: 23 24 d1 10  	sw	a3, 264(sp)
   1451a: 9b 85 06 00  	sext.w	a1, a3
   1451e: 3a ea        	sd	a4, 272(sp)
   14520: 9d 46        	addi	a3, zero, 7
   14522: 41 04        	addi	s0, s0, 16
   14524: 63 c8 b6 34  	blt	a3, a1, 0x14874 <_vfprintf_r+0x1386>
   14528: 5d ee        	bnez	a2, 0x145e6 <_vfprintf_r+0x10f8>
   1452a: e2 77        	ld	a5, 56(sp)
   1452c: 93 76 19 00  	andi	a3, s2, 1
   14530: dd 8e        	or	a3, a3, a5
   14532: 63 8d 06 a0  	beqz	a3, 0x1374c <_vfprintf_r+0x25e>
   14536: 03 26 81 10  	lw	a2, 264(sp)
   1453a: e6 67        	ld	a5, 88(sp)
   1453c: 86 76        	ld	a3, 96(sp)
   1453e: 3e 97        	add	a4, a4, a5
   14540: 14 e0        	sd	a3, 0(s0)
   14542: 9b 06 16 00  	addiw	a3, a2, 1
   14546: 1c e4        	sd	a5, 8(s0)
   14548: 3a ea        	sd	a4, 272(sp)
   1454a: 23 24 d1 10  	sw	a3, 264(sp)
   1454e: 1d 46        	addi	a2, zero, 7
   14550: 63 49 d6 52  	blt	a2, a3, 0x14a82 <_vfprintf_r+0x1594>
   14554: 41 04        	addi	s0, s0, 16
   14556: 65 a8        	j	0x1460e <_vfprintf_r+0x1120>
   14558: 82 76        	ld	a3, 32(sp)
   1455a: 93 77 09 04  	andi	a5, s2, 64
   1455e: 83 ab 06 00  	lw	s7, 0(a3)
   14562: 63 86 07 12  	beqz	a5, 0x1468e <_vfprintf_r+0x11a0>
   14566: c2 1b        	slli	s7, s7, 48
   14568: 93 db 0b 03  	srli	s7, s7, 48
   1456c: 31 b5        	j	0x14378 <_vfprintf_r+0xe8a>
   1456e: 82 65        	ld	a1, 0(sp)
   14570: 10 02        	addi	a2, sp, 256
   14572: 4e 85        	mv	a0, s3
   14574: ef 70 60 48  	jal	0x1b9fa <__sprint_r>
   14578: e3 13 05 98  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1457c: d2 6b        	ld	s7, 272(sp)
   1457e: 00 03        	addi	s0, sp, 384
   14580: 6f f0 4f f3  	j	0x13cb4 <_vfprintf_r+0x7c6>
   14584: 9b 86 17 00  	addiw	a3, a5, 1
   14588: 23 30 74 01  	sd	s7, 0(s0)
   1458c: 23 34 94 01  	sd	s9, 8(s0)
   14590: 66 97        	add	a4, a4, s9
   14592: 3a ea        	sd	a4, 272(sp)
   14594: 23 24 d1 10  	sw	a3, 264(sp)
   14598: 9d 47        	addi	a5, zero, 7
   1459a: 63 d8 d7 9a  	bge	a5, a3, 0x1374a <_vfprintf_r+0x25c>
   1459e: 6f f0 4f c1  	j	0x139b2 <_vfprintf_r+0x4c4>
   145a2: 3a f0        	sd	a4, 32(sp)
   145a4: 29 b4        	j	0x13fae <_vfprintf_r+0xac0>
   145a6: 4e 75        	ld	a0, 240(sp)
   145a8: ee 75        	ld	a1, 248(sp)
   145aa: 01 46        	mv	a2, zero
   145ac: 81 46        	mv	a3, zero
   145ae: ef a0 c0 6c  	jal	0x1ec7a <__lttf2>
   145b2: 63 40 05 7e  	bltz	a0, 0x14d92 <_vfprintf_r+0x18a4>
   145b6: 03 47 f1 0c  	lbu	a4, 207(sp)
   145ba: 93 07 70 04  	addi	a5, zero, 71
   145be: 63 d6 a7 2d  	bge	a5, s10, 0x1488a <_vfprintf_r+0x139c>
   145c2: b7 27 02 00  	lui	a5, 34
   145c6: 13 8b 87 bd  	addi	s6, a5, -1064
   145ca: 13 79 f9 f7  	andi	s2, s2, -129
   145ce: 02 ec        	sd	zero, 24(sp)
   145d0: 82 e4        	sd	zero, 72(sp)
   145d2: 82 e0        	sd	zero, 64(sp)
   145d4: 02 f8        	sd	zero, 48(sp)
   145d6: 01 29        	sext.w	s2, s2
   145d8: 0d 4c        	addi	s8, zero, 3
   145da: 8d 4d        	addi	s11, zero, 3
   145dc: 01 43        	mv	t1, zero
   145de: 63 10 07 c6  	bnez	a4, 0x13a3e <_vfprintf_r+0x550>
   145e2: 6f f0 4f 8c  	j	0x136a6 <_vfprintf_r+0x1b8>
   145e6: 83 25 81 10  	lw	a1, 264(sp)
   145ea: e6 67        	ld	a5, 88(sp)
   145ec: 86 76        	ld	a3, 96(sp)
   145ee: 41 04        	addi	s0, s0, 16
   145f0: 3e 97        	add	a4, a4, a5
   145f2: 23 38 d4 fe  	sd	a3, -16(s0)
   145f6: 9b 86 15 00  	addiw	a3, a1, 1
   145fa: 23 3c f4 fe  	sd	a5, -8(s0)
   145fe: 3a ea        	sd	a4, 272(sp)
   14600: 23 24 d1 10  	sw	a3, 264(sp)
   14604: 9d 45        	addi	a1, zero, 7
   14606: 63 ce d5 46  	blt	a1, a3, 0x14a82 <_vfprintf_r+0x1594>
   1460a: e3 42 06 06  	bltz	a2, 0x14e6e <_vfprintf_r+0x1980>
   1460e: e2 77        	ld	a5, 56(sp)
   14610: 1b 86 16 00  	addiw	a2, a3, 1
   14614: 23 30 64 01  	sd	s6, 0(s0)
   14618: 3e 97        	add	a4, a4, a5
   1461a: 1c e4        	sd	a5, 8(s0)
   1461c: 3a ea        	sd	a4, 272(sp)
   1461e: 23 24 c1 10  	sw	a2, 264(sp)
   14622: 9d 47        	addi	a5, zero, 7
   14624: 63 d3 c7 92  	bge	a5, a2, 0x1374a <_vfprintf_r+0x25c>
   14628: 6f f0 af b8  	j	0x139b2 <_vfprintf_r+0x4c4>
   1462c: 05 24        	addiw	s0, s0, 1
   1462e: 6e 9d        	add	s10, s10, s11
   14630: 23 b0 7c 01  	sd	s7, 0(s9)
   14634: 23 b4 bc 01  	sd	s11, 8(s9)
   14638: 6a ea        	sd	s10, 272(sp)
   1463a: 23 24 81 10  	sw	s0, 264(sp)
   1463e: 9d 47        	addi	a5, zero, 7
   14640: 63 d5 87 b4  	bge	a5, s0, 0x1398a <_vfprintf_r+0x49c>
   14644: 8d be        	j	0x141b6 <_vfprintf_r+0xcc8>
   14646: 5a 85        	mv	a0, s6
   14648: ef 50 60 67  	jal	0x19cbe <strlen>
   1464c: 9b 0d 05 00  	sext.w	s11, a0
   14650: 6f f0 ef bc  	j	0x13a1e <_vfprintf_r+0x530>
   14654: 93 77 09 20  	andi	a5, s2, 512
   14658: 63 8f 07 3e  	beqz	a5, 0x14a56 <_vfprintf_r+0x1568>
   1465c: 9b 9b 8b 01  	slliw	s7, s7, 24
   14660: 9b db 8b 41  	sraiw	s7, s7, 24
   14664: de 87        	mv	a5, s7
   14666: 6f f0 2f d4  	j	0x13ba8 <_vfprintf_r+0x6ba>
   1466a: 93 f7 0c 20  	andi	a5, s9, 512
   1466e: 63 8d 07 3c  	beqz	a5, 0x14a48 <_vfprintf_r+0x155a>
   14672: 93 fb fb 0f  	andi	s7, s7, 255
   14676: 3a f0        	sd	a4, 32(sp)
   14678: 85 47        	addi	a5, zero, 1
   1467a: 6f f0 0f cc  	j	0x13b3a <_vfprintf_r+0x64c>
   1467e: 93 77 09 20  	andi	a5, s2, 512
   14682: 63 8d 07 3c  	beqz	a5, 0x14a5c <_vfprintf_r+0x156e>
   14686: 93 fb fb 0f  	andi	s7, s7, 255
   1468a: 6f f0 0f d6  	j	0x13bea <_vfprintf_r+0x6fc>
   1468e: 93 77 09 20  	andi	a5, s2, 512
   14692: 63 82 07 3a  	beqz	a5, 0x14a36 <_vfprintf_r+0x1548>
   14696: 93 fb fb 0f  	andi	s7, s7, 255
   1469a: f9 b9        	j	0x14378 <_vfprintf_r+0xe8a>
   1469c: ee 77        	ld	a5, 248(sp)
   1469e: 63 cb 07 3c  	bltz	a5, 0x14a74 <_vfprintf_r+0x1586>
   146a2: 03 47 f1 0c  	lbu	a4, 207(sp)
   146a6: 93 07 70 04  	addi	a5, zero, 71
   146aa: 63 de a7 75  	bge	a5, s10, 0x14e06 <_vfprintf_r+0x1918>
   146ae: b7 27 02 00  	lui	a5, 34
   146b2: 13 8b 87 be  	addi	s6, a5, -1048
   146b6: 11 bf        	j	0x145ca <_vfprintf_r+0x10dc>
   146b8: 9b 05 16 00  	addiw	a1, a2, 1
   146bc: 6a 97        	add	a4, a4, s10
   146be: 23 30 74 01  	sd	s7, 0(s0)
   146c2: 23 34 a4 01  	sd	s10, 8(s0)
   146c6: 3a ea        	sd	a4, 272(sp)
   146c8: 23 24 b1 10  	sw	a1, 264(sp)
   146cc: 9d 46        	addi	a3, zero, 7
   146ce: 63 c6 b6 56  	blt	a3, a1, 0x14c3a <_vfprintf_r+0x174c>
   146d2: 41 04        	addi	s0, s0, 16
   146d4: ba 8b        	mv	s7, a4
   146d6: e9 ba        	j	0x140b0 <_vfprintf_r+0xbc2>
   146d8: 26 e9        	sd	s1, 144(sp)
   146da: 86 64        	ld	s1, 64(sp)
   146dc: 37 2e 02 00  	lui	t3, 34
   146e0: 52 f1        	sd	s4, 160(sp)
   146e2: e2 e0        	sd	s8, 64(sp)
   146e4: 4a ed        	sd	s2, 152(sp)
   146e6: 5e 87        	mv	a4, s7
   146e8: 0a 6a        	ld	s4, 128(sp)
   146ea: 26 6c        	ld	s8, 72(sp)
   146ec: 82 6c        	ld	s9, 0(sp)
   146ee: aa 6b        	ld	s7, 136(sp)
   146f0: 1d 43        	addi	t1, zero, 7
   146f2: 41 4d        	addi	s10, zero, 16
   146f4: 93 0d 0e e3  	addi	s11, t3, -464
   146f8: c1 c0        	beqz	s1, 0x14778 <_vfprintf_r+0x128a>
   146fa: 63 08 0c 10  	beqz	s8, 0x1480a <_vfprintf_r+0x131c>
   146fe: 7d 3c        	addiw	s8, s8, -1
   14700: 83 27 81 10  	lw	a5, 264(sp)
   14704: e6 76        	ld	a3, 120(sp)
   14706: 5e 97        	add	a4, a4, s7
   14708: 23 34 74 01  	sd	s7, 8(s0)
   1470c: 14 e0        	sd	a3, 0(s0)
   1470e: 9b 86 17 00  	addiw	a3, a5, 1
   14712: 3a ea        	sd	a4, 272(sp)
   14714: 23 24 d1 10  	sw	a3, 264(sp)
   14718: 41 04        	addi	s0, s0, 16
   1471a: 63 4b d3 0e  	blt	t1, a3, 0x14810 <_vfprintf_r+0x1322>
   1471e: 03 46 0a 00  	lbu	a2, 0(s4)
   14722: b3 87 6a 41  	sub	a5, s5, s6
   14726: 9b 85 07 00  	sext.w	a1, a5
   1472a: 9b 06 06 00  	sext.w	a3, a2
   1472e: 63 53 b6 00  	bge	a2, a1, 0x14734 <_vfprintf_r+0x1246>
   14732: b6 87        	mv	a5, a3
   14734: 1b 89 07 00  	sext.w	s2, a5
   14738: 63 55 20 03  	blez	s2, 0x14762 <_vfprintf_r+0x1274>
   1473c: 83 26 81 10  	lw	a3, 264(sp)
   14740: 4a 97        	add	a4, a4, s2
   14742: 23 30 64 01  	sd	s6, 0(s0)
   14746: 1b 86 16 00  	addiw	a2, a3, 1
   1474a: 23 34 24 01  	sd	s2, 8(s0)
   1474e: 3a ea        	sd	a4, 272(sp)
   14750: 23 24 c1 10  	sw	a2, 264(sp)
   14754: 63 49 c3 0c  	blt	t1, a2, 0x14826 <_vfprintf_r+0x1338>
   14758: 03 46 0a 00  	lbu	a2, 0(s4)
   1475c: 41 04        	addi	s0, s0, 16
   1475e: 9b 06 06 00  	sext.w	a3, a2
   14762: 93 45 f9 ff  	not	a1, s2
   14766: fd 95        	srai	a1, a1, 63
   14768: b3 77 b9 00  	and	a5, s2, a1
   1476c: 3b 89 f6 40  	subw	s2, a3, a5
   14770: 63 40 20 03  	bgtz	s2, 0x14790 <_vfprintf_r+0x12a2>
   14774: 32 9b        	add	s6, s6, a2
   14776: d1 f0        	bnez	s1, 0x146fa <_vfprintf_r+0x120c>
   14778: e3 13 0c f8  	bnez	s8, 0x146fe <_vfprintf_r+0x1210>
   1477c: 52 e1        	sd	s4, 128(sp)
   1477e: ca 64        	ld	s1, 144(sp)
   14780: 6a 69        	ld	s2, 152(sp)
   14782: 0a 7a        	ld	s4, 160(sp)
   14784: 06 6c        	ld	s8, 64(sp)
   14786: ba 8b        	mv	s7, a4
   14788: e3 fa 6a 93  	bgeu	s5, s6, 0x140bc <_vfprintf_r+0xbce>
   1478c: 56 8b        	mv	s6, s5
   1478e: 3d b2        	j	0x140bc <_vfprintf_r+0xbce>
   14790: b7 25 02 00  	lui	a1, 34
   14794: 83 27 81 10  	lw	a5, 264(sp)
   14798: 93 85 05 e3  	addi	a1, a1, -464
   1479c: 63 57 2d 05  	bge	s10, s2, 0x147ea <_vfprintf_r+0x12fc>
   147a0: 26 f8        	sd	s1, 48(sp)
   147a2: ca 84        	mv	s1, s2
   147a4: 6e 89        	mv	s2, s11
   147a6: 21 a0        	j	0x147ae <_vfprintf_r+0x12c0>
   147a8: c1 34        	addiw	s1, s1, -16
   147aa: 63 5d 9d 02  	bge	s10, s1, 0x147e4 <_vfprintf_r+0x12f6>
   147ae: 85 27        	addiw	a5, a5, 1
   147b0: 41 07        	addi	a4, a4, 16
   147b2: 23 30 b4 01  	sd	s11, 0(s0)
   147b6: 23 34 a4 01  	sd	s10, 8(s0)
   147ba: 3a ea        	sd	a4, 272(sp)
   147bc: 23 24 f1 10  	sw	a5, 264(sp)
   147c0: 41 04        	addi	s0, s0, 16
   147c2: e3 53 f3 fe  	bge	t1, a5, 0x147a8 <_vfprintf_r+0x12ba>
   147c6: 10 02        	addi	a2, sp, 256
   147c8: e6 85        	mv	a1, s9
   147ca: 4e 85        	mv	a0, s3
   147cc: ef 70 e0 22  	jal	0x1b9fa <__sprint_r>
   147d0: 63 17 05 f2  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   147d4: c1 34        	addiw	s1, s1, -16
   147d6: 52 67        	ld	a4, 272(sp)
   147d8: 83 27 81 10  	lw	a5, 264(sp)
   147dc: 00 03        	addi	s0, sp, 384
   147de: 1d 43        	addi	t1, zero, 7
   147e0: e3 47 9d fc  	blt	s10, s1, 0x147ae <_vfprintf_r+0x12c0>
   147e4: ca 85        	mv	a1, s2
   147e6: 26 89        	mv	s2, s1
   147e8: c2 74        	ld	s1, 48(sp)
   147ea: 9b 86 17 00  	addiw	a3, a5, 1
   147ee: 4a 97        	add	a4, a4, s2
   147f0: 0c e0        	sd	a1, 0(s0)
   147f2: 23 34 24 01  	sd	s2, 8(s0)
   147f6: 3a ea        	sd	a4, 272(sp)
   147f8: 23 24 d1 10  	sw	a3, 264(sp)
   147fc: 63 4a d3 56  	blt	t1, a3, 0x14d70 <_vfprintf_r+0x1882>
   14800: 03 46 0a 00  	lbu	a2, 0(s4)
   14804: 41 04        	addi	s0, s0, 16
   14806: 32 9b        	add	s6, s6, a2
   14808: bd b7        	j	0x14776 <_vfprintf_r+0x1288>
   1480a: 7d 1a        	addi	s4, s4, -1
   1480c: fd 34        	addiw	s1, s1, -1
   1480e: cd bd        	j	0x14700 <_vfprintf_r+0x1212>
   14810: 10 02        	addi	a2, sp, 256
   14812: e6 85        	mv	a1, s9
   14814: 4e 85        	mv	a0, s3
   14816: ef 70 40 1e  	jal	0x1b9fa <__sprint_r>
   1481a: 63 12 05 ee  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   1481e: 52 67        	ld	a4, 272(sp)
   14820: 00 03        	addi	s0, sp, 384
   14822: 1d 43        	addi	t1, zero, 7
   14824: ed bd        	j	0x1471e <_vfprintf_r+0x1230>
   14826: 10 02        	addi	a2, sp, 256
   14828: e6 85        	mv	a1, s9
   1482a: 4e 85        	mv	a0, s3
   1482c: ef 70 e0 1c  	jal	0x1b9fa <__sprint_r>
   14830: 63 17 05 ec  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14834: 03 46 0a 00  	lbu	a2, 0(s4)
   14838: 52 67        	ld	a4, 272(sp)
   1483a: 00 03        	addi	s0, sp, 384
   1483c: 9b 06 06 00  	sext.w	a3, a2
   14840: 1d 43        	addi	t1, zero, 7
   14842: 05 b7        	j	0x14762 <_vfprintf_r+0x1274>
   14844: 3e fc        	sd	a5, 56(sp)
   14846: aa 67        	ld	a5, 136(sp)
   14848: e6 75        	ld	a1, 120(sp)
   1484a: 1a f8        	sd	t1, 48(sp)
   1484c: 33 0b fb 40  	sub	s6, s6, a5
   14850: 3e 86        	mv	a2, a5
   14852: 5a 85        	mv	a0, s6
   14854: 42 ec        	sd	a6, 24(sp)
   14856: ef 50 80 4f  	jal	0x19d4e <strncpy>
   1485a: 62 68        	ld	a6, 24(sp)
   1485c: 29 47        	addi	a4, zero, 10
   1485e: 33 d7 eb 02  	divu	a4, s7, a4
   14862: 83 46 18 00  	lbu	a3, 1(a6)
   14866: 42 73        	ld	t1, 48(sp)
   14868: e2 77        	ld	a5, 56(sp)
   1486a: b3 36 d0 00  	snez	a3, a3
   1486e: 36 98        	add	a6, a6, a3
   14870: 81 46        	mv	a3, zero
   14872: 99 b9        	j	0x144c8 <_vfprintf_r+0xfda>
   14874: 82 65        	ld	a1, 0(sp)
   14876: 10 02        	addi	a2, sp, 256
   14878: 4e 85        	mv	a0, s3
   1487a: ef 70 00 18  	jal	0x1b9fa <__sprint_r>
   1487e: 63 10 05 e8  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14882: 6e 46        	lw	a2, 216(sp)
   14884: 52 67        	ld	a4, 272(sp)
   14886: 00 03        	addi	s0, sp, 384
   14888: 45 b1        	j	0x14528 <_vfprintf_r+0x103a>
   1488a: b7 27 02 00  	lui	a5, 34
   1488e: 13 8b 07 bd  	addi	s6, a5, -1072
   14892: 25 bb        	j	0x145ca <_vfprintf_r+0x10dc>
   14894: 1a 89        	mv	s2, t1
   14896: 89 46        	addi	a3, zero, 2
   14898: 4a 87        	mv	a4, s2
   1489a: 93 08 81 0e  	addi	a7, sp, 232
   1489e: 13 08 c1 0d  	addi	a6, sp, 220
   148a2: bc 09        	addi	a5, sp, 216
   148a4: e2 85        	mv	a1, s8
   148a6: 66 86        	mv	a2, s9
   148a8: 4e 85        	mv	a0, s3
   148aa: 1a f8        	sd	t1, 48(sp)
   148ac: ef 20 b0 08  	jal	0x17136 <_ldtoa_r>
   148b0: 13 07 70 06  	addi	a4, zero, 103
   148b4: 42 73        	ld	t1, 48(sp)
   148b6: 2a 8b        	mv	s6, a0
   148b8: 63 0a ed 48  	beq	s10, a4, 0x14d4c <_vfprintf_r+0x185e>
   148bc: 13 07 70 04  	addi	a4, zero, 71
   148c0: 63 06 ed 48  	beq	s10, a4, 0x14d4c <_vfprintf_r+0x185e>
   148c4: 93 7d fd fd  	andi	s11, s10, -33
   148c8: 13 07 60 04  	addi	a4, zero, 70
   148cc: b3 0a 25 01  	add	s5, a0, s2
   148d0: 63 85 ed 4e  	beq	s11, a4, 0x14dba <_vfprintf_r+0x18cc>
   148d4: 62 85        	mv	a0, s8
   148d6: 01 46        	mv	a2, zero
   148d8: 81 46        	mv	a3, zero
   148da: e6 85        	mv	a1, s9
   148dc: 1a f8        	sd	t1, 48(sp)
   148de: ef a0 20 24  	jal	0x1eb20 <__netf2>
   148e2: 42 73        	ld	t1, 48(sp)
   148e4: 56 8c        	mv	s8, s5
   148e6: 11 cd        	beqz	a0, 0x14902 <_vfprintf_r+0x1414>
   148e8: 2e 7c        	ld	s8, 232(sp)
   148ea: 63 7c 5c 01  	bgeu	s8, s5, 0x14902 <_vfprintf_r+0x1414>
   148ee: 93 06 00 03  	addi	a3, zero, 48
   148f2: 93 07 1c 00  	addi	a5, s8, 1
   148f6: be f5        	sd	a5, 232(sp)
   148f8: 23 00 dc 00  	sb	a3, 0(s8)
   148fc: 2e 7c        	ld	s8, 232(sp)
   148fe: e3 6a 5c ff  	bltu	s8, s5, 0x148f2 <_vfprintf_r+0x1404>
   14902: bb 07 6c 41  	subw	a5, s8, s6
   14906: 3e fc        	sd	a5, 56(sp)
   14908: 6e 46        	lw	a2, 216(sp)
   1490a: 13 07 70 04  	addi	a4, zero, 71
   1490e: 32 f8        	sd	a2, 48(sp)
   14910: 63 81 ed 3a  	beq	s11, a4, 0x14cb2 <_vfprintf_r+0x17c4>
   14914: 13 07 60 04  	addi	a4, zero, 70
   14918: 63 8f ed 50  	beq	s11, a4, 0x14e36 <_vfprintf_r+0x1948>
   1491c: 1b 07 f6 ff  	addiw	a4, a2, -1
   14920: ba cd        	sw	a4, 216(sp)
   14922: 93 06 10 06  	addi	a3, zero, 97
   14926: 63 02 dd 70  	beq	s10, a3, 0x1502a <_vfprintf_r+0x1b3c>
   1492a: 93 05 10 04  	addi	a1, zero, 65
   1492e: 93 06 00 05  	addi	a3, zero, 80
   14932: 63 1a bd 38  	bne	s10, a1, 0x14cc6 <_vfprintf_r+0x17d8>
   14936: 85 45        	addi	a1, zero, 1
   14938: 23 00 d1 0e  	sb	a3, 224(sp)
   1493c: 93 06 b0 02  	addi	a3, zero, 43
   14940: 63 56 07 00  	bgez	a4, 0x1494c <_vfprintf_r+0x145e>
   14944: 05 47        	addi	a4, zero, 1
   14946: 11 9f        	subw	a4, a4, a2
   14948: 93 06 d0 02  	addi	a3, zero, 45
   1494c: a3 00 d1 0e  	sb	a3, 225(sp)
   14950: a5 46        	addi	a3, zero, 9
   14952: 63 d0 e6 5a  	bge	a3, a4, 0x14ef2 <_vfprintf_r+0x1a04>
   14956: 13 08 f1 0e  	addi	a6, sp, 239
   1495a: c2 88        	mv	a7, a6
   1495c: 29 43        	addi	t1, zero, 10
   1495e: 13 0f 30 06  	addi	t5, zero, 99
   14962: bb 67 67 02  	remw	a5, a4, t1
   14966: 46 86        	mv	a2, a7
   14968: ba 86        	mv	a3, a4
   1496a: fd 18        	addi	a7, a7, -1
   1496c: 9b 87 07 03  	addiw	a5, a5, 48
   14970: a3 0f f6 fe  	sb	a5, -1(a2)
   14974: 3b 47 67 02  	divw	a4, a4, t1
   14978: e3 45 df fe  	blt	t5, a3, 0x14962 <_vfprintf_r+0x1474>
   1497c: 1b 05 07 03  	addiw	a0, a4, 48
   14980: a3 8f a8 fe  	sb	a0, -1(a7)
   14984: 13 07 e6 ff  	addi	a4, a2, -2
   14988: 63 7c 07 6d  	bgeu	a4, a6, 0x15060 <_vfprintf_r+0x1b72>
   1498c: 93 06 21 0e  	addi	a3, sp, 226
   14990: 83 47 07 00  	lbu	a5, 0(a4)
   14994: 05 07        	addi	a4, a4, 1
   14996: 85 06        	addi	a3, a3, 1
   14998: a3 8f f6 fe  	sb	a5, -1(a3)
   1499c: e3 1a 07 ff  	bne	a4, a6, 0x14990 <_vfprintf_r+0x14a2>
   149a0: 13 07 11 0f  	addi	a4, sp, 241
   149a4: 93 07 21 0e  	addi	a5, sp, 226
   149a8: 11 8f        	sub	a4, a4, a2
   149aa: 3e 97        	add	a4, a4, a5
   149ac: 94 11        	addi	a3, sp, 224
   149ae: bb 07 d7 40  	subw	a5, a4, a3
   149b2: be f8        	sd	a5, 112(sp)
   149b4: 46 77        	ld	a4, 112(sp)
   149b6: e2 77        	ld	a5, 56(sp)
   149b8: 05 46        	addi	a2, zero, 1
   149ba: bb 8d e7 00  	addw	s11, a5, a4
   149be: 6e 87        	mv	a4, s11
   149c0: 63 53 f6 5e  	bge	a2, a5, 0x14fa6 <_vfprintf_r+0x1ab8>
   149c4: e6 67        	ld	a5, 88(sp)
   149c6: bb 0d f7 00  	addw	s11, a4, a5
   149ca: 86 67        	ld	a5, 64(sp)
   149cc: 13 c8 fd ff  	not	a6, s11
   149d0: 13 58 f8 43  	srai	a6, a6, 63
   149d4: 13 f9 f7 bf  	andi	s2, a5, -1025
   149d8: 01 29        	sext.w	s2, s2
   149da: 33 f8 0d 01  	and	a6, s11, a6
   149de: 13 69 09 10  	ori	s2, s2, 256
   149e2: 1b 0c 08 00  	sext.w	s8, a6
   149e6: 82 e4        	sd	zero, 72(sp)
   149e8: 82 e0        	sd	zero, 64(sp)
   149ea: 02 f8        	sd	zero, 48(sp)
   149ec: ea 67        	ld	a5, 152(sp)
   149ee: 63 88 07 30  	beqz	a5, 0x14cfe <_vfprintf_r+0x1810>
   149f2: 13 07 d0 02  	addi	a4, zero, 45
   149f6: a3 07 e1 0c  	sb	a4, 207(sp)
   149fa: 01 43        	mv	t1, zero
   149fc: 05 2c        	addiw	s8, s8, 1
   149fe: 6f e0 9f ca  	j	0x136a6 <_vfprintf_r+0x1b8>
   14a02: 82 65        	ld	a1, 0(sp)
   14a04: 10 02        	addi	a2, sp, 256
   14a06: 4e 85        	mv	a0, s3
   14a08: ef 60 30 7f  	jal	0x1b9fa <__sprint_r>
   14a0c: 63 19 05 ce  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14a10: d2 6b        	ld	s7, 272(sp)
   14a12: 00 03        	addi	s0, sp, 384
   14a14: 6f f0 8f e8  	j	0x1409c <_vfprintf_r+0xbae>
   14a18: 99 46        	addi	a3, zero, 6
   14a1a: c5 b2        	j	0x143fa <_vfprintf_r+0xf0c>
   14a1c: b3 8c 6a 41  	sub	s9, s5, s6
   14a20: 3b 86 d7 40  	subw	a2, a5, a3
   14a24: 9b 87 0c 00  	sext.w	a5, s9
   14a28: 5e 87        	mv	a4, s7
   14a2a: 63 53 f6 00  	bge	a2, a5, 0x14a30 <_vfprintf_r+0x1542>
   14a2e: b2 8c        	mv	s9, a2
   14a30: 81 2c        	sext.w	s9, s9
   14a32: 6f f0 8f ef  	j	0x1412a <_vfprintf_r+0xc3c>
   14a36: 82 1b        	slli	s7, s7, 32
   14a38: 93 db 0b 02  	srli	s7, s7, 32
   14a3c: 35 ba        	j	0x14378 <_vfprintf_r+0xe8a>
   14a3e: c2 67        	ld	a5, 16(sp)
   14a40: 23 10 f7 00  	sh	a5, 0(a4)
   14a44: 6f f0 ef 9c  	j	0x13c12 <_vfprintf_r+0x724>
   14a48: 82 1b        	slli	s7, s7, 32
   14a4a: 93 db 0b 02  	srli	s7, s7, 32
   14a4e: 3a f0        	sd	a4, 32(sp)
   14a50: 85 47        	addi	a5, zero, 1
   14a52: 6f f0 8f 8e  	j	0x13b3a <_vfprintf_r+0x64c>
   14a56: de 87        	mv	a5, s7
   14a58: 6f f0 0f 95  	j	0x13ba8 <_vfprintf_r+0x6ba>
   14a5c: 82 1b        	slli	s7, s7, 32
   14a5e: 93 db 0b 02  	srli	s7, s7, 32
   14a62: 6f f0 8f 98  	j	0x13bea <_vfprintf_r+0x6fc>
   14a66: 82 65        	ld	a1, 0(sp)
   14a68: 46 65        	ld	a0, 80(sp)
   14a6a: 10 02        	addi	a2, sp, 256
   14a6c: ef 60 f0 78  	jal	0x1b9fa <__sprint_r>
   14a70: 6f e0 3f d2  	j	0x13792 <_vfprintf_r+0x2a4>
   14a74: 93 07 d0 02  	addi	a5, zero, 45
   14a78: a3 07 f1 0c  	sb	a5, 207(sp)
   14a7c: 13 07 d0 02  	addi	a4, zero, 45
   14a80: 1d b1        	j	0x146a6 <_vfprintf_r+0x11b8>
   14a82: 82 65        	ld	a1, 0(sp)
   14a84: 10 02        	addi	a2, sp, 256
   14a86: 4e 85        	mv	a0, s3
   14a88: ef 60 30 77  	jal	0x1b9fa <__sprint_r>
   14a8c: 63 19 05 c6  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14a90: 6e 46        	lw	a2, 216(sp)
   14a92: 52 67        	ld	a4, 272(sp)
   14a94: 83 26 81 10  	lw	a3, 264(sp)
   14a98: 00 03        	addi	s0, sp, 384
   14a9a: 85 be        	j	0x1460a <_vfprintf_r+0x111c>
   14a9c: 13 07 80 05  	addi	a4, zero, 88
   14aa0: 93 67 29 00  	ori	a5, s2, 2
   14aa4: 93 06 00 03  	addi	a3, zero, 48
   14aa8: 81 27        	sext.w	a5, a5
   14aaa: a3 08 e1 0c  	sb	a4, 209(sp)
   14aae: 23 08 d1 0c  	sb	a3, 208(sp)
   14ab2: 13 07 30 06  	addi	a4, zero, 99
   14ab6: be e0        	sd	a5, 64(sp)
   14ab8: 02 ec        	sd	zero, 24(sp)
   14aba: 13 0b 81 11  	addi	s6, sp, 280
   14abe: 63 48 67 48  	blt	a4, t1, 0x14f4e <_vfprintf_r+0x1a60>
   14ac2: 13 69 29 10  	ori	s2, s2, 258
   14ac6: ee 7c        	ld	s9, 248(sp)
   14ac8: 9b 07 09 00  	sext.w	a5, s2
   14acc: 3e e9        	sd	a5, 144(sp)
   14ace: 02 ed        	sd	zero, 152(sp)
   14ad0: 4e 7c        	ld	s8, 240(sp)
   14ad2: 63 c7 0c 28  	bltz	s9, 0x14d60 <_vfprintf_r+0x1872>
   14ad6: 13 07 10 06  	addi	a4, zero, 97
   14ada: 63 04 ed 00  	beq	s10, a4, 0x14ae2 <_vfprintf_r+0x15f4>
   14ade: 6f e0 9f de  	j	0x138c6 <_vfprintf_r+0x3d8>
   14ae2: e6 85        	mv	a1, s9
   14ae4: 62 85        	mv	a0, s8
   14ae6: 1a f8        	sd	t1, 48(sp)
   14ae8: ef b0 60 35  	jal	0x1fe3e <__trunctfdf2>
   14aec: a8 09        	addi	a0, sp, 216
   14aee: ef 40 b0 65  	jal	0x19948 <frexp>
   14af2: ef b0 80 28  	jal	0x1fd7a <__extenddftf2>
   14af6: 37 27 02 00  	lui	a4, 34
   14afa: 83 36 87 e5  	ld	a3, -424(a4)
   14afe: 01 46        	mv	a2, zero
   14b00: ef a0 a0 23  	jal	0x1ed3a <__multf3>
   14b04: 01 46        	mv	a2, zero
   14b06: 81 46        	mv	a3, zero
   14b08: aa 8d        	mv	s11, a0
   14b0a: 2e 89        	mv	s2, a1
   14b0c: ef a0 40 01  	jal	0x1eb20 <__netf2>
   14b10: 42 73        	ld	t1, 48(sp)
   14b12: 19 e1        	bnez	a0, 0x14b18 <_vfprintf_r+0x162a>
   14b14: 05 47        	addi	a4, zero, 1
   14b16: ba cd        	sw	a4, 216(sp)
   14b18: 37 27 02 00  	lui	a4, 34
   14b1c: 93 07 87 c0  	addi	a5, a4, -1016
   14b20: 3e fc        	sd	a5, 56(sp)
   14b22: b7 26 02 00  	lui	a3, 34
   14b26: 83 b7 86 e6  	ld	a5, -408(a3)
   14b2a: 13 17 03 02  	slli	a4, t1, 32
   14b2e: 01 93        	srli	a4, a4, 32
   14b30: 05 07        	addi	a4, a4, 1
   14b32: 3e f8        	sd	a5, 48(sp)
   14b34: 1b 06 03 00  	sext.w	a2, t1
   14b38: b3 07 eb 00  	add	a5, s6, a4
   14b3c: a6 e4        	sd	s1, 72(sp)
   14b3e: 52 f1        	sd	s4, 160(sp)
   14b40: 22 f9        	sd	s0, 176(sp)
   14b42: 1a f5        	sd	t1, 168(sp)
   14b44: 62 74        	ld	s0, 56(sp)
   14b46: be 84        	mv	s1, a5
   14b48: 5a 8a        	mv	s4, s6
   14b4a: 32 fd        	sd	a2, 184(sp)
   14b4c: ce 8b        	mv	s7, s3
   14b4e: da 8c        	mv	s9, s6
   14b50: 39 a0        	j	0x14b5e <_vfprintf_r+0x1670>
   14b52: 01 46        	mv	a2, zero
   14b54: 81 46        	mv	a3, zero
   14b56: ef 90 b0 7c  	jal	0x1eb20 <__netf2>
   14b5a: 63 09 05 40  	beqz	a0, 0x14f6c <_vfprintf_r+0x1a7e>
   14b5e: c2 76        	ld	a3, 48(sp)
   14b60: 01 46        	mv	a2, zero
   14b62: 6e 85        	mv	a0, s11
   14b64: ca 85        	mv	a1, s2
   14b66: ef a0 40 1d  	jal	0x1ed3a <__multf3>
   14b6a: 2e 89        	mv	s2, a1
   14b6c: aa 89        	mv	s3, a0
   14b6e: ef b0 60 12  	jal	0x1fc94 <__fixtfsi>
   14b72: 1b 0c 05 00  	sext.w	s8, a0
   14b76: 62 85        	mv	a0, s8
   14b78: ef b0 c0 1a  	jal	0x1fd24 <__floatsitf>
   14b7c: ae 86        	mv	a3, a1
   14b7e: 2a 86        	mv	a2, a0
   14b80: ca 85        	mv	a1, s2
   14b82: 4e 85        	mv	a0, s3
   14b84: ef a0 80 7d  	jal	0x1f35c <__subtf3>
   14b88: 33 07 84 01  	add	a4, s0, s8
   14b8c: 83 46 07 00  	lbu	a3, 0(a4)
   14b90: d2 89        	mv	s3, s4
   14b92: 05 0a        	addi	s4, s4, 1
   14b94: a3 0f da fe  	sb	a3, -1(s4)
   14b98: aa 8a        	mv	s5, a0
   14b9a: aa 8d        	mv	s11, a0
   14b9c: 2e 89        	mv	s2, a1
   14b9e: e3 9a 44 fb  	bne	s1, s4, 0x14b52 <_vfprintf_r+0x1664>
   14ba2: d2 8a        	mv	s5, s4
   14ba4: a6 64        	ld	s1, 72(sp)
   14ba6: 2a 73        	ld	t1, 168(sp)
   14ba8: 4a 74        	ld	s0, 176(sp)
   14baa: 0a 7a        	ld	s4, 160(sp)
   14bac: 4e f8        	sd	s3, 48(sp)
   14bae: 66 8b        	mv	s6, s9
   14bb0: de 89        	mv	s3, s7
   14bb2: ae 8c        	mv	s9, a1
   14bb4: aa 8b        	mv	s7, a0
   14bb6: fd 5d        	addi	s11, zero, -1
   14bb8: b7 26 02 00  	lui	a3, 34
   14bbc: 03 b9 86 e7  	ld	s2, -392(a3)
   14bc0: 01 46        	mv	a2, zero
   14bc2: 5e 85        	mv	a0, s7
   14bc4: ca 86        	mv	a3, s2
   14bc6: e6 85        	mv	a1, s9
   14bc8: 9a e4        	sd	t1, 72(sp)
   14bca: ef 90 10 7f  	jal	0x1ebba <__gttf2>
   14bce: 26 63        	ld	t1, 72(sp)
   14bd0: 63 4f a0 00  	bgtz	a0, 0x14bee <_vfprintf_r+0x1700>
   14bd4: 01 46        	mv	a2, zero
   14bd6: ca 86        	mv	a3, s2
   14bd8: 5e 85        	mv	a0, s7
   14bda: e6 85        	mv	a1, s9
   14bdc: ef 90 50 74  	jal	0x1eb20 <__netf2>
   14be0: 26 63        	ld	t1, 72(sp)
   14be2: 63 16 05 0e  	bnez	a0, 0x14cce <_vfprintf_r+0x17e0>
   14be6: 13 7c 1c 00  	andi	s8, s8, 1
   14bea: 63 02 0c 0e  	beqz	s8, 0x14cce <_vfprintf_r+0x17e0>
   14bee: c2 77        	ld	a5, 48(sp)
   14bf0: d6 86        	mv	a3, s5
   14bf2: be f5        	sd	a5, 232(sp)
   14bf4: e2 77        	ld	a5, 56(sp)
   14bf6: 03 c6 fa ff  	lbu	a2, -1(s5)
   14bfa: 83 c5 f7 00  	lbu	a1, 15(a5)
   14bfe: 63 9e c5 00  	bne	a1, a2, 0x14c1a <_vfprintf_r+0x172c>
   14c02: 13 05 00 03  	addi	a0, zero, 48
   14c06: a3 8f a6 fe  	sb	a0, -1(a3)
   14c0a: ae 76        	ld	a3, 232(sp)
   14c0c: 93 87 f6 ff  	addi	a5, a3, -1
   14c10: be f5        	sd	a5, 232(sp)
   14c12: 03 c6 f6 ff  	lbu	a2, -1(a3)
   14c16: e3 88 c5 fe  	beq	a1, a2, 0x14c06 <_vfprintf_r+0x1718>
   14c1a: 9b 05 16 00  	addiw	a1, a2, 1
   14c1e: 13 05 90 03  	addi	a0, zero, 57
   14c22: 93 f5 f5 0f  	andi	a1, a1, 255
   14c26: 63 0c a6 02  	beq	a2, a0, 0x14c5e <_vfprintf_r+0x1770>
   14c2a: a3 8f b6 fe  	sb	a1, -1(a3)
   14c2e: bb 87 6a 41  	subw	a5, s5, s6
   14c32: 3e fc        	sd	a5, 56(sp)
   14c34: 93 7d fd fd  	andi	s11, s10, -33
   14c38: c1 b9        	j	0x14908 <_vfprintf_r+0x141a>
   14c3a: 82 65        	ld	a1, 0(sp)
   14c3c: 10 02        	addi	a2, sp, 256
   14c3e: 4e 85        	mv	a0, s3
   14c40: ef 60 b0 5b  	jal	0x1b9fa <__sprint_r>
   14c44: 63 1d 05 aa  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14c48: d2 6b        	ld	s7, 272(sp)
   14c4a: 00 03        	addi	s0, sp, 384
   14c4c: 6f f0 4f c6  	j	0x140b0 <_vfprintf_r+0xbc2>
   14c50: 1b 09 13 00  	addiw	s2, t1, 1
   14c54: 89 46        	addi	a3, zero, 2
   14c56: 89 b1        	j	0x14898 <_vfprintf_r+0x13aa>
   14c58: 1a 89        	mv	s2, t1
   14c5a: 8d 46        	addi	a3, zero, 3
   14c5c: 35 b9        	j	0x14898 <_vfprintf_r+0x13aa>
   14c5e: e2 77        	ld	a5, 56(sp)
   14c60: 83 c5 a7 00  	lbu	a1, 10(a5)
   14c64: d9 b7        	j	0x14c2a <_vfprintf_r+0x173c>
   14c66: e6 85        	mv	a1, s9
   14c68: 62 85        	mv	a0, s8
   14c6a: 1a f8        	sd	t1, 48(sp)
   14c6c: ef b0 20 1d  	jal	0x1fe3e <__trunctfdf2>
   14c70: a8 09        	addi	a0, sp, 216
   14c72: ef 40 70 4d  	jal	0x19948 <frexp>
   14c76: ef b0 40 10  	jal	0x1fd7a <__extenddftf2>
   14c7a: 37 27 02 00  	lui	a4, 34
   14c7e: 83 36 87 e5  	ld	a3, -424(a4)
   14c82: 01 46        	mv	a2, zero
   14c84: ef a0 60 0b  	jal	0x1ed3a <__multf3>
   14c88: 01 46        	mv	a2, zero
   14c8a: 81 46        	mv	a3, zero
   14c8c: aa 8d        	mv	s11, a0
   14c8e: 2e 89        	mv	s2, a1
   14c90: ef 90 10 69  	jal	0x1eb20 <__netf2>
   14c94: 42 73        	ld	t1, 48(sp)
   14c96: 19 e1        	bnez	a0, 0x14c9c <_vfprintf_r+0x17ae>
   14c98: 05 47        	addi	a4, zero, 1
   14c9a: ba cd        	sw	a4, 216(sp)
   14c9c: 37 27 02 00  	lui	a4, 34
   14ca0: 93 07 07 bf  	addi	a5, a4, -1040
   14ca4: 3e fc        	sd	a5, 56(sp)
   14ca6: b5 bd        	j	0x14b22 <_vfprintf_r+0x1634>
   14ca8: 2e 7c        	ld	s8, 232(sp)
   14caa: 6e 46        	lw	a2, 216(sp)
   14cac: bb 07 6c 41  	subw	a5, s8, s6
   14cb0: 3e fc        	sd	a5, 56(sp)
   14cb2: 75 57        	addi	a4, zero, -3
   14cb4: 32 f8        	sd	a2, 48(sp)
   14cb6: 63 44 e6 00  	blt	a2, a4, 0x14cbe <_vfprintf_r+0x17d0>
   14cba: 63 5a c3 04  	bge	t1, a2, 0x14d0e <_vfprintf_r+0x1820>
   14cbe: 1b 07 f6 ff  	addiw	a4, a2, -1
   14cc2: 79 3d        	addiw	s10, s10, -2
   14cc4: ba cd        	sw	a4, 216(sp)
   14cc6: 93 76 fd 0f  	andi	a3, s10, 255
   14cca: 81 45        	mv	a1, zero
   14ccc: b5 b1        	j	0x14938 <_vfprintf_r+0x144a>
   14cce: e3 c0 0d f6  	bltz	s11, 0x14c2e <_vfprintf_r+0x1740>
   14cd2: 85 2d        	addiw	s11, s11, 1
   14cd4: d7 75 00 04  	<unknown>
   14cd8: 93 96 0d 02  	slli	a3, s11, 32
   14cdc: 93 05 00 03  	addi	a1, zero, 48
   14ce0: 81 92        	srli	a3, a3, 32
   14ce2: 56 86        	mv	a2, s5
   14ce4: 73 25 20 c2  	csrr	a0, vlenb
   14ce8: 57 cc 05 5e  	<unknown>
   14cec: d7 f7 06 04  	<unknown>
   14cf0: 27 0c 06 02  	<unknown>
   14cf4: 9d 8e        	sub	a3, a3, a5
   14cf6: 2a 96        	add	a2, a2, a0
   14cf8: f5 fa        	bnez	a3, 0x14cec <_vfprintf_r+0x17fe>
   14cfa: ee 9a        	add	s5, s5, s11
   14cfc: 0d bf        	j	0x14c2e <_vfprintf_r+0x1740>
   14cfe: 03 47 f1 0c  	lbu	a4, 207(sp)
   14d02: 01 43        	mv	t1, zero
   14d04: 19 c3        	beqz	a4, 0x14d0a <_vfprintf_r+0x181c>
   14d06: 6f e0 9f d3  	j	0x13a3e <_vfprintf_r+0x550>
   14d0a: 6f e0 df 99  	j	0x136a6 <_vfprintf_r+0x1b8>
   14d0e: e2 77        	ld	a5, 56(sp)
   14d10: 63 4b f6 1a  	blt	a2, a5, 0x14ec6 <_vfprintf_r+0x19d8>
   14d14: 86 67        	ld	a5, 64(sp)
   14d16: b2 8d        	mv	s11, a2
   14d18: 13 f7 17 00  	andi	a4, a5, 1
   14d1c: 01 c7        	beqz	a4, 0x14d24 <_vfprintf_r+0x1836>
   14d1e: e6 67        	ld	a5, 88(sp)
   14d20: bb 8d c7 00  	addw	s11, a5, a2
   14d24: 86 67        	ld	a5, 64(sp)
   14d26: 13 f7 07 40  	andi	a4, a5, 1024
   14d2a: 19 c3        	beqz	a4, 0x14d30 <_vfprintf_r+0x1842>
   14d2c: 63 43 c0 28  	bgtz	a2, 0x14fb2 <_vfprintf_r+0x1ac4>
   14d30: 13 c8 fd ff  	not	a6, s11
   14d34: 13 58 f8 43  	srai	a6, a6, 63
   14d38: 4a 69        	ld	s2, 144(sp)
   14d3a: 33 f8 0d 01  	and	a6, s11, a6
   14d3e: 1b 0c 08 00  	sext.w	s8, a6
   14d42: 13 0d 70 06  	addi	s10, zero, 103
   14d46: 82 e4        	sd	zero, 72(sp)
   14d48: 82 e0        	sd	zero, 64(sp)
   14d4a: 4d b1        	j	0x149ec <_vfprintf_r+0x14fe>
   14d4c: 86 67        	ld	a5, 64(sp)
   14d4e: 13 f7 17 00  	andi	a4, a5, 1
   14d52: 39 db        	beqz	a4, 0x14ca8 <_vfprintf_r+0x17ba>
   14d54: b3 0a 2b 01  	add	s5, s6, s2
   14d58: 93 0d 70 04  	addi	s11, zero, 71
   14d5c: a5 be        	j	0x148d4 <_vfprintf_r+0x13e6>
   14d5e: 02 ec        	sd	zero, 24(sp)
   14d60: 7d 57        	addi	a4, zero, -1
   14d62: 7e 17        	slli	a4, a4, 63
   14d64: 93 07 d0 02  	addi	a5, zero, 45
   14d68: b3 cc ec 00  	xor	s9, s9, a4
   14d6c: 3e ed        	sd	a5, 152(sp)
   14d6e: a5 b3        	j	0x14ad6 <_vfprintf_r+0x15e8>
   14d70: 10 02        	addi	a2, sp, 256
   14d72: e6 85        	mv	a1, s9
   14d74: 4e 85        	mv	a0, s3
   14d76: ef 60 50 48  	jal	0x1b9fa <__sprint_r>
   14d7a: 63 12 05 98  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14d7e: 03 46 0a 00  	lbu	a2, 0(s4)
   14d82: 52 67        	ld	a4, 272(sp)
   14d84: 00 03        	addi	s0, sp, 384
   14d86: 1d 43        	addi	t1, zero, 7
   14d88: 32 9b        	add	s6, s6, a2
   14d8a: f5 b2        	j	0x14776 <_vfprintf_r+0x1288>
   14d8c: 13 07 80 07  	addi	a4, zero, 120
   14d90: 01 bb        	j	0x14aa0 <_vfprintf_r+0x15b2>
   14d92: 93 07 d0 02  	addi	a5, zero, 45
   14d96: a3 07 f1 0c  	sb	a5, 207(sp)
   14d9a: 13 07 d0 02  	addi	a4, zero, 45
   14d9e: 6f f0 df 81  	j	0x145ba <_vfprintf_r+0x10cc>
   14da2: 82 65        	ld	a1, 0(sp)
   14da4: 10 02        	addi	a2, sp, 256
   14da6: 4e 85        	mv	a0, s3
   14da8: ef 60 30 45  	jal	0x1b9fa <__sprint_r>
   14dac: 63 19 05 94  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14db0: ee 46        	lw	a3, 216(sp)
   14db2: 52 67        	ld	a4, 272(sp)
   14db4: 00 03        	addi	s0, sp, 384
   14db6: 6f f0 cf b3  	j	0x140f2 <_vfprintf_r+0xc04>
   14dba: 83 46 05 00  	lbu	a3, 0(a0)
   14dbe: 13 07 00 03  	addi	a4, zero, 48
   14dc2: 63 85 e6 00  	beq	a3, a4, 0x14dcc <_vfprintf_r+0x18de>
   14dc6: ee 46        	lw	a3, 216(sp)
   14dc8: b6 9a        	add	s5, s5, a3
   14dca: 29 b6        	j	0x148d4 <_vfprintf_r+0x13e6>
   14dcc: 01 46        	mv	a2, zero
   14dce: 81 46        	mv	a3, zero
   14dd0: 62 85        	mv	a0, s8
   14dd2: e6 85        	mv	a1, s9
   14dd4: 1a f8        	sd	t1, 48(sp)
   14dd6: ef 90 b0 54  	jal	0x1eb20 <__netf2>
   14dda: 42 73        	ld	t1, 48(sp)
   14ddc: 6d d5        	beqz	a0, 0x14dc6 <_vfprintf_r+0x18d8>
   14dde: 05 47        	addi	a4, zero, 1
   14de0: bb 06 27 41  	subw	a3, a4, s2
   14de4: b6 cd        	sw	a3, 216(sp)
   14de6: b6 9a        	add	s5, s5, a3
   14de8: f5 b4        	j	0x148d4 <_vfprintf_r+0x13e6>
   14dea: 03 47 f1 0c  	lbu	a4, 207(sp)
   14dee: 56 f0        	sd	s5, 32(sp)
   14df0: 82 e4        	sd	zero, 72(sp)
   14df2: 82 e0        	sd	zero, 64(sp)
   14df4: 02 f8        	sd	zero, 48(sp)
   14df6: 1a 8c        	mv	s8, t1
   14df8: 9a 8d        	mv	s11, t1
   14dfa: 01 43        	mv	t1, zero
   14dfc: 19 c3        	beqz	a4, 0x14e02 <_vfprintf_r+0x1914>
   14dfe: 6f e0 1f c4  	j	0x13a3e <_vfprintf_r+0x550>
   14e02: 6f e0 5f 8a  	j	0x136a6 <_vfprintf_r+0x1b8>
   14e06: b7 27 02 00  	lui	a5, 34
   14e0a: 13 8b 07 be  	addi	s6, a5, -1056
   14e0e: 6f f0 cf fb  	j	0x145ca <_vfprintf_r+0x10dc>
   14e12: 82 65        	ld	a1, 0(sp)
   14e14: 10 02        	addi	a2, sp, 256
   14e16: 4e 85        	mv	a0, s3
   14e18: ef 60 30 3e  	jal	0x1b9fa <__sprint_r>
   14e1c: 63 11 05 8e  	bnez	a0, 0x13efe <_vfprintf_r+0xa10>
   14e20: 6e 46        	lw	a2, 216(sp)
   14e22: e2 77        	ld	a5, 56(sp)
   14e24: 52 67        	ld	a4, 272(sp)
   14e26: 00 03        	addi	s0, sp, 384
   14e28: 3b 86 c7 40  	subw	a2, a5, a2
   14e2c: 6f f0 ef af  	j	0x1412a <_vfprintf_r+0xc3c>
   14e30: ca 8c        	mv	s9, s2
   14e32: 6f f0 2f 98  	j	0x13fb4 <_vfprintf_r+0xac6>
   14e36: 86 67        	ld	a5, 64(sp)
   14e38: 13 f7 17 00  	andi	a4, a5, 1
   14e3c: c2 77        	ld	a5, 48(sp)
   14e3e: 33 67 e3 00  	or	a4, t1, a4
   14e42: 63 57 f0 1e  	blez	a5, 0x15030 <_vfprintf_r+0x1b42>
   14e46: 63 1a 07 1c  	bnez	a4, 0x1501a <_vfprintf_r+0x1b2c>
   14e4a: c2 7d        	ld	s11, 48(sp)
   14e4c: 13 0d 60 06  	addi	s10, zero, 102
   14e50: 86 67        	ld	a5, 64(sp)
   14e52: 13 f7 07 40  	andi	a4, a5, 1024
   14e56: 63 10 07 16  	bnez	a4, 0x14fb6 <_vfprintf_r+0x1ac8>
   14e5a: 13 c8 fd ff  	not	a6, s11
   14e5e: 13 58 f8 43  	srai	a6, a6, 63
   14e62: 33 f8 0d 01  	and	a6, s11, a6
   14e66: 4a 69        	ld	s2, 144(sp)
   14e68: 1b 0c 08 00  	sext.w	s8, a6
   14e6c: e9 bd        	j	0x14d46 <_vfprintf_r+0x1858>
   14e6e: 37 2f 02 00  	lui	t5, 34
   14e72: c1 55        	addi	a1, zero, -16
   14e74: bb 0c c0 40  	negw	s9, a2
   14e78: 93 0b 0f e3  	addi	s7, t5, -464
   14e7c: 41 4d        	addi	s10, zero, 16
   14e7e: 9d 4d        	addi	s11, zero, 7
   14e80: 63 46 b6 00  	blt	a2, a1, 0x14e8c <_vfprintf_r+0x199e>
   14e84: 59 a8        	j	0x14f1a <_vfprintf_r+0x1a2c>
   14e86: c1 3c        	addiw	s9, s9, -16
   14e88: 63 59 9d 09  	bge	s10, s9, 0x14f1a <_vfprintf_r+0x1a2c>
   14e8c: 85 26        	addiw	a3, a3, 1
   14e8e: 41 07        	addi	a4, a4, 16
   14e90: 23 30 74 01  	sd	s7, 0(s0)
   14e94: 23 34 a4 01  	sd	s10, 8(s0)
   14e98: 3a ea        	sd	a4, 272(sp)
   14e9a: 23 24 d1 10  	sw	a3, 264(sp)
   14e9e: 41 04        	addi	s0, s0, 16
   14ea0: e3 d3 dd fe  	bge	s11, a3, 0x14e86 <_vfprintf_r+0x1998>
   14ea4: 82 65        	ld	a1, 0(sp)
   14ea6: 10 02        	addi	a2, sp, 256
   14ea8: 4e 85        	mv	a0, s3
   14eaa: ef 60 10 35  	jal	0x1b9fa <__sprint_r>
   14eae: 19 c1        	beqz	a0, 0x14eb4 <_vfprintf_r+0x19c6>
   14eb0: 6f f0 ef 84  	j	0x13efe <_vfprintf_r+0xa10>
   14eb4: 52 67        	ld	a4, 272(sp)
   14eb6: 83 26 81 10  	lw	a3, 264(sp)
   14eba: 00 03        	addi	s0, sp, 384
   14ebc: e9 b7        	j	0x14e86 <_vfprintf_r+0x1998>
   14ebe: fd 57        	addi	a5, zero, -1
   14ec0: 3e e8        	sd	a5, 16(sp)
   14ec2: 6f e0 1f 8e  	j	0x137a2 <_vfprintf_r+0x2b4>
   14ec6: e6 67        	ld	a5, 88(sp)
   14ec8: 62 77        	ld	a4, 56(sp)
   14eca: 13 0d 70 06  	addi	s10, zero, 103
   14ece: bb 8d e7 00  	addw	s11, a5, a4
   14ed2: c2 77        	ld	a5, 48(sp)
   14ed4: e3 4e f0 f6  	bgtz	a5, 0x14e50 <_vfprintf_r+0x1962>
   14ed8: 3b 87 fd 40  	subw	a4, s11, a5
   14edc: 05 27        	addiw	a4, a4, 1
   14ede: 9b 0d 07 00  	sext.w	s11, a4
   14ee2: 13 c6 fd ff  	not	a2, s11
   14ee6: 7d 96        	srai	a2, a2, 63
   14ee8: 71 8f        	and	a4, a4, a2
   14eea: 4a 69        	ld	s2, 144(sp)
   14eec: 1b 0c 07 00  	sext.w	s8, a4
   14ef0: 99 bd        	j	0x14d46 <_vfprintf_r+0x1858>
   14ef2: 93 06 21 0e  	addi	a3, sp, 226
   14ef6: 99 e5        	bnez	a1, 0x14f04 <_vfprintf_r+0x1a16>
   14ef8: 93 06 00 03  	addi	a3, zero, 48
   14efc: 23 01 d1 0e  	sb	a3, 226(sp)
   14f00: 93 06 31 0e  	addi	a3, sp, 227
   14f04: 1c 04        	addi	a5, sp, 512
   14f06: 33 86 f6 40  	sub	a2, a3, a5
   14f0a: 1b 07 07 03  	addiw	a4, a4, 48
   14f0e: 9b 07 16 12  	addiw	a5, a2, 289
   14f12: 23 80 e6 00  	sb	a4, 0(a3)
   14f16: be f8        	sd	a5, 112(sp)
   14f18: 71 bc        	j	0x149b4 <_vfprintf_r+0x14c6>
   14f1a: 85 26        	addiw	a3, a3, 1
   14f1c: 66 97        	add	a4, a4, s9
   14f1e: 23 30 74 01  	sd	s7, 0(s0)
   14f22: 23 34 94 01  	sd	s9, 8(s0)
   14f26: 3a ea        	sd	a4, 272(sp)
   14f28: 23 24 d1 10  	sw	a3, 264(sp)
   14f2c: 1d 46        	addi	a2, zero, 7
   14f2e: 63 53 d6 e2  	bge	a2, a3, 0x14554 <_vfprintf_r+0x1066>
   14f32: 82 65        	ld	a1, 0(sp)
   14f34: 10 02        	addi	a2, sp, 256
   14f36: 4e 85        	mv	a0, s3
   14f38: ef 60 30 2c  	jal	0x1b9fa <__sprint_r>
   14f3c: 19 c1        	beqz	a0, 0x14f42 <_vfprintf_r+0x1a54>
   14f3e: 6f e0 1f fc  	j	0x13efe <_vfprintf_r+0xa10>
   14f42: 52 67        	ld	a4, 272(sp)
   14f44: 83 26 81 10  	lw	a3, 264(sp)
   14f48: 00 03        	addi	s0, sp, 384
   14f4a: 6f f0 4f ec  	j	0x1460e <_vfprintf_r+0x1120>
   14f4e: 9b 05 13 00  	addiw	a1, t1, 1
   14f52: 4e 85        	mv	a0, s3
   14f54: 1a ec        	sd	t1, 24(sp)
   14f56: ef 30 a0 5d  	jal	0x18530 <_malloc_r>
   14f5a: 62 63        	ld	t1, 24(sp)
   14f5c: 2a 8b        	mv	s6, a0
   14f5e: 63 06 05 12  	beqz	a0, 0x1508a <_vfprintf_r+0x1b9c>
   14f62: 2a ec        	sd	a0, 24(sp)
   14f64: b9 be        	j	0x14ac2 <_vfprintf_r+0x15d4>
   14f66: 3a f0        	sd	a4, 32(sp)
   14f68: 6f f0 cf 84  	j	0x13fb4 <_vfprintf_r+0xac6>
   14f6c: 6a 76        	ld	a2, 184(sp)
   14f6e: 66 8b        	mv	s6, s9
   14f70: 4e f8        	sd	s3, 48(sp)
   14f72: 3b 06 cb 00  	addw	a2, s6, a2
   14f76: de 89        	mv	s3, s7
   14f78: d6 8b        	mv	s7, s5
   14f7a: d2 8a        	mv	s5, s4
   14f7c: a6 64        	ld	s1, 72(sp)
   14f7e: 2a 73        	ld	t1, 168(sp)
   14f80: 4a 74        	ld	s0, 176(sp)
   14f82: 0a 7a        	ld	s4, 160(sp)
   14f84: ca 8c        	mv	s9, s2
   14f86: bb 0d 56 41  	subw	s11, a2, s5
   14f8a: 3d b1        	j	0x14bb8 <_vfprintf_r+0x16ca>
   14f8c: 1c c3        	sw	a5, 0(a4)
   14f8e: 6f e0 5f c8  	j	0x13c12 <_vfprintf_r+0x724>
   14f92: 63 04 03 00  	beqz	t1, 0x14f9a <_vfprintf_r+0x1aac>
   14f96: 6f e0 5f 91  	j	0x138aa <_vfprintf_r+0x3bc>
   14f9a: 05 43        	addi	t1, zero, 1
   14f9c: 6f e0 ff 90  	j	0x138aa <_vfprintf_r+0x3bc>
   14fa0: 19 43        	addi	t1, zero, 6
   14fa2: 6f e0 9f 90  	j	0x138aa <_vfprintf_r+0x3bc>
   14fa6: 86 67        	ld	a5, 64(sp)
   14fa8: 13 f6 17 00  	andi	a2, a5, 1
   14fac: e3 0f 06 a0  	beqz	a2, 0x149ca <_vfprintf_r+0x14dc>
   14fb0: 11 bc        	j	0x149c4 <_vfprintf_r+0x14d6>
   14fb2: 13 0d 70 06  	addi	s10, zero, 103
   14fb6: 8a 66        	ld	a3, 128(sp)
   14fb8: 13 06 f0 0f  	addi	a2, zero, 255
   14fbc: 03 c7 06 00  	lbu	a4, 0(a3)
   14fc0: 63 02 c7 0c  	beq	a4, a2, 0x15084 <_vfprintf_r+0x1b96>
   14fc4: c2 77        	ld	a5, 48(sp)
   14fc6: 01 48        	mv	a6, zero
   14fc8: 01 45        	mv	a0, zero
   14fca: 93 05 f0 0f  	addi	a1, zero, 255
   14fce: 1b 06 07 00  	sext.w	a2, a4
   14fd2: 63 5a f7 00  	bge	a4, a5, 0x14fe6 <_vfprintf_r+0x1af8>
   14fd6: 03 c7 16 00  	lbu	a4, 1(a3)
   14fda: 91 9f        	subw	a5, a5, a2
   14fdc: 1d cb        	beqz	a4, 0x15012 <_vfprintf_r+0x1b24>
   14fde: 05 25        	addiw	a0, a0, 1
   14fe0: 85 06        	addi	a3, a3, 1
   14fe2: e3 16 b7 fe  	bne	a4, a1, 0x14fce <_vfprintf_r+0x1ae0>
   14fe6: 3e f8        	sd	a5, 48(sp)
   14fe8: 36 e1        	sd	a3, 128(sp)
   14fea: aa e0        	sd	a0, 64(sp)
   14fec: c2 e4        	sd	a6, 72(sp)
   14fee: a6 67        	ld	a5, 72(sp)
   14ff0: 06 67        	ld	a4, 64(sp)
   14ff2: 4a 69        	ld	s2, 144(sp)
   14ff4: 3d 9f        	addw	a4, a4, a5
   14ff6: aa 67        	ld	a5, 136(sp)
   14ff8: 3b 07 f7 02  	mulw	a4, a4, a5
   14ffc: 3b 07 b7 01  	addw	a4, a4, s11
   15000: 9b 0d 07 00  	sext.w	s11, a4
   15004: 13 c6 fd ff  	not	a2, s11
   15008: 7d 96        	srai	a2, a2, 63
   1500a: 71 8f        	and	a4, a4, a2
   1500c: 1b 0c 07 00  	sext.w	s8, a4
   15010: f1 ba        	j	0x149ec <_vfprintf_r+0x14fe>
   15012: 03 c7 06 00  	lbu	a4, 0(a3)
   15016: 05 28        	addiw	a6, a6, 1
   15018: e9 b7        	j	0x14fe2 <_vfprintf_r+0x1af4>
   1501a: 66 67        	ld	a4, 88(sp)
   1501c: 13 0d 60 06  	addi	s10, zero, 102
   15020: bb 06 f7 00  	addw	a3, a4, a5
   15024: bb 8d 66 00  	addw	s11, a3, t1
   15028: 25 b5        	j	0x14e50 <_vfprintf_r+0x1962>
   1502a: 93 06 00 07  	addi	a3, zero, 112
   1502e: 21 b2        	j	0x14936 <_vfprintf_r+0x1448>
   15030: 19 e7        	bnez	a4, 0x1503e <_vfprintf_r+0x1b50>
   15032: 4a 69        	ld	s2, 144(sp)
   15034: 05 4c        	addi	s8, zero, 1
   15036: 13 0d 60 06  	addi	s10, zero, 102
   1503a: 85 4d        	addi	s11, zero, 1
   1503c: 29 b3        	j	0x14d46 <_vfprintf_r+0x1858>
   1503e: e6 67        	ld	a5, 88(sp)
   15040: 4a 69        	ld	s2, 144(sp)
   15042: 13 0d 60 06  	addi	s10, zero, 102
   15046: 1b 87 17 00  	addiw	a4, a5, 1
   1504a: 3b 07 67 00  	addw	a4, a4, t1
   1504e: 9b 0d 07 00  	sext.w	s11, a4
   15052: 13 c6 fd ff  	not	a2, s11
   15056: 7d 96        	srai	a2, a2, 63
   15058: 71 8f        	and	a4, a4, a2
   1505a: 1b 0c 07 00  	sext.w	s8, a4
   1505e: e5 b1        	j	0x14d46 <_vfprintf_r+0x1858>
   15060: 89 47        	addi	a5, zero, 2
   15062: be f8        	sd	a5, 112(sp)
   15064: 81 ba        	j	0x149b4 <_vfprintf_r+0x14c6>
   15066: 82 76        	ld	a3, 32(sp)
   15068: 98 42        	lw	a4, 0(a3)
   1506a: a1 06        	addi	a3, a3, 8
   1506c: 3a 83        	mv	t1, a4
   1506e: 63 53 07 00  	bgez	a4, 0x15074 <_vfprintf_r+0x1b86>
   15072: 7d 53        	addi	t1, zero, -1
   15074: 03 ce 14 00  	lbu	t3, 1(s1)
   15078: 9b 0a 03 00  	sext.w	s5, t1
   1507c: 36 f0        	sd	a3, 32(sp)
   1507e: be 84        	mv	s1, a5
   15080: 6f e0 0f db  	j	0x13630 <_vfprintf_r+0x142>
   15084: 82 e4        	sd	zero, 72(sp)
   15086: 82 e0        	sd	zero, 64(sp)
   15088: 9d b7        	j	0x14fee <_vfprintf_r+0x1b00>
   1508a: 02 67        	ld	a4, 0(sp)
   1508c: 83 57 07 01  	lhu	a5, 16(a4)
   15090: 93 e7 07 04  	ori	a5, a5, 64
   15094: 23 18 f7 00  	sh	a5, 16(a4)
   15098: 6f e0 af ef  	j	0x13792 <_vfprintf_r+0x2a4>

000000000001509c <vfprintf>:
   1509c: aa 87        	mv	a5, a0
   1509e: 03 b5 81 93  	ld	a0, -1736(gp)
   150a2: b2 86        	mv	a3, a2
   150a4: 2e 86        	mv	a2, a1
   150a6: be 85        	mv	a1, a5
   150a8: 6f e0 6f c4  	j	0x134ee <_vfprintf_r>

00000000000150ac <__sbprintf>:
   150ac: 83 d7 05 01  	lhu	a5, 16(a1)
   150b0: 03 ae c5 0a  	lw	t3, 172(a1)
   150b4: 03 d3 25 01  	lhu	t1, 18(a1)
   150b8: 83 b8 05 03  	ld	a7, 48(a1)
   150bc: 03 b8 05 04  	ld	a6, 64(a1)
   150c0: 13 01 01 b3  	addi	sp, sp, -1232
   150c4: f5 9b        	andi	a5, a5, -3
   150c6: 13 07 00 40  	addi	a4, zero, 1024
   150ca: 23 30 81 4c  	sd	s0, 1216(sp)
   150ce: 23 18 f1 00  	sh	a5, 16(sp)
   150d2: 2e 84        	mv	s0, a1
   150d4: 1c 19        	addi	a5, sp, 176
   150d6: 8a 85        	mv	a1, sp
   150d8: 23 3c 91 4a  	sd	s1, 1208(sp)
   150dc: 23 38 21 4b  	sd	s2, 1200(sp)
   150e0: 23 34 11 4c  	sd	ra, 1224(sp)
   150e4: 2a 89        	mv	s2, a0
   150e6: 72 d7        	sw	t3, 172(sp)
   150e8: 23 19 61 00  	sh	t1, 18(sp)
   150ec: 46 f8        	sd	a7, 48(sp)
   150ee: c2 e0        	sd	a6, 64(sp)
   150f0: 3e e0        	sd	a5, 0(sp)
   150f2: 3e ec        	sd	a5, 24(sp)
   150f4: 3a c6        	sw	a4, 12(sp)
   150f6: 3a d0        	sw	a4, 32(sp)
   150f8: 02 d4        	sw	zero, 40(sp)
   150fa: ef e0 4f bf  	jal	0x134ee <_vfprintf_r>
   150fe: aa 84        	mv	s1, a0
   15100: 63 59 05 02  	bgez	a0, 0x15132 <__sbprintf+0x86>
   15104: 83 57 01 01  	lhu	a5, 16(sp)
   15108: 93 f7 07 04  	andi	a5, a5, 64
   1510c: 99 c7        	beqz	a5, 0x1511a <__sbprintf+0x6e>
   1510e: 83 57 04 01  	lhu	a5, 16(s0)
   15112: 93 e7 07 04  	ori	a5, a5, 64
   15116: 23 18 f4 00  	sh	a5, 16(s0)
   1511a: 83 30 81 4c  	ld	ra, 1224(sp)
   1511e: 03 34 01 4c  	ld	s0, 1216(sp)
   15122: 03 39 01 4b  	ld	s2, 1200(sp)
   15126: 26 85        	mv	a0, s1
   15128: 83 34 81 4b  	ld	s1, 1208(sp)
   1512c: 13 01 01 4d  	addi	sp, sp, 1232
   15130: 82 80        	ret
   15132: 8a 85        	mv	a1, sp
   15134: 4a 85        	mv	a0, s2
   15136: ef 00 a0 32  	jal	0x15460 <_fflush_r>
   1513a: 69 d5        	beqz	a0, 0x15104 <__sbprintf+0x58>
   1513c: fd 54        	addi	s1, zero, -1
   1513e: d9 b7        	j	0x15104 <__sbprintf+0x58>

0000000000015140 <__swsetup_r>:
   15140: 83 b7 81 93  	ld	a5, -1736(gp)
   15144: 01 11        	addi	sp, sp, -32
   15146: 22 e8        	sd	s0, 16(sp)
   15148: 26 e4        	sd	s1, 8(sp)
   1514a: 06 ec        	sd	ra, 24(sp)
   1514c: aa 84        	mv	s1, a0
   1514e: 2e 84        	mv	s0, a1
   15150: 99 c3        	beqz	a5, 0x15156 <__swsetup_r+0x16>
   15152: b8 4b        	lw	a4, 80(a5)
   15154: 29 cb        	beqz	a4, 0x151a6 <__swsetup_r+0x66>
   15156: 83 17 04 01  	lh	a5, 16(s0)
   1515a: 13 f7 87 00  	andi	a4, a5, 8
   1515e: 21 cf        	beqz	a4, 0x151b6 <__swsetup_r+0x76>
   15160: 18 6c        	ld	a4, 24(s0)
   15162: 35 c7        	beqz	a4, 0x151ce <__swsetup_r+0x8e>
   15164: 93 f6 17 00  	andi	a3, a5, 1
   15168: 91 ce        	beqz	a3, 0x15184 <__swsetup_r+0x44>
   1516a: 14 50        	lw	a3, 32(s0)
   1516c: 23 26 04 00  	sw	zero, 12(s0)
   15170: 01 45        	mv	a0, zero
   15172: bb 06 d0 40  	negw	a3, a3
   15176: 14 d4        	sw	a3, 40(s0)
   15178: 11 cf        	beqz	a4, 0x15194 <__swsetup_r+0x54>
   1517a: e2 60        	ld	ra, 24(sp)
   1517c: 42 64        	ld	s0, 16(sp)
   1517e: a2 64        	ld	s1, 8(sp)
   15180: 05 61        	addi	sp, sp, 32
   15182: 82 80        	ret
   15184: 93 f6 27 00  	andi	a3, a5, 2
   15188: 01 46        	mv	a2, zero
   1518a: 91 e2        	bnez	a3, 0x1518e <__swsetup_r+0x4e>
   1518c: 10 50        	lw	a2, 32(s0)
   1518e: 50 c4        	sw	a2, 12(s0)
   15190: 01 45        	mv	a0, zero
   15192: 65 f7        	bnez	a4, 0x1517a <__swsetup_r+0x3a>
   15194: 13 f7 07 08  	andi	a4, a5, 128
   15198: 6d d3        	beqz	a4, 0x1517a <__swsetup_r+0x3a>
   1519a: 93 e7 07 04  	ori	a5, a5, 64
   1519e: 23 18 f4 00  	sh	a5, 16(s0)
   151a2: 7d 55        	addi	a0, zero, -1
   151a4: d9 bf        	j	0x1517a <__swsetup_r+0x3a>
   151a6: 3e 85        	mv	a0, a5
   151a8: ef 00 a0 58  	jal	0x15732 <__sinit>
   151ac: 83 17 04 01  	lh	a5, 16(s0)
   151b0: 13 f7 87 00  	andi	a4, a5, 8
   151b4: 55 f7        	bnez	a4, 0x15160 <__swsetup_r+0x20>
   151b6: 13 f7 07 01  	andi	a4, a5, 16
   151ba: 21 cf        	beqz	a4, 0x15212 <__swsetup_r+0xd2>
   151bc: 13 f7 47 00  	andi	a4, a5, 4
   151c0: 0d e7        	bnez	a4, 0x151ea <__swsetup_r+0xaa>
   151c2: 18 6c        	ld	a4, 24(s0)
   151c4: 93 e7 87 00  	ori	a5, a5, 8
   151c8: 23 18 f4 00  	sh	a5, 16(s0)
   151cc: 41 ff        	bnez	a4, 0x15164 <__swsetup_r+0x24>
   151ce: 93 f6 07 28  	andi	a3, a5, 640
   151d2: 13 06 00 20  	addi	a2, zero, 512
   151d6: e3 87 c6 f8  	beq	a3, a2, 0x15164 <__swsetup_r+0x24>
   151da: a2 85        	mv	a1, s0
   151dc: 26 85        	mv	a0, s1
   151de: ef 30 60 1b  	jal	0x18394 <__smakebuf_r>
   151e2: 83 17 04 01  	lh	a5, 16(s0)
   151e6: 18 6c        	ld	a4, 24(s0)
   151e8: b5 bf        	j	0x15164 <__swsetup_r+0x24>
   151ea: 2c 6c        	ld	a1, 88(s0)
   151ec: 81 cd        	beqz	a1, 0x15204 <__swsetup_r+0xc4>
   151ee: 13 07 44 07  	addi	a4, s0, 116
   151f2: 63 87 e5 00  	beq	a1, a4, 0x15200 <__swsetup_r+0xc0>
   151f6: 26 85        	mv	a0, s1
   151f8: ef 00 e0 66  	jal	0x15866 <_free_r>
   151fc: 83 17 04 01  	lh	a5, 16(s0)
   15200: 23 3c 04 04  	sd	zero, 88(s0)
   15204: 18 6c        	ld	a4, 24(s0)
   15206: 93 f7 b7 fd  	andi	a5, a5, -37
   1520a: 23 24 04 00  	sw	zero, 8(s0)
   1520e: 18 e0        	sd	a4, 0(s0)
   15210: 55 bf        	j	0x151c4 <__swsetup_r+0x84>
   15212: 25 47        	addi	a4, zero, 9
   15214: 98 c0        	sw	a4, 0(s1)
   15216: 93 e7 07 04  	ori	a5, a5, 64
   1521a: 51 b7        	j	0x1519e <__swsetup_r+0x5e>

000000000001521c <__call_exitprocs>:
   1521c: 5d 71        	addi	sp, sp, -80
   1521e: 52 f0        	sd	s4, 32(sp)
   15220: 03 ba 81 83  	ld	s4, -1992(gp)
   15224: 4a f8        	sd	s2, 48(sp)
   15226: 86 e4        	sd	ra, 72(sp)
   15228: 03 39 8a 1f  	ld	s2, 504(s4)
   1522c: a2 e0        	sd	s0, 64(sp)
   1522e: 26 fc        	sd	s1, 56(sp)
   15230: 4e f4        	sd	s3, 40(sp)
   15232: 56 ec        	sd	s5, 24(sp)
   15234: 5a e8        	sd	s6, 16(sp)
   15236: 5e e4        	sd	s7, 8(sp)
   15238: 62 e0        	sd	s8, 0(sp)
   1523a: 63 08 09 02  	beqz	s2, 0x1526a <__call_exitprocs+0x4e>
   1523e: 2a 8b        	mv	s6, a0
   15240: ae 8b        	mv	s7, a1
   15242: 85 4a        	addi	s5, zero, 1
   15244: fd 59        	addi	s3, zero, -1
   15246: 83 24 89 00  	lw	s1, 8(s2)
   1524a: 1b 84 f4 ff  	addiw	s0, s1, -1
   1524e: 63 4e 04 00  	bltz	s0, 0x1526a <__call_exitprocs+0x4e>
   15252: 8e 04        	slli	s1, s1, 3
   15254: ca 94        	add	s1, s1, s2
   15256: 63 86 0b 02  	beqz	s7, 0x15282 <__call_exitprocs+0x66>
   1525a: 83 b7 84 20  	ld	a5, 520(s1)
   1525e: 63 82 77 03  	beq	a5, s7, 0x15282 <__call_exitprocs+0x66>
   15262: 7d 34        	addiw	s0, s0, -1
   15264: e1 14        	addi	s1, s1, -8
   15266: e3 18 34 ff  	bne	s0, s3, 0x15256 <__call_exitprocs+0x3a>
   1526a: a6 60        	ld	ra, 72(sp)
   1526c: 06 64        	ld	s0, 64(sp)
   1526e: e2 74        	ld	s1, 56(sp)
   15270: 42 79        	ld	s2, 48(sp)
   15272: a2 79        	ld	s3, 40(sp)
   15274: 02 7a        	ld	s4, 32(sp)
   15276: e2 6a        	ld	s5, 24(sp)
   15278: 42 6b        	ld	s6, 16(sp)
   1527a: a2 6b        	ld	s7, 8(sp)
   1527c: 02 6c        	ld	s8, 0(sp)
   1527e: 61 61        	addi	sp, sp, 80
   15280: 82 80        	ret
   15282: 83 27 89 00  	lw	a5, 8(s2)
   15286: 98 64        	ld	a4, 8(s1)
   15288: fd 37        	addiw	a5, a5, -1
   1528a: 63 84 87 04  	beq	a5, s0, 0x152d2 <__call_exitprocs+0xb6>
   1528e: 23 b4 04 00  	sd	zero, 8(s1)
   15292: 61 db        	beqz	a4, 0x15262 <__call_exitprocs+0x46>
   15294: 83 27 09 31  	lw	a5, 784(s2)
   15298: bb 96 8a 00  	sllw	a3, s5, s0
   1529c: 03 2c 89 00  	lw	s8, 8(s2)
   152a0: f5 8f        	and	a5, a5, a3
   152a2: 81 27        	sext.w	a5, a5
   152a4: 89 ef        	bnez	a5, 0x152be <__call_exitprocs+0xa2>
   152a6: 02 97        	jalr	a4
   152a8: 03 27 89 00  	lw	a4, 8(s2)
   152ac: 83 37 8a 1f  	ld	a5, 504(s4)
   152b0: 63 14 87 01  	bne	a4, s8, 0x152b8 <__call_exitprocs+0x9c>
   152b4: e3 87 27 fb  	beq	a5, s2, 0x15262 <__call_exitprocs+0x46>
   152b8: cd db        	beqz	a5, 0x1526a <__call_exitprocs+0x4e>
   152ba: 3e 89        	mv	s2, a5
   152bc: 69 b7        	j	0x15246 <__call_exitprocs+0x2a>
   152be: 83 27 49 31  	lw	a5, 788(s2)
   152c2: 83 b5 84 10  	ld	a1, 264(s1)
   152c6: f5 8f        	and	a5, a5, a3
   152c8: 81 27        	sext.w	a5, a5
   152ca: 99 e7        	bnez	a5, 0x152d8 <__call_exitprocs+0xbc>
   152cc: 5a 85        	mv	a0, s6
   152ce: 02 97        	jalr	a4
   152d0: e1 bf        	j	0x152a8 <__call_exitprocs+0x8c>
   152d2: 23 24 89 00  	sw	s0, 8(s2)
   152d6: 75 bf        	j	0x15292 <__call_exitprocs+0x76>
   152d8: 2e 85        	mv	a0, a1
   152da: 02 97        	jalr	a4
   152dc: f1 b7        	j	0x152a8 <__call_exitprocs+0x8c>

00000000000152de <__sflush_r>:
   152de: 83 97 05 01  	lh	a5, 16(a1)
   152e2: 79 71        	addi	sp, sp, -48
   152e4: 22 f0        	sd	s0, 32(sp)
   152e6: 4e e4        	sd	s3, 8(sp)
   152e8: 06 f4        	sd	ra, 40(sp)
   152ea: 26 ec        	sd	s1, 24(sp)
   152ec: 4a e8        	sd	s2, 16(sp)
   152ee: 13 f7 87 00  	andi	a4, a5, 8
   152f2: 2e 84        	mv	s0, a1
   152f4: aa 89        	mv	s3, a0
   152f6: 69 e3        	bnez	a4, 0x153b8 <__sflush_r+0xda>
   152f8: 05 67        	lui	a4, 1
   152fa: 13 07 07 80  	addi	a4, a4, -2048
   152fe: 94 45        	lw	a3, 8(a1)
   15300: 5d 8f        	or	a4, a4, a5
   15302: 23 98 e5 00  	sh	a4, 16(a1)
   15306: 63 54 d0 10  	blez	a3, 0x1540e <__sflush_r+0x130>
   1530a: 03 38 84 04  	ld	a6, 72(s0)
   1530e: 63 0d 08 08  	beqz	a6, 0x153a8 <__sflush_r+0xca>
   15312: b1 83        	srli	a5, a5, 12
   15314: 83 a4 09 00  	lw	s1, 0(s3)
   15318: 85 8b        	andi	a5, a5, 1
   1531a: 23 a0 09 00  	sw	zero, 0(s3)
   1531e: 0c 78        	ld	a1, 48(s0)
   15320: e5 ef        	bnez	a5, 0x15418 <__sflush_r+0x13a>
   15322: 01 46        	mv	a2, zero
   15324: 85 46        	addi	a3, zero, 1
   15326: 4e 85        	mv	a0, s3
   15328: 02 98        	jalr	a6
   1532a: fd 57        	addi	a5, zero, -1
   1532c: 2a 86        	mv	a2, a0
   1532e: 63 08 f5 10  	beq	a0, a5, 0x1543e <__sflush_r+0x160>
   15332: 03 17 04 01  	lh	a4, 16(s0)
   15336: 03 38 84 04  	ld	a6, 72(s0)
   1533a: 0c 78        	ld	a1, 48(s0)
   1533c: 11 8b        	andi	a4, a4, 4
   1533e: 19 c7        	beqz	a4, 0x1534c <__sflush_r+0x6e>
   15340: 18 44        	lw	a4, 8(s0)
   15342: 3c 6c        	ld	a5, 88(s0)
   15344: 19 8e        	sub	a2, a2, a4
   15346: 99 c3        	beqz	a5, 0x1534c <__sflush_r+0x6e>
   15348: 3c 58        	lw	a5, 112(s0)
   1534a: 1d 8e        	sub	a2, a2, a5
   1534c: 81 46        	mv	a3, zero
   1534e: 4e 85        	mv	a0, s3
   15350: 02 98        	jalr	a6
   15352: fd 57        	addi	a5, zero, -1
   15354: 63 14 f5 0c  	bne	a0, a5, 0x1541c <__sflush_r+0x13e>
   15358: 83 a6 09 00  	lw	a3, 0(s3)
   1535c: f5 47        	addi	a5, zero, 29
   1535e: 63 ea d7 08  	bltu	a5, a3, 0x153f2 <__sflush_r+0x114>
   15362: 37 07 c0 df  	lui	a4, 916480
   15366: 79 17        	addi	a4, a4, -2
   15368: 33 57 d7 40  	sra	a4, a4, a3
   1536c: 05 8b        	andi	a4, a4, 1
   1536e: 83 17 04 01  	lh	a5, 16(s0)
   15372: 51 e3        	bnez	a4, 0x153f6 <__sflush_r+0x118>
   15374: 10 6c        	ld	a2, 24(s0)
   15376: 7d 77        	lui	a4, 1048575
   15378: 13 07 f7 7f  	addi	a4, a4, 2047
   1537c: 7d 8f        	and	a4, a4, a5
   1537e: b1 83        	srli	a5, a5, 12
   15380: 23 18 e4 00  	sh	a4, 16(s0)
   15384: 23 24 04 00  	sw	zero, 8(s0)
   15388: 10 e0        	sd	a2, 0(s0)
   1538a: 85 8b        	andi	a5, a5, 1
   1538c: f9 e7        	bnez	a5, 0x1545a <__sflush_r+0x17c>
   1538e: 2c 6c        	ld	a1, 88(s0)
   15390: 23 a0 99 00  	sw	s1, 0(s3)
   15394: 91 c9        	beqz	a1, 0x153a8 <__sflush_r+0xca>
   15396: 93 07 44 07  	addi	a5, s0, 116
   1539a: 63 85 f5 00  	beq	a1, a5, 0x153a4 <__sflush_r+0xc6>
   1539e: 4e 85        	mv	a0, s3
   153a0: ef 00 60 4c  	jal	0x15866 <_free_r>
   153a4: 23 3c 04 04  	sd	zero, 88(s0)
   153a8: 01 45        	mv	a0, zero
   153aa: a2 70        	ld	ra, 40(sp)
   153ac: 02 74        	ld	s0, 32(sp)
   153ae: e2 64        	ld	s1, 24(sp)
   153b0: 42 69        	ld	s2, 16(sp)
   153b2: a2 69        	ld	s3, 8(sp)
   153b4: 45 61        	addi	sp, sp, 48
   153b6: 82 80        	ret
   153b8: 03 b9 85 01  	ld	s2, 24(a1)
   153bc: e3 06 09 fe  	beqz	s2, 0x153a8 <__sflush_r+0xca>
   153c0: 84 61        	ld	s1, 0(a1)
   153c2: 8d 8b        	andi	a5, a5, 3
   153c4: 23 b0 25 01  	sd	s2, 0(a1)
   153c8: bb 84 24 41  	subw	s1, s1, s2
   153cc: 01 47        	mv	a4, zero
   153ce: 91 e3        	bnez	a5, 0x153d2 <__sflush_r+0xf4>
   153d0: 98 51        	lw	a4, 32(a1)
   153d2: 58 c4        	sw	a4, 12(s0)
   153d4: 63 46 90 00  	bgtz	s1, 0x153e0 <__sflush_r+0x102>
   153d8: c1 bf        	j	0x153a8 <__sflush_r+0xca>
   153da: 2a 99        	add	s2, s2, a0
   153dc: e3 56 90 fc  	blez	s1, 0x153a8 <__sflush_r+0xca>
   153e0: 3c 60        	ld	a5, 64(s0)
   153e2: 0c 78        	ld	a1, 48(s0)
   153e4: a6 86        	mv	a3, s1
   153e6: 4a 86        	mv	a2, s2
   153e8: 4e 85        	mv	a0, s3
   153ea: 82 97        	jalr	a5
   153ec: 89 9c        	subw	s1, s1, a0
   153ee: e3 46 a0 fe  	bgtz	a0, 0x153da <__sflush_r+0xfc>
   153f2: 83 17 04 01  	lh	a5, 16(s0)
   153f6: 93 e7 07 04  	ori	a5, a5, 64
   153fa: a2 70        	ld	ra, 40(sp)
   153fc: 23 18 f4 00  	sh	a5, 16(s0)
   15400: 02 74        	ld	s0, 32(sp)
   15402: e2 64        	ld	s1, 24(sp)
   15404: 42 69        	ld	s2, 16(sp)
   15406: a2 69        	ld	s3, 8(sp)
   15408: 7d 55        	addi	a0, zero, -1
   1540a: 45 61        	addi	sp, sp, 48
   1540c: 82 80        	ret
   1540e: b4 59        	lw	a3, 112(a1)
   15410: e3 4d d0 ee  	bgtz	a3, 0x1530a <__sflush_r+0x2c>
   15414: 01 45        	mv	a0, zero
   15416: 51 bf        	j	0x153aa <__sflush_r+0xcc>
   15418: 50 68        	ld	a2, 144(s0)
   1541a: 0d b7        	j	0x1533c <__sflush_r+0x5e>
   1541c: 83 17 04 01  	lh	a5, 16(s0)
   15420: 14 6c        	ld	a3, 24(s0)
   15422: 7d 77        	lui	a4, 1048575
   15424: 13 07 f7 7f  	addi	a4, a4, 2047
   15428: 7d 8f        	and	a4, a4, a5
   1542a: b1 83        	srli	a5, a5, 12
   1542c: 23 18 e4 00  	sh	a4, 16(s0)
   15430: 23 24 04 00  	sw	zero, 8(s0)
   15434: 14 e0        	sd	a3, 0(s0)
   15436: 85 8b        	andi	a5, a5, 1
   15438: b9 db        	beqz	a5, 0x1538e <__sflush_r+0xb0>
   1543a: 48 e8        	sd	a0, 144(s0)
   1543c: 89 bf        	j	0x1538e <__sflush_r+0xb0>
   1543e: 83 a7 09 00  	lw	a5, 0(s3)
   15442: e3 88 07 ee  	beqz	a5, 0x15332 <__sflush_r+0x54>
   15446: 75 47        	addi	a4, zero, 29
   15448: 63 85 e7 00  	beq	a5, a4, 0x15452 <__sflush_r+0x174>
   1544c: 59 47        	addi	a4, zero, 22
   1544e: e3 92 e7 fa  	bne	a5, a4, 0x153f2 <__sflush_r+0x114>
   15452: 23 a0 99 00  	sw	s1, 0(s3)
   15456: 01 45        	mv	a0, zero
   15458: 89 bf        	j	0x153aa <__sflush_r+0xcc>
   1545a: 95 fa        	bnez	a3, 0x1538e <__sflush_r+0xb0>
   1545c: 48 e8        	sd	a0, 144(s0)
   1545e: 05 bf        	j	0x1538e <__sflush_r+0xb0>

0000000000015460 <_fflush_r>:
   15460: 01 11        	addi	sp, sp, -32
   15462: 22 e8        	sd	s0, 16(sp)
   15464: 06 ec        	sd	ra, 24(sp)
   15466: 2a 84        	mv	s0, a0
   15468: 19 c1        	beqz	a0, 0x1546e <_fflush_r+0xe>
   1546a: 3c 49        	lw	a5, 80(a0)
   1546c: 89 cb        	beqz	a5, 0x1547e <_fflush_r+0x1e>
   1546e: 83 97 05 01  	lh	a5, 16(a1)
   15472: 89 ef        	bnez	a5, 0x1548c <_fflush_r+0x2c>
   15474: e2 60        	ld	ra, 24(sp)
   15476: 42 64        	ld	s0, 16(sp)
   15478: 01 45        	mv	a0, zero
   1547a: 05 61        	addi	sp, sp, 32
   1547c: 82 80        	ret
   1547e: 2e e4        	sd	a1, 8(sp)
   15480: ef 00 20 2b  	jal	0x15732 <__sinit>
   15484: a2 65        	ld	a1, 8(sp)
   15486: 83 97 05 01  	lh	a5, 16(a1)
   1548a: ed d7        	beqz	a5, 0x15474 <_fflush_r+0x14>
   1548c: 22 85        	mv	a0, s0
   1548e: 42 64        	ld	s0, 16(sp)
   15490: e2 60        	ld	ra, 24(sp)
   15492: 05 61        	addi	sp, sp, 32
   15494: a9 b5        	j	0x152de <__sflush_r>

0000000000015496 <fflush>:
   15496: 1d cd        	beqz	a0, 0x154d4 <fflush+0x3e>
   15498: 01 11        	addi	sp, sp, -32
   1549a: 22 e8        	sd	s0, 16(sp)
   1549c: 2a 84        	mv	s0, a0
   1549e: 03 b5 81 93  	ld	a0, -1736(gp)
   154a2: 06 ec        	sd	ra, 24(sp)
   154a4: 19 c1        	beqz	a0, 0x154aa <fflush+0x14>
   154a6: 3c 49        	lw	a5, 80(a0)
   154a8: 91 cf        	beqz	a5, 0x154c4 <fflush+0x2e>
   154aa: 83 17 04 01  	lh	a5, 16(s0)
   154ae: 91 e7        	bnez	a5, 0x154ba <fflush+0x24>
   154b0: e2 60        	ld	ra, 24(sp)
   154b2: 42 64        	ld	s0, 16(sp)
   154b4: 01 45        	mv	a0, zero
   154b6: 05 61        	addi	sp, sp, 32
   154b8: 82 80        	ret
   154ba: a2 85        	mv	a1, s0
   154bc: 42 64        	ld	s0, 16(sp)
   154be: e2 60        	ld	ra, 24(sp)
   154c0: 05 61        	addi	sp, sp, 32
   154c2: 31 bd        	j	0x152de <__sflush_r>
   154c4: 2a e4        	sd	a0, 8(sp)
   154c6: ef 00 c0 26  	jal	0x15732 <__sinit>
   154ca: 83 17 04 01  	lh	a5, 16(s0)
   154ce: 22 65        	ld	a0, 8(sp)
   154d0: e5 d3        	beqz	a5, 0x154b0 <fflush+0x1a>
   154d2: e5 b7        	j	0x154ba <fflush+0x24>
   154d4: 03 b5 81 83  	ld	a0, -1992(gp)
   154d8: d5 65        	lui	a1, 21
   154da: 93 85 05 46  	addi	a1, a1, 1120
   154de: 89 ad        	j	0x15b30 <_fwalk_reent>

00000000000154e0 <__fp_lock>:
   154e0: 01 45        	mv	a0, zero
   154e2: 82 80        	ret

00000000000154e4 <_cleanup_r>:
   154e4: f5 65        	lui	a1, 29
   154e6: 93 85 c5 93  	addi	a1, a1, -1732
   154ea: 99 a5        	j	0x15b30 <_fwalk_reent>

00000000000154ec <__fp_unlock>:
   154ec: 01 45        	mv	a0, zero
   154ee: 82 80        	ret

00000000000154f0 <__sinit.part.0>:
   154f0: 39 71        	addi	sp, sp, -64
   154f2: d5 67        	lui	a5, 21
   154f4: 06 fc        	sd	ra, 56(sp)
   154f6: 22 f8        	sd	s0, 48(sp)
   154f8: 26 f4        	sd	s1, 40(sp)
   154fa: 4a f0        	sd	s2, 32(sp)
   154fc: 4e ec        	sd	s3, 24(sp)
   154fe: 52 e8        	sd	s4, 16(sp)
   15500: 56 e4        	sd	s5, 8(sp)
   15502: 5a e0        	sd	s6, 0(sp)
   15504: 00 65        	ld	s0, 8(a0)
   15506: 93 87 47 4e  	addi	a5, a5, 1252
   1550a: 3c ed        	sd	a5, 88(a0)
   1550c: 13 07 85 53  	addi	a4, a0, 1336
   15510: 8d 47        	addi	a5, zero, 3
   15512: 23 38 e5 52  	sd	a4, 1328(a0)
   15516: 23 24 f5 52  	sw	a5, 1320(a0)
   1551a: 23 30 05 52  	sd	zero, 1312(a0)
   1551e: 91 47        	addi	a5, zero, 4
   15520: 2a 89        	mv	s2, a0
   15522: 1c c8        	sw	a5, 16(s0)
   15524: 21 46        	addi	a2, zero, 8
   15526: 81 45        	mv	a1, zero
   15528: 23 26 04 0a  	sw	zero, 172(s0)
   1552c: 23 30 04 00  	sd	zero, 0(s0)
   15530: 23 34 04 00  	sd	zero, 8(s0)
   15534: 23 3c 04 00  	sd	zero, 24(s0)
   15538: 23 20 04 02  	sw	zero, 32(s0)
   1553c: 23 24 04 02  	sw	zero, 40(s0)
   15540: 13 05 44 0a  	addi	a0, s0, 164
   15544: ef d0 3f eb  	jal	0x133f6 <memset>
   15548: 69 6b        	lui	s6, 26
   1554a: 83 34 09 01  	ld	s1, 16(s2)
   1554e: e9 6a        	lui	s5, 26
   15550: 69 6a        	lui	s4, 26
   15552: e9 69        	lui	s3, 26
   15554: 13 0b eb a4  	addi	s6, s6, -1458
   15558: 93 8a aa a8  	addi	s5, s5, -1398
   1555c: 13 0a ea ad  	addi	s4, s4, -1314
   15560: 93 89 09 b3  	addi	s3, s3, -1232
   15564: c1 67        	lui	a5, 16
   15566: 23 3c 64 03  	sd	s6, 56(s0)
   1556a: 23 30 54 05  	sd	s5, 64(s0)
   1556e: 23 34 44 05  	sd	s4, 72(s0)
   15572: 23 38 34 05  	sd	s3, 80(s0)
   15576: 00 f8        	sd	s0, 48(s0)
   15578: a5 07        	addi	a5, a5, 9
   1557a: 9c c8        	sw	a5, 16(s1)
   1557c: 21 46        	addi	a2, zero, 8
   1557e: 81 45        	mv	a1, zero
   15580: 23 a6 04 0a  	sw	zero, 172(s1)
   15584: 23 b0 04 00  	sd	zero, 0(s1)
   15588: 23 b4 04 00  	sd	zero, 8(s1)
   1558c: 23 bc 04 00  	sd	zero, 24(s1)
   15590: 23 a0 04 02  	sw	zero, 32(s1)
   15594: 23 a4 04 02  	sw	zero, 40(s1)
   15598: 13 85 44 0a  	addi	a0, s1, 164
   1559c: ef d0 bf e5  	jal	0x133f6 <memset>
   155a0: 03 34 89 01  	ld	s0, 24(s2)
   155a4: b7 07 02 00  	lui	a5, 32
   155a8: 23 bc 64 03  	sd	s6, 56(s1)
   155ac: 23 b0 54 05  	sd	s5, 64(s1)
   155b0: 23 b4 44 05  	sd	s4, 72(s1)
   155b4: 23 b8 34 05  	sd	s3, 80(s1)
   155b8: 84 f8        	sd	s1, 48(s1)
   155ba: c9 07        	addi	a5, a5, 18
   155bc: 1c c8        	sw	a5, 16(s0)
   155be: 23 26 04 0a  	sw	zero, 172(s0)
   155c2: 23 30 04 00  	sd	zero, 0(s0)
   155c6: 23 34 04 00  	sd	zero, 8(s0)
   155ca: 23 3c 04 00  	sd	zero, 24(s0)
   155ce: 23 20 04 02  	sw	zero, 32(s0)
   155d2: 23 24 04 02  	sw	zero, 40(s0)
   155d6: 13 05 44 0a  	addi	a0, s0, 164
   155da: 21 46        	addi	a2, zero, 8
   155dc: 81 45        	mv	a1, zero
   155de: ef d0 9f e1  	jal	0x133f6 <memset>
   155e2: e2 70        	ld	ra, 56(sp)
   155e4: 23 3c 64 03  	sd	s6, 56(s0)
   155e8: 23 30 54 05  	sd	s5, 64(s0)
   155ec: 23 34 44 05  	sd	s4, 72(s0)
   155f0: 23 38 34 05  	sd	s3, 80(s0)
   155f4: 00 f8        	sd	s0, 48(s0)
   155f6: 42 74        	ld	s0, 48(sp)
   155f8: 85 47        	addi	a5, zero, 1
   155fa: 23 28 f9 04  	sw	a5, 80(s2)
   155fe: a2 74        	ld	s1, 40(sp)
   15600: 02 79        	ld	s2, 32(sp)
   15602: e2 69        	ld	s3, 24(sp)
   15604: 42 6a        	ld	s4, 16(sp)
   15606: a2 6a        	ld	s5, 8(sp)
   15608: 02 6b        	ld	s6, 0(sp)
   1560a: 21 61        	addi	sp, sp, 64
   1560c: 82 80        	ret

000000000001560e <__sfmoreglue>:
   1560e: 01 11        	addi	sp, sp, -32
   15610: 26 e4        	sd	s1, 8(sp)
   15612: 93 07 00 0b  	addi	a5, zero, 176
   15616: 9b 84 f5 ff  	addiw	s1, a1, -1
   1561a: b3 84 f4 02  	mul	s1, s1, a5
   1561e: 4a e0        	sd	s2, 0(sp)
   15620: 2e 89        	mv	s2, a1
   15622: 22 e8        	sd	s0, 16(sp)
   15624: 06 ec        	sd	ra, 24(sp)
   15626: 93 85 84 0c  	addi	a1, s1, 200
   1562a: ef 20 70 70  	jal	0x18530 <_malloc_r>
   1562e: 2a 84        	mv	s0, a0
   15630: 01 cd        	beqz	a0, 0x15648 <__sfmoreglue+0x3a>
   15632: 61 05        	addi	a0, a0, 24
   15634: 23 30 04 00  	sd	zero, 0(s0)
   15638: 23 24 24 01  	sw	s2, 8(s0)
   1563c: 08 e8        	sd	a0, 16(s0)
   1563e: 13 86 04 0b  	addi	a2, s1, 176
   15642: 81 45        	mv	a1, zero
   15644: ef d0 3f db  	jal	0x133f6 <memset>
   15648: e2 60        	ld	ra, 24(sp)
   1564a: 22 85        	mv	a0, s0
   1564c: 42 64        	ld	s0, 16(sp)
   1564e: a2 64        	ld	s1, 8(sp)
   15650: 02 69        	ld	s2, 0(sp)
   15652: 05 61        	addi	sp, sp, 32
   15654: 82 80        	ret

0000000000015656 <__sfp>:
   15656: 79 71        	addi	sp, sp, -48
   15658: 26 ec        	sd	s1, 24(sp)
   1565a: 83 b4 81 83  	ld	s1, -1992(gp)
   1565e: 4a e8        	sd	s2, 16(sp)
   15660: 06 f4        	sd	ra, 40(sp)
   15662: bc 48        	lw	a5, 80(s1)
   15664: 22 f0        	sd	s0, 32(sp)
   15666: 4e e4        	sd	s3, 8(sp)
   15668: 2a 89        	mv	s2, a0
   1566a: bd cf        	beqz	a5, 0x156e8 <__sfp+0x92>
   1566c: 93 84 04 52  	addi	s1, s1, 1312
   15670: 93 09 00 0b  	addi	s3, zero, 176
   15674: 98 44        	lw	a4, 8(s1)
   15676: 80 68        	ld	s0, 16(s1)
   15678: 63 54 e0 06  	blez	a4, 0x156e0 <__sfp+0x8a>
   1567c: 02 17        	slli	a4, a4, 32
   1567e: 01 93        	srli	a4, a4, 32
   15680: 33 07 37 03  	mul	a4, a4, s3
   15684: 22 97        	add	a4, a4, s0
   15686: 29 a0        	j	0x15690 <__sfp+0x3a>
   15688: 13 04 04 0b  	addi	s0, s0, 176
   1568c: 63 0a e4 04  	beq	s0, a4, 0x156e0 <__sfp+0x8a>
   15690: 83 17 04 01  	lh	a5, 16(s0)
   15694: f5 fb        	bnez	a5, 0x15688 <__sfp+0x32>
   15696: c1 77        	lui	a5, 1048560
   15698: 85 07        	addi	a5, a5, 1
   1569a: 23 26 04 0a  	sw	zero, 172(s0)
   1569e: 23 30 04 00  	sd	zero, 0(s0)
   156a2: 23 34 04 00  	sd	zero, 8(s0)
   156a6: 1c c8        	sw	a5, 16(s0)
   156a8: 23 3c 04 00  	sd	zero, 24(s0)
   156ac: 23 20 04 02  	sw	zero, 32(s0)
   156b0: 23 24 04 02  	sw	zero, 40(s0)
   156b4: 21 46        	addi	a2, zero, 8
   156b6: 81 45        	mv	a1, zero
   156b8: 13 05 44 0a  	addi	a0, s0, 164
   156bc: ef d0 bf d3  	jal	0x133f6 <memset>
   156c0: 23 3c 04 04  	sd	zero, 88(s0)
   156c4: 23 20 04 06  	sw	zero, 96(s0)
   156c8: 23 3c 04 06  	sd	zero, 120(s0)
   156cc: 23 20 04 08  	sw	zero, 128(s0)
   156d0: a2 70        	ld	ra, 40(sp)
   156d2: 22 85        	mv	a0, s0
   156d4: 02 74        	ld	s0, 32(sp)
   156d6: e2 64        	ld	s1, 24(sp)
   156d8: 42 69        	ld	s2, 16(sp)
   156da: a2 69        	ld	s3, 8(sp)
   156dc: 45 61        	addi	sp, sp, 48
   156de: 82 80        	ret
   156e0: 80 60        	ld	s0, 0(s1)
   156e2: 19 c4        	beqz	s0, 0x156f0 <__sfp+0x9a>
   156e4: a2 84        	mv	s1, s0
   156e6: 79 b7        	j	0x15674 <__sfp+0x1e>
   156e8: 26 85        	mv	a0, s1
   156ea: ef f0 7f e0  	jal	0x154f0 <__sinit.part.0>
   156ee: bd bf        	j	0x1566c <__sfp+0x16>
   156f0: 93 05 80 2d  	addi	a1, zero, 728
   156f4: 4a 85        	mv	a0, s2
   156f6: ef 20 b0 63  	jal	0x18530 <_malloc_r>
   156fa: 2a 84        	mv	s0, a0
   156fc: 19 cd        	beqz	a0, 0x1571a <__sfp+0xc4>
   156fe: 61 05        	addi	a0, a0, 24
   15700: 91 47        	addi	a5, zero, 4
   15702: 23 30 04 00  	sd	zero, 0(s0)
   15706: 1c c4        	sw	a5, 8(s0)
   15708: 08 e8        	sd	a0, 16(s0)
   1570a: 13 06 00 2c  	addi	a2, zero, 704
   1570e: 81 45        	mv	a1, zero
   15710: ef d0 7f ce  	jal	0x133f6 <memset>
   15714: 80 e0        	sd	s0, 0(s1)
   15716: a2 84        	mv	s1, s0
   15718: b1 bf        	j	0x15674 <__sfp+0x1e>
   1571a: 23 b0 04 00  	sd	zero, 0(s1)
   1571e: b1 47        	addi	a5, zero, 12
   15720: 23 20 f9 00  	sw	a5, 0(s2)
   15724: 75 b7        	j	0x156d0 <__sfp+0x7a>

0000000000015726 <_cleanup>:
   15726: 03 b5 81 83  	ld	a0, -1992(gp)
   1572a: f5 65        	lui	a1, 29
   1572c: 93 85 c5 93  	addi	a1, a1, -1732
   15730: 01 a1        	j	0x15b30 <_fwalk_reent>

0000000000015732 <__sinit>:
   15732: 3c 49        	lw	a5, 80(a0)
   15734: 91 c3        	beqz	a5, 0x15738 <__sinit+0x6>
   15736: 82 80        	ret
   15738: 65 bb        	j	0x154f0 <__sinit.part.0>

000000000001573a <__sfp_lock_acquire>:
   1573a: 82 80        	ret

000000000001573c <__sfp_lock_release>:
   1573c: 82 80        	ret

000000000001573e <__sinit_lock_acquire>:
   1573e: 82 80        	ret

0000000000015740 <__sinit_lock_release>:
   15740: 82 80        	ret

0000000000015742 <__fp_lock_all>:
   15742: 03 b5 81 93  	ld	a0, -1736(gp)
   15746: d5 65        	lui	a1, 21
   15748: 93 85 05 4e  	addi	a1, a1, 1248
   1574c: a5 a6        	j	0x15ab4 <_fwalk>

000000000001574e <__fp_unlock_all>:
   1574e: 03 b5 81 93  	ld	a0, -1736(gp)
   15752: d5 65        	lui	a1, 21
   15754: 93 85 c5 4e  	addi	a1, a1, 1260
   15758: b1 ae        	j	0x15ab4 <_fwalk>

000000000001575a <__libc_fini_array>:
   1575a: 01 11        	addi	sp, sp, -32
   1575c: 22 e8        	sd	s0, 16(sp)
   1575e: b7 37 02 00  	lui	a5, 35
   15762: 37 34 02 00  	lui	s0, 35
   15766: 93 87 87 01  	addi	a5, a5, 24
   1576a: 13 04 04 02  	addi	s0, s0, 32
   1576e: 1d 8c        	sub	s0, s0, a5
   15770: 26 e4        	sd	s1, 8(sp)
   15772: 06 ec        	sd	ra, 24(sp)
   15774: 93 54 34 40  	srai	s1, s0, 3
   15778: 81 c8        	beqz	s1, 0x15788 <__libc_fini_array+0x2e>
   1577a: 61 14        	addi	s0, s0, -8
   1577c: 3e 94        	add	s0, s0, a5
   1577e: 1c 60        	ld	a5, 0(s0)
   15780: fd 14        	addi	s1, s1, -1
   15782: 61 14        	addi	s0, s0, -8
   15784: 82 97        	jalr	a5
   15786: e5 fc        	bnez	s1, 0x1577e <__libc_fini_array+0x24>
   15788: e2 60        	ld	ra, 24(sp)
   1578a: 42 64        	ld	s0, 16(sp)
   1578c: a2 64        	ld	s1, 8(sp)
   1578e: 05 61        	addi	sp, sp, 32
   15790: 82 80        	ret

0000000000015792 <_malloc_trim_r>:
   15792: 79 71        	addi	sp, sp, -48
   15794: 4e e4        	sd	s3, 8(sp)
   15796: b7 49 02 00  	lui	s3, 36
   1579a: 22 f0        	sd	s0, 32(sp)
   1579c: 26 ec        	sd	s1, 24(sp)
   1579e: 4a e8        	sd	s2, 16(sp)
   157a0: 52 e0        	sd	s4, 0(sp)
   157a2: 06 f4        	sd	ra, 40(sp)
   157a4: 2e 8a        	mv	s4, a1
   157a6: 2a 89        	mv	s2, a0
   157a8: 93 89 09 91  	addi	s3, s3, -1776
   157ac: ef 30 60 45  	jal	0x18c02 <__malloc_lock>
   157b0: 03 b7 09 01  	ld	a4, 16(s3)
   157b4: 85 67        	lui	a5, 1
   157b6: 13 84 f7 fd  	addi	s0, a5, -33
   157ba: 04 67        	ld	s1, 8(a4)
   157bc: f1 98        	andi	s1, s1, -4
   157be: 26 94        	add	s0, s0, s1
   157c0: 33 04 44 41  	sub	s0, s0, s4
   157c4: 31 80        	srli	s0, s0, 12
   157c6: 7d 14        	addi	s0, s0, -1
   157c8: 32 04        	slli	s0, s0, 12
   157ca: 63 4b f4 00  	blt	s0, a5, 0x157e0 <_malloc_trim_r+0x4e>
   157ce: 81 45        	mv	a1, zero
   157d0: 4a 85        	mv	a0, s2
   157d2: ef 40 00 14  	jal	0x19912 <_sbrk_r>
   157d6: 83 b7 09 01  	ld	a5, 16(s3)
   157da: a6 97        	add	a5, a5, s1
   157dc: 63 0e f5 00  	beq	a0, a5, 0x157f8 <_malloc_trim_r+0x66>
   157e0: 4a 85        	mv	a0, s2
   157e2: ef 30 20 42  	jal	0x18c04 <__malloc_unlock>
   157e6: a2 70        	ld	ra, 40(sp)
   157e8: 02 74        	ld	s0, 32(sp)
   157ea: e2 64        	ld	s1, 24(sp)
   157ec: 42 69        	ld	s2, 16(sp)
   157ee: a2 69        	ld	s3, 8(sp)
   157f0: 02 6a        	ld	s4, 0(sp)
   157f2: 01 45        	mv	a0, zero
   157f4: 45 61        	addi	sp, sp, 48
   157f6: 82 80        	ret
   157f8: b3 05 80 40  	neg	a1, s0
   157fc: 4a 85        	mv	a0, s2
   157fe: ef 40 40 11  	jal	0x19912 <_sbrk_r>
   15802: fd 57        	addi	a5, zero, -1
   15804: 63 0b f5 02  	beq	a0, a5, 0x1583a <_malloc_trim_r+0xa8>
   15808: b7 67 02 00  	lui	a5, 38
   1580c: 93 87 07 54  	addi	a5, a5, 1344
   15810: 98 43        	lw	a4, 0(a5)
   15812: 83 b6 09 01  	ld	a3, 16(s3)
   15816: 81 8c        	sub	s1, s1, s0
   15818: 93 e4 14 00  	ori	s1, s1, 1
   1581c: 01 9f        	subw	a4, a4, s0
   1581e: 4a 85        	mv	a0, s2
   15820: 84 e6        	sd	s1, 8(a3)
   15822: 98 c3        	sw	a4, 0(a5)
   15824: ef 30 00 3e  	jal	0x18c04 <__malloc_unlock>
   15828: a2 70        	ld	ra, 40(sp)
   1582a: 02 74        	ld	s0, 32(sp)
   1582c: e2 64        	ld	s1, 24(sp)
   1582e: 42 69        	ld	s2, 16(sp)
   15830: a2 69        	ld	s3, 8(sp)
   15832: 02 6a        	ld	s4, 0(sp)
   15834: 05 45        	addi	a0, zero, 1
   15836: 45 61        	addi	sp, sp, 48
   15838: 82 80        	ret
   1583a: 81 45        	mv	a1, zero
   1583c: 4a 85        	mv	a0, s2
   1583e: ef 40 40 0d  	jal	0x19912 <_sbrk_r>
   15842: 03 b7 09 01  	ld	a4, 16(s3)
   15846: fd 46        	addi	a3, zero, 31
   15848: b3 07 e5 40  	sub	a5, a0, a4
   1584c: e3 da f6 f8  	bge	a3, a5, 0x157e0 <_malloc_trim_r+0x4e>
   15850: 83 b6 01 94  	ld	a3, -1728(gp)
   15854: 93 e7 17 00  	ori	a5, a5, 1
   15858: 1c e7        	sd	a5, 8(a4)
   1585a: 15 8d        	sub	a0, a0, a3
   1585c: b7 66 02 00  	lui	a3, 38
   15860: 23 a0 a6 54  	sw	a0, 1344(a3)
   15864: b5 bf        	j	0x157e0 <_malloc_trim_r+0x4e>

0000000000015866 <_free_r>:
   15866: 63 85 05 12  	beqz	a1, 0x15990 <_free_r+0x12a>
   1586a: 01 11        	addi	sp, sp, -32
   1586c: 22 e8        	sd	s0, 16(sp)
   1586e: 26 e4        	sd	s1, 8(sp)
   15870: 2e 84        	mv	s0, a1
   15872: aa 84        	mv	s1, a0
   15874: 06 ec        	sd	ra, 24(sp)
   15876: ef 30 c0 38  	jal	0x18c02 <__malloc_lock>
   1587a: 03 35 84 ff  	ld	a0, -8(s0)
   1587e: 13 07 04 ff  	addi	a4, s0, -16
   15882: b7 45 02 00  	lui	a1, 36
   15886: 93 77 e5 ff  	andi	a5, a0, -2
   1588a: 33 06 f7 00  	add	a2, a4, a5
   1588e: 93 85 05 91  	addi	a1, a1, -1776
   15892: 14 66        	ld	a3, 8(a2)
   15894: 03 b8 05 01  	ld	a6, 16(a1)
   15898: f1 9a        	andi	a3, a3, -4
   1589a: 63 0e c8 12  	beq	a6, a2, 0x159d6 <_free_r+0x170>
   1589e: 14 e6        	sd	a3, 8(a2)
   158a0: 05 89        	andi	a0, a0, 1
   158a2: 33 08 d6 00  	add	a6, a2, a3
   158a6: 49 e1        	bnez	a0, 0x15928 <_free_r+0xc2>
   158a8: 03 33 04 ff  	ld	t1, -16(s0)
   158ac: 03 38 88 00  	ld	a6, 8(a6)
   158b0: 37 45 02 00  	lui	a0, 36
   158b4: 33 07 67 40  	sub	a4, a4, t1
   158b8: 83 38 07 01  	ld	a7, 16(a4)
   158bc: 13 05 05 92  	addi	a0, a0, -1760
   158c0: 9a 97        	add	a5, a5, t1
   158c2: 13 78 18 00  	andi	a6, a6, 1
   158c6: 63 8b a8 0e  	beq	a7, a0, 0x159bc <_free_r+0x156>
   158ca: 03 33 87 01  	ld	t1, 24(a4)
   158ce: 23 bc 68 00  	sd	t1, 24(a7)
   158d2: 23 38 13 01  	sd	a7, 16(t1)
   158d6: 63 09 08 12  	beqz	a6, 0x15a08 <_free_r+0x1a2>
   158da: 93 e6 17 00  	ori	a3, a5, 1
   158de: 14 e7        	sd	a3, 8(a4)
   158e0: 1c e2        	sd	a5, 0(a2)
   158e2: 93 06 f0 1f  	addi	a3, zero, 511
   158e6: 63 ee f6 04  	bltu	a3, a5, 0x15942 <_free_r+0xdc>
   158ea: 8d 83        	srli	a5, a5, 3
   158ec: 9b 86 17 00  	addiw	a3, a5, 1
   158f0: 9b 96 16 00  	slliw	a3, a3, 1
   158f4: 8e 06        	slli	a3, a3, 3
   158f6: 03 b8 85 00  	ld	a6, 8(a1)
   158fa: ae 96        	add	a3, a3, a1
   158fc: 88 62        	ld	a0, 0(a3)
   158fe: 1b d6 27 40  	sraiw	a2, a5, 2
   15902: 85 47        	addi	a5, zero, 1
   15904: b3 97 c7 00  	sll	a5, a5, a2
   15908: b3 e7 07 01  	or	a5, a5, a6
   1590c: 13 86 06 ff  	addi	a2, a3, -16
   15910: 10 ef        	sd	a2, 24(a4)
   15912: 08 eb        	sd	a0, 16(a4)
   15914: 9c e5        	sd	a5, 8(a1)
   15916: 98 e2        	sd	a4, 0(a3)
   15918: 18 ed        	sd	a4, 24(a0)
   1591a: 42 64        	ld	s0, 16(sp)
   1591c: e2 60        	ld	ra, 24(sp)
   1591e: 26 85        	mv	a0, s1
   15920: a2 64        	ld	s1, 8(sp)
   15922: 05 61        	addi	sp, sp, 32
   15924: 6f 30 00 2e  	j	0x18c04 <__malloc_unlock>
   15928: 03 35 88 00  	ld	a0, 8(a6)
   1592c: 05 89        	andi	a0, a0, 1
   1592e: 35 c1        	beqz	a0, 0x15992 <_free_r+0x12c>
   15930: 93 e6 17 00  	ori	a3, a5, 1
   15934: 23 3c d4 fe  	sd	a3, -8(s0)
   15938: 1c e2        	sd	a5, 0(a2)
   1593a: 93 06 f0 1f  	addi	a3, zero, 511
   1593e: e3 f6 f6 fa  	bgeu	a3, a5, 0x158ea <_free_r+0x84>
   15942: 93 d6 97 00  	srli	a3, a5, 9
   15946: 11 46        	addi	a2, zero, 4
   15948: 63 62 d6 0c  	bltu	a2, a3, 0x15a0c <_free_r+0x1a6>
   1594c: 93 d6 67 00  	srli	a3, a5, 6
   15950: 1b 85 96 03  	addiw	a0, a3, 57
   15954: 1b 15 15 00  	slliw	a0, a0, 1
   15958: 1b 86 86 03  	addiw	a2, a3, 56
   1595c: 0e 05        	slli	a0, a0, 3
   1595e: 2e 95        	add	a0, a0, a1
   15960: 14 61        	ld	a3, 0(a0)
   15962: 41 15        	addi	a0, a0, -16
   15964: 63 16 d5 00  	bne	a0, a3, 0x15970 <_free_r+0x10a>
   15968: ed a0        	j	0x15a52 <_free_r+0x1ec>
   1596a: 94 6a        	ld	a3, 16(a3)
   1596c: 63 06 d5 00  	beq	a0, a3, 0x15978 <_free_r+0x112>
   15970: 90 66        	ld	a2, 8(a3)
   15972: 71 9a        	andi	a2, a2, -4
   15974: e3 eb c7 fe  	bltu	a5, a2, 0x1596a <_free_r+0x104>
   15978: 88 6e        	ld	a0, 24(a3)
   1597a: 08 ef        	sd	a0, 24(a4)
   1597c: 14 eb        	sd	a3, 16(a4)
   1597e: 42 64        	ld	s0, 16(sp)
   15980: 18 e9        	sd	a4, 16(a0)
   15982: e2 60        	ld	ra, 24(sp)
   15984: 26 85        	mv	a0, s1
   15986: a2 64        	ld	s1, 8(sp)
   15988: 98 ee        	sd	a4, 24(a3)
   1598a: 05 61        	addi	sp, sp, 32
   1598c: 6f 30 80 27  	j	0x18c04 <__malloc_unlock>
   15990: 82 80        	ret
   15992: 37 45 02 00  	lui	a0, 36
   15996: b6 97        	add	a5, a5, a3
   15998: 13 05 05 92  	addi	a0, a0, -1760
   1599c: 14 6a        	ld	a3, 16(a2)
   1599e: 63 80 a6 0a  	beq	a3, a0, 0x15a3e <_free_r+0x1d8>
   159a2: 03 38 86 01  	ld	a6, 24(a2)
   159a6: 13 e5 17 00  	ori	a0, a5, 1
   159aa: 33 06 f7 00  	add	a2, a4, a5
   159ae: 23 bc 06 01  	sd	a6, 24(a3)
   159b2: 23 38 d8 00  	sd	a3, 16(a6)
   159b6: 08 e7        	sd	a0, 8(a4)
   159b8: 1c e2        	sd	a5, 0(a2)
   159ba: 25 b7        	j	0x158e2 <_free_r+0x7c>
   159bc: 63 17 08 0e  	bnez	a6, 0x15aaa <_free_r+0x244>
   159c0: 0c 6a        	ld	a1, 16(a2)
   159c2: 10 6e        	ld	a2, 24(a2)
   159c4: be 96        	add	a3, a3, a5
   159c6: 93 e7 16 00  	ori	a5, a3, 1
   159ca: 90 ed        	sd	a2, 24(a1)
   159cc: 0c ea        	sd	a1, 16(a2)
   159ce: 1c e7        	sd	a5, 8(a4)
   159d0: 36 97        	add	a4, a4, a3
   159d2: 14 e3        	sd	a3, 0(a4)
   159d4: 99 b7        	j	0x1591a <_free_r+0xb4>
   159d6: 05 89        	andi	a0, a0, 1
   159d8: be 96        	add	a3, a3, a5
   159da: 09 e9        	bnez	a0, 0x159ec <_free_r+0x186>
   159dc: 03 35 04 ff  	ld	a0, -16(s0)
   159e0: 09 8f        	sub	a4, a4, a0
   159e2: 1c 6f        	ld	a5, 24(a4)
   159e4: 10 6b        	ld	a2, 16(a4)
   159e6: aa 96        	add	a3, a3, a0
   159e8: 1c ee        	sd	a5, 24(a2)
   159ea: 90 eb        	sd	a2, 16(a5)
   159ec: 13 e6 16 00  	ori	a2, a3, 1
   159f0: 83 b7 81 94  	ld	a5, -1720(gp)
   159f4: 10 e7        	sd	a2, 8(a4)
   159f6: 98 e9        	sd	a4, 16(a1)
   159f8: e3 e1 f6 f2  	bltu	a3, a5, 0x1591a <_free_r+0xb4>
   159fc: 83 b5 01 97  	ld	a1, -1680(gp)
   15a00: 26 85        	mv	a0, s1
   15a02: ef f0 1f d9  	jal	0x15792 <_malloc_trim_r>
   15a06: 11 bf        	j	0x1591a <_free_r+0xb4>
   15a08: b6 97        	add	a5, a5, a3
   15a0a: 49 bf        	j	0x1599c <_free_r+0x136>
   15a0c: 51 46        	addi	a2, zero, 20
   15a0e: 63 70 d6 02  	bgeu	a2, a3, 0x15a2e <_free_r+0x1c8>
   15a12: 13 06 40 05  	addi	a2, zero, 84
   15a16: 63 69 d6 04  	bltu	a2, a3, 0x15a68 <_free_r+0x202>
   15a1a: 93 d6 c7 00  	srli	a3, a5, 12
   15a1e: 1b 85 f6 06  	addiw	a0, a3, 111
   15a22: 1b 15 15 00  	slliw	a0, a0, 1
   15a26: 1b 86 e6 06  	addiw	a2, a3, 110
   15a2a: 0e 05        	slli	a0, a0, 3
   15a2c: 0d bf        	j	0x1595e <_free_r+0xf8>
   15a2e: 1b 85 c6 05  	addiw	a0, a3, 92
   15a32: 1b 15 15 00  	slliw	a0, a0, 1
   15a36: 1b 86 b6 05  	addiw	a2, a3, 91
   15a3a: 0e 05        	slli	a0, a0, 3
   15a3c: 0d b7        	j	0x1595e <_free_r+0xf8>
   15a3e: 98 f5        	sd	a4, 40(a1)
   15a40: 98 f1        	sd	a4, 32(a1)
   15a42: 93 e6 17 00  	ori	a3, a5, 1
   15a46: 08 ef        	sd	a0, 24(a4)
   15a48: 08 eb        	sd	a0, 16(a4)
   15a4a: 14 e7        	sd	a3, 8(a4)
   15a4c: 3e 97        	add	a4, a4, a5
   15a4e: 1c e3        	sd	a5, 0(a4)
   15a50: e9 b5        	j	0x1591a <_free_r+0xb4>
   15a52: 03 b8 85 00  	ld	a6, 8(a1)
   15a56: 1b 56 26 40  	sraiw	a2, a2, 2
   15a5a: 85 47        	addi	a5, zero, 1
   15a5c: b3 97 c7 00  	sll	a5, a5, a2
   15a60: b3 e7 07 01  	or	a5, a5, a6
   15a64: 9c e5        	sd	a5, 8(a1)
   15a66: 11 bf        	j	0x1597a <_free_r+0x114>
   15a68: 13 06 40 15  	addi	a2, zero, 340
   15a6c: 63 6c d6 00  	bltu	a2, a3, 0x15a84 <_free_r+0x21e>
   15a70: 93 d6 f7 00  	srli	a3, a5, 15
   15a74: 1b 85 86 07  	addiw	a0, a3, 120
   15a78: 1b 15 15 00  	slliw	a0, a0, 1
   15a7c: 1b 86 76 07  	addiw	a2, a3, 119
   15a80: 0e 05        	slli	a0, a0, 3
   15a82: f1 bd        	j	0x1595e <_free_r+0xf8>
   15a84: 13 06 40 55  	addi	a2, zero, 1364
   15a88: 63 6c d6 00  	bltu	a2, a3, 0x15aa0 <_free_r+0x23a>
   15a8c: 93 d6 27 01  	srli	a3, a5, 18
   15a90: 1b 85 d6 07  	addiw	a0, a3, 125
   15a94: 1b 15 15 00  	slliw	a0, a0, 1
   15a98: 1b 86 c6 07  	addiw	a2, a3, 124
   15a9c: 0e 05        	slli	a0, a0, 3
   15a9e: c1 b5        	j	0x1595e <_free_r+0xf8>
   15aa0: 13 05 00 7f  	addi	a0, zero, 2032
   15aa4: 13 06 e0 07  	addi	a2, zero, 126
   15aa8: 5d bd        	j	0x1595e <_free_r+0xf8>
   15aaa: 93 e6 17 00  	ori	a3, a5, 1
   15aae: 14 e7        	sd	a3, 8(a4)
   15ab0: 1c e2        	sd	a5, 0(a2)
   15ab2: a5 b5        	j	0x1591a <_free_r+0xb4>

0000000000015ab4 <_fwalk>:
   15ab4: 5d 71        	addi	sp, sp, -80
   15ab6: 4a f8        	sd	s2, 48(sp)
   15ab8: 4e f4        	sd	s3, 40(sp)
   15aba: 52 f0        	sd	s4, 32(sp)
   15abc: 56 ec        	sd	s5, 24(sp)
   15abe: 5a e8        	sd	s6, 16(sp)
   15ac0: 5e e4        	sd	s7, 8(sp)
   15ac2: 86 e4        	sd	ra, 72(sp)
   15ac4: a2 e0        	sd	s0, 64(sp)
   15ac6: 26 fc        	sd	s1, 56(sp)
   15ac8: ae 8a        	mv	s5, a1
   15aca: 93 09 05 52  	addi	s3, a0, 1312
   15ace: 01 49        	mv	s2, zero
   15ad0: 13 0b 00 0b  	addi	s6, zero, 176
   15ad4: 85 4b        	addi	s7, zero, 1
   15ad6: 7d 5a        	addi	s4, zero, -1
   15ad8: 83 a4 89 00  	lw	s1, 8(s3)
   15adc: 03 b4 09 01  	ld	s0, 16(s3)
   15ae0: 63 58 90 02  	blez	s1, 0x15b10 <_fwalk+0x5c>
   15ae4: 82 14        	slli	s1, s1, 32
   15ae6: 81 90        	srli	s1, s1, 32
   15ae8: b3 84 64 03  	mul	s1, s1, s6
   15aec: a2 94        	add	s1, s1, s0
   15aee: 83 57 04 01  	lhu	a5, 16(s0)
   15af2: 63 fb fb 00  	bgeu	s7, a5, 0x15b08 <_fwalk+0x54>
   15af6: 83 17 24 01  	lh	a5, 18(s0)
   15afa: 22 85        	mv	a0, s0
   15afc: 63 86 47 01  	beq	a5, s4, 0x15b08 <_fwalk+0x54>
   15b00: 82 9a        	jalr	s5
   15b02: 33 69 25 01  	or	s2, a0, s2
   15b06: 01 29        	sext.w	s2, s2
   15b08: 13 04 04 0b  	addi	s0, s0, 176
   15b0c: e3 91 84 fe  	bne	s1, s0, 0x15aee <_fwalk+0x3a>
   15b10: 83 b9 09 00  	ld	s3, 0(s3)
   15b14: e3 92 09 fc  	bnez	s3, 0x15ad8 <_fwalk+0x24>
   15b18: a6 60        	ld	ra, 72(sp)
   15b1a: 06 64        	ld	s0, 64(sp)
   15b1c: e2 74        	ld	s1, 56(sp)
   15b1e: a2 79        	ld	s3, 40(sp)
   15b20: 02 7a        	ld	s4, 32(sp)
   15b22: e2 6a        	ld	s5, 24(sp)
   15b24: 42 6b        	ld	s6, 16(sp)
   15b26: a2 6b        	ld	s7, 8(sp)
   15b28: 4a 85        	mv	a0, s2
   15b2a: 42 79        	ld	s2, 48(sp)
   15b2c: 61 61        	addi	sp, sp, 80
   15b2e: 82 80        	ret

0000000000015b30 <_fwalk_reent>:
   15b30: 5d 71        	addi	sp, sp, -80
   15b32: 4a f8        	sd	s2, 48(sp)
   15b34: 4e f4        	sd	s3, 40(sp)
   15b36: 52 f0        	sd	s4, 32(sp)
   15b38: 56 ec        	sd	s5, 24(sp)
   15b3a: 5a e8        	sd	s6, 16(sp)
   15b3c: 5e e4        	sd	s7, 8(sp)
   15b3e: 62 e0        	sd	s8, 0(sp)
   15b40: 86 e4        	sd	ra, 72(sp)
   15b42: a2 e0        	sd	s0, 64(sp)
   15b44: 26 fc        	sd	s1, 56(sp)
   15b46: aa 89        	mv	s3, a0
   15b48: 2e 8b        	mv	s6, a1
   15b4a: 13 0a 05 52  	addi	s4, a0, 1312
   15b4e: 01 49        	mv	s2, zero
   15b50: 93 0b 00 0b  	addi	s7, zero, 176
   15b54: 05 4c        	addi	s8, zero, 1
   15b56: fd 5a        	addi	s5, zero, -1
   15b58: 83 24 8a 00  	lw	s1, 8(s4)
   15b5c: 03 34 0a 01  	ld	s0, 16(s4)
   15b60: 63 59 90 02  	blez	s1, 0x15b92 <_fwalk_reent+0x62>
   15b64: 82 14        	slli	s1, s1, 32
   15b66: 81 90        	srli	s1, s1, 32
   15b68: b3 84 74 03  	mul	s1, s1, s7
   15b6c: a2 94        	add	s1, s1, s0
   15b6e: 83 57 04 01  	lhu	a5, 16(s0)
   15b72: 63 7c fc 00  	bgeu	s8, a5, 0x15b8a <_fwalk_reent+0x5a>
   15b76: 83 17 24 01  	lh	a5, 18(s0)
   15b7a: a2 85        	mv	a1, s0
   15b7c: 4e 85        	mv	a0, s3
   15b7e: 63 86 57 01  	beq	a5, s5, 0x15b8a <_fwalk_reent+0x5a>
   15b82: 02 9b        	jalr	s6
   15b84: 33 69 25 01  	or	s2, a0, s2
   15b88: 01 29        	sext.w	s2, s2
   15b8a: 13 04 04 0b  	addi	s0, s0, 176
   15b8e: e3 90 84 fe  	bne	s1, s0, 0x15b6e <_fwalk_reent+0x3e>
   15b92: 03 3a 0a 00  	ld	s4, 0(s4)
   15b96: e3 11 0a fc  	bnez	s4, 0x15b58 <_fwalk_reent+0x28>
   15b9a: a6 60        	ld	ra, 72(sp)
   15b9c: 06 64        	ld	s0, 64(sp)
   15b9e: e2 74        	ld	s1, 56(sp)
   15ba0: a2 79        	ld	s3, 40(sp)
   15ba2: 02 7a        	ld	s4, 32(sp)
   15ba4: e2 6a        	ld	s5, 24(sp)
   15ba6: 42 6b        	ld	s6, 16(sp)
   15ba8: a2 6b        	ld	s7, 8(sp)
   15baa: 02 6c        	ld	s8, 0(sp)
   15bac: 4a 85        	mv	a0, s2
   15bae: 42 79        	ld	s2, 48(sp)
   15bb0: 61 61        	addi	sp, sp, 80
   15bb2: 82 80        	ret

0000000000015bb4 <eshdn1>:
   15bb4: 93 06 45 00  	addi	a3, a0, 4
   15bb8: 01 47        	mv	a4, zero
   15bba: 69 05        	addi	a0, a0, 26
   15bbc: 61 78        	lui	a6, 1048568
   15bbe: 83 d7 06 00  	lhu	a5, 0(a3)
   15bc2: 13 f6 17 00  	andi	a2, a5, 1
   15bc6: 19 c2        	beqz	a2, 0x15bcc <eshdn1+0x18>
   15bc8: 13 67 17 00  	ori	a4, a4, 1
   15bcc: 13 76 27 00  	andi	a2, a4, 2
   15bd0: 85 83        	srli	a5, a5, 1
   15bd2: 1b 17 17 00  	slliw	a4, a4, 1
   15bd6: b3 e5 07 01  	or	a1, a5, a6
   15bda: 01 c6        	beqz	a2, 0x15be2 <eshdn1+0x2e>
   15bdc: 93 97 05 03  	slli	a5, a1, 48
   15be0: c1 93        	srli	a5, a5, 48
   15be2: 23 90 f6 00  	sh	a5, 0(a3)
   15be6: 42 17        	slli	a4, a4, 48
   15be8: 89 06        	addi	a3, a3, 2
   15bea: 41 93        	srli	a4, a4, 48
   15bec: e3 19 d5 fc  	bne	a0, a3, 0x15bbe <eshdn1+0xa>
   15bf0: 82 80        	ret

0000000000015bf2 <eshup1>:
   15bf2: 93 06 85 01  	addi	a3, a0, 24
   15bf6: 01 47        	mv	a4, zero
   15bf8: 09 05        	addi	a0, a0, 2
   15bfa: 83 d7 06 00  	lhu	a5, 0(a3)
   15bfe: 13 d6 f7 00  	srli	a2, a5, 15
   15c02: 9b 97 17 00  	slliw	a5, a5, 1
   15c06: 19 c2        	beqz	a2, 0x15c0c <eshup1+0x1a>
   15c08: 13 67 17 00  	ori	a4, a4, 1
   15c0c: 13 76 27 00  	andi	a2, a4, 2
   15c10: c2 17        	slli	a5, a5, 48
   15c12: 1b 17 17 00  	slliw	a4, a4, 1
   15c16: c1 93        	srli	a5, a5, 48
   15c18: 19 c2        	beqz	a2, 0x15c1e <eshup1+0x2c>
   15c1a: 93 e7 17 00  	ori	a5, a5, 1
   15c1e: 23 90 f6 00  	sh	a5, 0(a3)
   15c22: 42 17        	slli	a4, a4, 48
   15c24: f9 16        	addi	a3, a3, -2
   15c26: 41 93        	srli	a4, a4, 48
   15c28: e3 19 d5 fc  	bne	a0, a3, 0x15bfa <eshup1+0x8>
   15c2c: 82 80        	ret

0000000000015c2e <m16m>:
   15c2e: 01 11        	addi	sp, sp, -32
   15c30: 41 6e        	lui	t3, 16
   15c32: 23 1b 01 00  	sh	zero, 22(sp)
   15c36: 23 1c 01 00  	sh	zero, 24(sp)
   15c3a: e1 05        	addi	a1, a1, 24
   15c3c: 38 08        	addi	a4, sp, 24
   15c3e: 13 08 41 00  	addi	a6, sp, 4
   15c42: 7d 1e        	addi	t3, t3, -1
   15c44: 83 d7 05 00  	lhu	a5, 0(a1)
   15c48: 79 17        	addi	a4, a4, -2
   15c4a: f9 15        	addi	a1, a1, -2
   15c4c: 85 c7        	beqz	a5, 0x15c74 <m16m+0x46>
   15c4e: b3 87 a7 02  	mul	a5, a5, a0
   15c52: 83 56 27 00  	lhu	a3, 2(a4)
   15c56: 03 53 07 00  	lhu	t1, 0(a4)
   15c5a: b3 f8 c7 01  	and	a7, a5, t3
   15c5e: c6 96        	add	a3, a3, a7
   15c60: c1 83        	srli	a5, a5, 16
   15c62: 93 d8 06 01  	srli	a7, a3, 16
   15c66: 9a 97        	add	a5, a5, t1
   15c68: c6 97        	add	a5, a5, a7
   15c6a: 23 10 f7 00  	sh	a5, 0(a4)
   15c6e: 23 11 d7 00  	sh	a3, 2(a4)
   15c72: c1 83        	srli	a5, a5, 16
   15c74: 23 1f f7 fe  	sh	a5, -2(a4)
   15c78: e3 16 07 fd  	bne	a4, a6, 0x15c44 <m16m+0x16>
   15c7c: 93 07 46 00  	addi	a5, a2, 4
   15c80: d7 f6 85 c4  	<unknown>
   15c84: ad 45        	addi	a1, zero, 11
   15c86: 07 5c 07 02  	<unknown>
   15c8a: 27 dc 07 02  	<unknown>
   15c8e: b3 87 d5 40  	sub	a5, a1, a3
   15c92: 63 81 d5 02  	beq	a1, a3, 0x15cb4 <m16m+0x86>
   15c96: 73 27 20 c2  	csrr	a4, vlenb
   15c9a: 93 06 47 fe  	addi	a3, a4, -28
   15c9e: 93 86 06 02  	addi	a3, a3, 32
   15ca2: 11 07        	addi	a4, a4, 4
   15ca4: 8a 96        	add	a3, a3, sp
   15ca6: 3a 96        	add	a2, a2, a4
   15ca8: 57 f0 87 04  	<unknown>
   15cac: 07 dc 06 02  	<unknown>
   15cb0: 27 5c 06 02  	<unknown>
   15cb4: 05 61        	addi	sp, sp, 32
   15cb6: 82 80        	ret

0000000000015cb8 <emovo.constprop.0>:
   15cb8: 03 57 05 00  	lhu	a4, 0(a0)
   15cbc: 83 57 25 00  	lhu	a5, 2(a0)
   15cc0: 19 c3        	beqz	a4, 0x15cc6 <emovo.constprop.0+0xe>
   15cc2: 21 67        	lui	a4, 8
   15cc4: d9 8f        	or	a5, a5, a4
   15cc6: 23 99 f5 00  	sh	a5, 18(a1)
   15cca: 03 57 25 00  	lhu	a4, 2(a0)
   15cce: a1 67        	lui	a5, 8
   15cd0: fd 17        	addi	a5, a5, -1
   15cd2: 63 0f f7 00  	beq	a4, a5, 0x15cf0 <emovo.constprop.0+0x38>
   15cd6: 93 07 65 00  	addi	a5, a0, 6
   15cda: c1 05        	addi	a1, a1, 16
   15cdc: 61 05        	addi	a0, a0, 24
   15cde: 03 d7 07 00  	lhu	a4, 0(a5)
   15ce2: 89 07        	addi	a5, a5, 2
   15ce4: f9 15        	addi	a1, a1, -2
   15ce6: 23 91 e5 00  	sh	a4, 2(a1)
   15cea: e3 9a a7 fe  	bne	a5, a0, 0x15cde <emovo.constprop.0+0x26>
   15cee: 82 80        	ret
   15cf0: 93 07 65 00  	addi	a5, a0, 6
   15cf4: 69 05        	addi	a0, a0, 26
   15cf6: 03 d7 07 00  	lhu	a4, 0(a5)
   15cfa: 89 07        	addi	a5, a5, 2
   15cfc: 29 e3        	bnez	a4, 0x15d3e <emovo.constprop.0+0x86>
   15cfe: e3 9c a7 fe  	bne	a5, a0, 0x15cf6 <emovo.constprop.0+0x3e>
   15d02: d7 f7 84 c4  	<unknown>
   15d06: 57 77 80 04  	<unknown>
   15d0a: 25 47        	addi	a4, zero, 9
   15d0c: 57 3c 00 5e  	<unknown>
   15d10: 57 f0 84 c4  	<unknown>
   15d14: 27 dc 05 02  	<unknown>
   15d18: b3 06 f7 40  	sub	a3, a4, a5
   15d1c: 63 09 f7 00  	beq	a4, a5, 0x15d2e <emovo.constprop.0+0x76>
   15d20: f3 27 20 c2  	csrr	a5, vlenb
   15d24: ae 97        	add	a5, a5, a1
   15d26: 57 f0 86 04  	<unknown>
   15d2a: 27 dc 07 02  	<unknown>
   15d2e: 83 d7 25 01  	lhu	a5, 18(a1)
   15d32: 21 67        	lui	a4, 8
   15d34: 7d 17        	addi	a4, a4, -1
   15d36: d9 8f        	or	a5, a5, a4
   15d38: 23 99 f5 00  	sh	a5, 18(a1)
   15d3c: 82 80        	ret
   15d3e: d7 77 80 04  	<unknown>
   15d42: f1 77        	lui	a5, 1048572
   15d44: 57 3c 00 5e  	<unknown>
   15d48: 57 70 84 c4  	<unknown>
   15d4c: 27 dc 05 02  	<unknown>
   15d50: 23 98 f5 00  	sh	a5, 16(a1)
   15d54: e1 77        	lui	a5, 1048568
   15d56: 93 c7 f7 ff  	not	a5, a5
   15d5a: 23 99 f5 00  	sh	a5, 18(a1)
   15d5e: 82 80        	ret

0000000000015d60 <enormlz>:
   15d60: 83 57 45 00  	lhu	a5, 4(a0)
   15d64: 01 11        	addi	sp, sp, -32
   15d66: 26 e4        	sd	s1, 8(sp)
   15d68: 06 ec        	sd	ra, 24(sp)
   15d6a: 22 e8        	sd	s0, 16(sp)
   15d6c: 4a e0        	sd	s2, 0(sp)
   15d6e: aa 84        	mv	s1, a0
   15d70: ad e7        	bnez	a5, 0x15dda <enormlz+0x7a>
   15d72: 83 57 65 00  	lhu	a5, 6(a0)
   15d76: 01 44        	mv	s0, zero
   15d78: 13 d7 f7 00  	srli	a4, a5, 15
   15d7c: 21 eb        	bnez	a4, 0x15dcc <enormlz+0x6c>
   15d7e: 73 27 20 c2  	csrr	a4, vlenb
   15d82: 13 08 67 00  	addi	a6, a4, 6
   15d86: 13 06 47 00  	addi	a2, a4, 4
   15d8a: d7 75 85 c4  	<unknown>
   15d8e: 29 47        	addi	a4, zero, 10
   15d90: 93 06 45 00  	addi	a3, a0, 4
   15d94: 0d 8f        	sub	a4, a4, a1
   15d96: 19 05        	addi	a0, a0, 6
   15d98: 26 98        	add	a6, a6, s1
   15d9a: 26 96        	add	a2, a2, s1
   15d9c: 93 05 00 0a  	addi	a1, zero, 160
   15da0: 19 a0        	j	0x15da6 <enormlz+0x46>
   15da2: 83 d7 64 00  	lhu	a5, 6(s1)
   15da6: 57 70 85 c4  	<unknown>
   15daa: c1 eb        	bnez	a5, 0x15e3a <enormlz+0xda>
   15dac: 07 5c 05 02  	<unknown>
   15db0: 27 dc 06 02  	<unknown>
   15db4: 41 24        	addiw	s0, s0, 16
   15db6: 19 c7        	beqz	a4, 0x15dc4 <enormlz+0x64>
   15db8: 57 70 87 04  	<unknown>
   15dbc: 07 5c 08 02  	<unknown>
   15dc0: 27 5c 06 02  	<unknown>
   15dc4: 23 9c 04 00  	sh	zero, 24(s1)
   15dc8: e3 1d b4 fc  	bne	s0, a1, 0x15da2 <enormlz+0x42>
   15dcc: e2 60        	ld	ra, 24(sp)
   15dce: 22 85        	mv	a0, s0
   15dd0: 42 64        	ld	s0, 16(sp)
   15dd2: a2 64        	ld	s1, 8(sp)
   15dd4: 02 69        	ld	s2, 0(sp)
   15dd6: 05 61        	addi	sp, sp, 32
   15dd8: 82 80        	ret
   15dda: 13 07 f0 0f  	addi	a4, zero, 255
   15dde: 01 44        	mv	s0, zero
   15de0: 63 65 f7 02  	bltu	a4, a5, 0x15e0a <enormlz+0xaa>
   15de4: 13 09 f0 f6  	addi	s2, zero, -145
   15de8: 01 a8        	j	0x15df8 <enormlz+0x98>
   15dea: 7d 34        	addiw	s0, s0, -1
   15dec: ef f0 9f dc  	jal	0x15bb4 <eshdn1>
   15df0: e3 0e 24 fd  	beq	s0, s2, 0x15dcc <enormlz+0x6c>
   15df4: 83 d7 44 00  	lhu	a5, 4(s1)
   15df8: 26 85        	mv	a0, s1
   15dfa: e5 fb        	bnez	a5, 0x15dea <enormlz+0x8a>
   15dfc: e2 60        	ld	ra, 24(sp)
   15dfe: 22 85        	mv	a0, s0
   15e00: 42 64        	ld	s0, 16(sp)
   15e02: a2 64        	ld	s1, 8(sp)
   15e04: 02 69        	ld	s2, 0(sp)
   15e06: 05 61        	addi	sp, sp, 32
   15e08: 82 80        	ret
   15e0a: 13 07 45 00  	addi	a4, a0, 4
   15e0e: 93 05 a5 01  	addi	a1, a0, 26
   15e12: 81 47        	mv	a5, zero
   15e14: 83 56 07 00  	lhu	a3, 0(a4)
   15e18: 09 07        	addi	a4, a4, 2
   15e1a: 1b d6 86 00  	srliw	a2, a3, 8
   15e1e: d1 8f        	or	a5, a5, a2
   15e20: 9b 96 86 00  	slliw	a3, a3, 8
   15e24: 23 1f f7 fe  	sh	a5, -2(a4)
   15e28: 93 97 06 03  	slli	a5, a3, 48
   15e2c: c1 93        	srli	a5, a5, 48
   15e2e: e3 13 b7 fe  	bne	a4, a1, 0x15e14 <enormlz+0xb4>
   15e32: 83 d7 44 00  	lhu	a5, 4(s1)
   15e36: 61 54        	addi	s0, zero, -8
   15e38: 75 b7        	j	0x15de4 <enormlz+0x84>
   15e3a: 13 07 f0 0f  	addi	a4, zero, 255
   15e3e: 63 6b f7 02  	bltu	a4, a5, 0x15e74 <enormlz+0x114>
   15e42: 13 88 84 01  	addi	a6, s1, 24
   15e46: 93 85 24 00  	addi	a1, s1, 2
   15e4a: 13 05 f0 0f  	addi	a0, zero, 255
   15e4e: 81 47        	mv	a5, zero
   15e50: 42 87        	mv	a4, a6
   15e52: 83 56 07 00  	lhu	a3, 0(a4)
   15e56: 79 17        	addi	a4, a4, -2
   15e58: 1b 96 86 00  	slliw	a2, a3, 8
   15e5c: d1 8f        	or	a5, a5, a2
   15e5e: 23 11 f7 00  	sh	a5, 2(a4)
   15e62: 93 d7 86 00  	srli	a5, a3, 8
   15e66: e3 16 b7 fe  	bne	a4, a1, 0x15e52 <enormlz+0xf2>
   15e6a: 83 d7 64 00  	lhu	a5, 6(s1)
   15e6e: 21 24        	addiw	s0, s0, 8
   15e70: e3 7f f5 fc  	bgeu	a0, a5, 0x15e4e <enormlz+0xee>
   15e74: 13 09 00 0a  	addi	s2, zero, 160
   15e78: 01 a8        	j	0x15e88 <enormlz+0x128>
   15e7a: 05 24        	addiw	s0, s0, 1
   15e7c: ef f0 7f d7  	jal	0x15bf2 <eshup1>
   15e80: e3 46 89 f4  	blt	s2, s0, 0x15dcc <enormlz+0x6c>
   15e84: 83 d7 64 00  	lhu	a5, 6(s1)
   15e88: bd 83        	srli	a5, a5, 15
   15e8a: 26 85        	mv	a0, s1
   15e8c: fd d7        	beqz	a5, 0x15e7a <enormlz+0x11a>
   15e8e: 3d bf        	j	0x15dcc <enormlz+0x6c>

0000000000015e90 <eshift.part.0>:
   15e90: 79 71        	addi	sp, sp, -48
   15e92: 22 f0        	sd	s0, 32(sp)
   15e94: 26 ec        	sd	s1, 24(sp)
   15e96: 06 f4        	sd	ra, 40(sp)
   15e98: 4a e8        	sd	s2, 16(sp)
   15e9a: 4e e4        	sd	s3, 8(sp)
   15e9c: ae 84        	mv	s1, a1
   15e9e: 2a 84        	mv	s0, a0
   15ea0: 63 cc 05 08  	bltz	a1, 0x15f38 <eshift.part.0+0xa8>
   15ea4: bd 47        	addi	a5, zero, 15
   15ea6: 63 d6 b7 04  	bge	a5, a1, 0x15ef2 <eshift.part.0+0x62>
   15eaa: f3 27 20 c2  	csrr	a5, vlenb
   15eae: 13 88 67 00  	addi	a6, a5, 6
   15eb2: 13 86 47 00  	addi	a2, a5, 4
   15eb6: 29 47        	addi	a4, zero, 10
   15eb8: d7 77 85 c4  	<unknown>
   15ebc: 93 06 45 00  	addi	a3, a0, 4
   15ec0: 1d 8f        	sub	a4, a4, a5
   15ec2: 19 05        	addi	a0, a0, 6
   15ec4: ae 87        	mv	a5, a1
   15ec6: 22 98        	add	a6, a6, s0
   15ec8: 22 96        	add	a2, a2, s0
   15eca: bd 45        	addi	a1, zero, 15
   15ecc: 57 70 85 c4  	<unknown>
   15ed0: 07 5c 05 02  	<unknown>
   15ed4: 27 dc 06 02  	<unknown>
   15ed8: c1 37        	addiw	a5, a5, -16
   15eda: 19 c7        	beqz	a4, 0x15ee8 <eshift.part.0+0x58>
   15edc: 57 70 87 04  	<unknown>
   15ee0: 07 5c 08 02  	<unknown>
   15ee4: 27 5c 06 02  	<unknown>
   15ee8: 23 1c 04 00  	sh	zero, 24(s0)
   15eec: e3 c0 f5 fe  	blt	a1, a5, 0x15ecc <eshift.part.0+0x3c>
   15ef0: bd 88        	andi	s1, s1, 15
   15ef2: 9d 47        	addi	a5, zero, 7
   15ef4: 63 d4 97 02  	bge	a5, s1, 0x15f1c <eshift.part.0+0x8c>
   15ef8: 13 07 84 01  	addi	a4, s0, 24
   15efc: 93 05 24 00  	addi	a1, s0, 2
   15f00: 81 47        	mv	a5, zero
   15f02: 83 56 07 00  	lhu	a3, 0(a4)
   15f06: 79 17        	addi	a4, a4, -2
   15f08: 1b 96 86 00  	slliw	a2, a3, 8
   15f0c: d1 8f        	or	a5, a5, a2
   15f0e: 23 11 f7 00  	sh	a5, 2(a4)
   15f12: 93 d7 86 00  	srli	a5, a3, 8
   15f16: e3 16 b7 fe  	bne	a4, a1, 0x15f02 <eshift.part.0+0x72>
   15f1a: e1 34        	addiw	s1, s1, -8
   15f1c: 91 c4        	beqz	s1, 0x15f28 <eshift.part.0+0x98>
   15f1e: fd 34        	addiw	s1, s1, -1
   15f20: 22 85        	mv	a0, s0
   15f22: ef f0 1f cd  	jal	0x15bf2 <eshup1>
   15f26: e5 fc        	bnez	s1, 0x15f1e <eshift.part.0+0x8e>
   15f28: 01 45        	mv	a0, zero
   15f2a: a2 70        	ld	ra, 40(sp)
   15f2c: 02 74        	ld	s0, 32(sp)
   15f2e: e2 64        	ld	s1, 24(sp)
   15f30: 42 69        	ld	s2, 16(sp)
   15f32: a2 69        	ld	s3, 8(sp)
   15f34: 45 61        	addi	sp, sp, 48
   15f36: 82 80        	ret
   15f38: c5 57        	addi	a5, zero, -15
   15f3a: 3b 09 b0 40  	negw	s2, a1
   15f3e: 81 49        	mv	s3, zero
   15f40: 93 05 85 01  	addi	a1, a0, 24
   15f44: 93 06 45 00  	addi	a3, a0, 4
   15f48: 3d 46        	addi	a2, zero, 15
   15f4a: 63 d4 f4 0a  	bge	s1, a5, 0x15ff2 <eshift.part.0+0x162>
   15f4e: 03 57 84 01  	lhu	a4, 24(s0)
   15f52: ae 87        	mv	a5, a1
   15f54: b3 e9 e9 00  	or	s3, s3, a4
   15f58: 03 d7 e7 ff  	lhu	a4, -2(a5)
   15f5c: f9 17        	addi	a5, a5, -2
   15f5e: 23 91 e7 00  	sh	a4, 2(a5)
   15f62: e3 9b d7 fe  	bne	a5, a3, 0x15f58 <eshift.part.0+0xc8>
   15f66: 23 12 04 00  	sh	zero, 4(s0)
   15f6a: 41 39        	addiw	s2, s2, -16
   15f6c: e3 41 26 ff  	blt	a2, s2, 0x15f4e <eshift.part.0+0xbe>
   15f70: 41 59        	addi	s2, zero, -16
   15f72: 45 57        	addi	a4, zero, -15
   15f74: 3b 09 99 40  	subw	s2, s2, s1
   15f78: 81 47        	mv	a5, zero
   15f7a: 63 c7 e4 06  	blt	s1, a4, 0x15fe8 <eshift.part.0+0x158>
   15f7e: 3b 09 f9 00  	addw	s2, s2, a5
   15f82: 9d 47        	addi	a5, zero, 7
   15f84: 63 d9 27 03  	bge	a5, s2, 0x15fb6 <eshift.part.0+0x126>
   15f88: 83 47 84 01  	lbu	a5, 24(s0)
   15f8c: 93 05 a4 01  	addi	a1, s0, 26
   15f90: b3 e9 f9 00  	or	s3, s3, a5
   15f94: 81 47        	mv	a5, zero
   15f96: 03 d7 06 00  	lhu	a4, 0(a3)
   15f9a: 89 06        	addi	a3, a3, 2
   15f9c: 1b 56 87 00  	srliw	a2, a4, 8
   15fa0: d1 8f        	or	a5, a5, a2
   15fa2: 1b 17 87 00  	slliw	a4, a4, 8
   15fa6: 23 9f f6 fe  	sh	a5, -2(a3)
   15faa: 93 17 07 03  	slli	a5, a4, 48
   15fae: c1 93        	srli	a5, a5, 48
   15fb0: e3 93 b6 fe  	bne	a3, a1, 0x15f96 <eshift.part.0+0x106>
   15fb4: 61 39        	addiw	s2, s2, -8
   15fb6: 33 35 30 01  	snez	a0, s3
   15fba: e3 08 09 f6  	beqz	s2, 0x15f2a <eshift.part.0+0x9a>
   15fbe: 83 57 84 01  	lhu	a5, 24(s0)
   15fc2: 7d 39        	addiw	s2, s2, -1
   15fc4: 22 85        	mv	a0, s0
   15fc6: 85 8b        	andi	a5, a5, 1
   15fc8: b3 e9 f9 00  	or	s3, s3, a5
   15fcc: ef f0 9f be  	jal	0x15bb4 <eshdn1>
   15fd0: e3 17 09 fe  	bnez	s2, 0x15fbe <eshift.part.0+0x12e>
   15fd4: a2 70        	ld	ra, 40(sp)
   15fd6: 02 74        	ld	s0, 32(sp)
   15fd8: 81 29        	sext.w	s3, s3
   15fda: e2 64        	ld	s1, 24(sp)
   15fdc: 42 69        	ld	s2, 16(sp)
   15fde: 33 35 30 01  	snez	a0, s3
   15fe2: a2 69        	ld	s3, 8(sp)
   15fe4: 45 61        	addi	sp, sp, 48
   15fe6: 82 80        	ret
   15fe8: 93 77 09 ff  	andi	a5, s2, -16
   15fec: bb 07 f0 40  	negw	a5, a5
   15ff0: 79 b7        	j	0x15f7e <eshift.part.0+0xee>
   15ff2: e5 57        	addi	a5, zero, -7
   15ff4: e3 ca f4 f8  	blt	s1, a5, 0x15f88 <eshift.part.0+0xf8>
   15ff8: d9 b7        	j	0x15fbe <eshift.part.0+0x12e>

0000000000015ffa <emovi>:
   15ffa: 83 57 25 01  	lhu	a5, 18(a0)
   15ffe: a1 66        	lui	a3, 8
   16000: fd 16        	addi	a3, a3, -1
   16002: 9b d7 f7 00  	srliw	a5, a5, 15
   16006: bb 07 f0 40  	negw	a5, a5
   1600a: 23 90 f5 00  	sh	a5, 0(a1)
   1600e: 03 57 25 01  	lhu	a4, 18(a0)
   16012: 93 07 05 01  	addi	a5, a0, 16
   16016: 75 8f        	and	a4, a4, a3
   16018: 23 91 e5 00  	sh	a4, 2(a1)
   1601c: 1b 06 07 00  	sext.w	a2, a4
   16020: 63 02 d7 02  	beq	a4, a3, 0x16044 <emovi+0x4a>
   16024: 13 87 65 00  	addi	a4, a1, 6
   16028: 23 92 05 00  	sh	zero, 4(a1)
   1602c: 79 15        	addi	a0, a0, -2
   1602e: 83 d6 07 00  	lhu	a3, 0(a5)
   16032: f9 17        	addi	a5, a5, -2
   16034: 09 07        	addi	a4, a4, 2
   16036: 23 1f d7 fe  	sh	a3, -2(a4)
   1603a: e3 1a f5 fe  	bne	a0, a5, 0x1602e <emovi+0x34>
   1603e: 23 9c 05 00  	sh	zero, 24(a1)
   16042: 82 80        	ret
   16044: 03 57 25 01  	lhu	a4, 18(a0)
   16048: 71 8f        	and	a4, a4, a2
   1604a: 63 1b c7 00  	bne	a4, a2, 0x16060 <emovi+0x66>
   1604e: 13 06 25 01  	addi	a2, a0, 18
   16052: 2a 87        	mv	a4, a0
   16054: 83 56 07 00  	lhu	a3, 0(a4)
   16058: 09 07        	addi	a4, a4, 2
   1605a: 95 ee        	bnez	a3, 0x16096 <emovi+0x9c>
   1605c: e3 1c e6 fe  	bne	a2, a4, 0x16054 <emovi+0x5a>
   16060: 57 f7 85 c4  	<unknown>
   16064: d7 77 80 04  	<unknown>
   16068: 93 87 45 00  	addi	a5, a1, 4
   1606c: 57 3c 00 5e  	<unknown>
   16070: ad 46        	addi	a3, zero, 11
   16072: 57 f0 85 c4  	<unknown>
   16076: 27 dc 07 02  	<unknown>
   1607a: 33 86 e6 40  	sub	a2, a3, a4
   1607e: 63 8b e6 00  	beq	a3, a4, 0x16094 <emovi+0x9a>
   16082: f3 27 20 c2  	csrr	a5, vlenb
   16086: 91 07        	addi	a5, a5, 4
   16088: ae 97        	add	a5, a5, a1
   1608a: 57 70 86 04  	<unknown>
   1608e: 27 dc 07 02  	<unknown>
   16092: 82 80        	ret
   16094: 82 80        	ret
   16096: 13 87 65 00  	addi	a4, a1, 6
   1609a: 23 92 05 00  	sh	zero, 4(a1)
   1609e: 71 15        	addi	a0, a0, -4
   160a0: 83 d6 07 00  	lhu	a3, 0(a5)
   160a4: f9 17        	addi	a5, a5, -2
   160a6: 09 07        	addi	a4, a4, 2
   160a8: 23 1f d7 fe  	sh	a3, -2(a4)
   160ac: e3 1a f5 fe  	bne	a0, a5, 0x160a0 <emovi+0xa6>
   160b0: 82 80        	ret

00000000000160b2 <ecmp>:
   160b2: 83 57 25 01  	lhu	a5, 18(a0)
   160b6: 5d 71        	addi	sp, sp, -80
   160b8: a2 e0        	sd	s0, 64(sp)
   160ba: 93 c7 f7 ff  	not	a5, a5
   160be: 86 e4        	sd	ra, 72(sp)
   160c0: 13 97 17 03  	slli	a4, a5, 49
   160c4: 2e 84        	mv	s0, a1
   160c6: 11 eb        	bnez	a4, 0x160da <ecmp+0x28>
   160c8: 93 06 25 01  	addi	a3, a0, 18
   160cc: aa 87        	mv	a5, a0
   160ce: 03 d7 07 00  	lhu	a4, 0(a5)
   160d2: 89 07        	addi	a5, a5, 2
   160d4: 25 eb        	bnez	a4, 0x16144 <ecmp+0x92>
   160d6: e3 9c d7 fe  	bne	a5, a3, 0x160ce <ecmp+0x1c>
   160da: 83 57 24 01  	lhu	a5, 18(s0)
   160de: 93 c7 f7 ff  	not	a5, a5
   160e2: 13 97 17 03  	slli	a4, a5, 49
   160e6: 31 c3        	beqz	a4, 0x1612a <ecmp+0x78>
   160e8: 8a 85        	mv	a1, sp
   160ea: ef f0 1f f1  	jal	0x15ffa <emovi>
   160ee: 0c 10        	addi	a1, sp, 32
   160f0: 22 85        	mv	a0, s0
   160f2: ef f0 9f f0  	jal	0x15ffa <emovi>
   160f6: 83 55 01 00  	lhu	a1, 0(sp)
   160fa: 83 57 01 02  	lhu	a5, 32(sp)
   160fe: 63 88 b7 04  	beq	a5, a1, 0x1614e <ecmp+0x9c>
   16102: 93 07 21 00  	addi	a5, sp, 2
   16106: 13 07 21 02  	addi	a4, sp, 34
   1610a: 30 08        	addi	a2, sp, 24
   1610c: 83 d6 07 00  	lhu	a3, 0(a5)
   16110: 89 07        	addi	a5, a5, 2
   16112: b5 ea        	bnez	a3, 0x16186 <ecmp+0xd4>
   16114: 83 56 07 00  	lhu	a3, 0(a4)
   16118: 09 07        	addi	a4, a4, 2
   1611a: b5 e6        	bnez	a3, 0x16186 <ecmp+0xd4>
   1611c: e3 18 f6 fe  	bne	a2, a5, 0x1610c <ecmp+0x5a>
   16120: 01 45        	mv	a0, zero
   16122: a6 60        	ld	ra, 72(sp)
   16124: 06 64        	ld	s0, 64(sp)
   16126: 61 61        	addi	sp, sp, 80
   16128: 82 80        	ret
   1612a: a2 87        	mv	a5, s0
   1612c: 03 d7 07 00  	lhu	a4, 0(a5)
   16130: 93 06 24 01  	addi	a3, s0, 18
   16134: 89 07        	addi	a5, a5, 2
   16136: 19 e7        	bnez	a4, 0x16144 <ecmp+0x92>
   16138: e3 88 d7 fa  	beq	a5, a3, 0x160e8 <ecmp+0x36>
   1613c: 03 d7 07 00  	lhu	a4, 0(a5)
   16140: 89 07        	addi	a5, a5, 2
   16142: 7d db        	beqz	a4, 0x16138 <ecmp+0x86>
   16144: a6 60        	ld	ra, 72(sp)
   16146: 06 64        	ld	s0, 64(sp)
   16148: 79 55        	addi	a0, zero, -2
   1614a: 61 61        	addi	sp, sp, 80
   1614c: 82 80        	ret
   1614e: 7d 55        	addi	a0, zero, -1
   16150: a9 c1        	beqz	a1, 0x16192 <ecmp+0xe0>
   16152: 8a 87        	mv	a5, sp
   16154: 03 d6 07 00  	lhu	a2, 0(a5)
   16158: 83 56 01 02  	lhu	a3, 32(sp)
   1615c: 18 10        	addi	a4, sp, 32
   1615e: 2c 08        	addi	a1, sp, 24
   16160: 89 07        	addi	a5, a5, 2
   16162: 09 07        	addi	a4, a4, 2
   16164: 63 1c d6 00  	bne	a2, a3, 0x1617c <ecmp+0xca>
   16168: e3 8c b7 fa  	beq	a5, a1, 0x16120 <ecmp+0x6e>
   1616c: 03 d6 07 00  	lhu	a2, 0(a5)
   16170: 83 56 07 00  	lhu	a3, 0(a4)
   16174: 89 07        	addi	a5, a5, 2
   16176: 09 07        	addi	a4, a4, 2
   16178: e3 08 d6 fe  	beq	a2, a3, 0x16168 <ecmp+0xb6>
   1617c: e3 e3 c6 fa  	bltu	a3, a2, 0x16122 <ecmp+0x70>
   16180: 3b 05 a0 40  	negw	a0, a0
   16184: 79 bf        	j	0x16122 <ecmp+0x70>
   16186: 81 e9        	bnez	a1, 0x16196 <ecmp+0xe4>
   16188: a6 60        	ld	ra, 72(sp)
   1618a: 06 64        	ld	s0, 64(sp)
   1618c: 05 45        	addi	a0, zero, 1
   1618e: 61 61        	addi	sp, sp, 80
   16190: 82 80        	ret
   16192: 05 45        	addi	a0, zero, 1
   16194: 7d bf        	j	0x16152 <ecmp+0xa0>
   16196: 7d 55        	addi	a0, zero, -1
   16198: 69 b7        	j	0x16122 <ecmp+0x70>

000000000001619a <emdnorm>:
   1619a: 39 71        	addi	sp, sp, -64
   1619c: 22 f8        	sd	s0, 48(sp)
   1619e: 26 f4        	sd	s1, 40(sp)
   161a0: 4a f0        	sd	s2, 32(sp)
   161a2: 4e ec        	sd	s3, 24(sp)
   161a4: 52 e8        	sd	s4, 16(sp)
   161a6: 56 e4        	sd	s5, 8(sp)
   161a8: b6 84        	mv	s1, a3
   161aa: 3e 84        	mv	s0, a5
   161ac: 06 fc        	sd	ra, 56(sp)
   161ae: 2a 8a        	mv	s4, a0
   161b0: 2e 89        	mv	s2, a1
   161b2: b2 89        	mv	s3, a2
   161b4: ba 8a        	mv	s5, a4
   161b6: ef f0 bf ba  	jal	0x15d60 <enormlz>
   161ba: 93 07 00 09  	addi	a5, zero, 144
   161be: 89 8c        	sub	s1, s1, a0
   161c0: 63 d7 a7 16  	bge	a5, a0, 0x1632e <emdnorm+0x194>
   161c4: a1 67        	lui	a5, 8
   161c6: f9 17        	addi	a5, a5, -2
   161c8: 63 d2 97 1a  	bge	a5, s1, 0x1636c <emdnorm+0x1d2>
   161cc: 63 8c 0a 20  	beqz	s5, 0x163e4 <emdnorm+0x24a>
   161d0: 83 2a 44 00  	lw	s5, 4(s0)
   161d4: 1c 40        	lw	a5, 0(s0)
   161d6: 63 84 fa 28  	beq	s5, a5, 0x1645e <emdnorm+0x2c4>
   161da: 57 f7 86 c4  	<unknown>
   161de: d7 77 80 04  	<unknown>
   161e2: 93 07 a4 01  	addi	a5, s0, 26
   161e6: 57 3c 00 5e  	<unknown>
   161ea: b5 46        	addi	a3, zero, 13
   161ec: 57 f0 86 c4  	<unknown>
   161f0: 27 dc 07 02  	<unknown>
   161f4: 33 86 e6 40  	sub	a2, a3, a4
   161f8: 63 96 e6 28  	bne	a3, a4, 0x16484 <emdnorm+0x2ea>
   161fc: 93 07 80 03  	addi	a5, zero, 56
   16200: 63 89 fa 34  	beq	s5, a5, 0x16552 <emdnorm+0x3b8>
   16204: 63 de 57 19  	bge	a5, s5, 0x163a0 <emdnorm+0x206>
   16208: 93 07 00 04  	addi	a5, zero, 64
   1620c: 63 85 fa 28  	beq	s5, a5, 0x16496 <emdnorm+0x2fc>
   16210: 93 07 10 07  	addi	a5, zero, 113
   16214: 63 9e fa 30  	bne	s5, a5, 0x16530 <emdnorm+0x396>
   16218: b7 87 00 40  	lui	a5, 262152
   1621c: a9 46        	addi	a3, zero, 10
   1621e: 61 77        	lui	a4, 1048568
   16220: fd 17        	addi	a5, a5, -1
   16222: 14 c4        	sw	a3, 8(s0)
   16224: 5c c8        	sw	a5, 20(s0)
   16226: 23 1c e4 00  	sh	a4, 24(s0)
   1622a: 54 c4        	sw	a3, 12(s0)
   1622c: 23 17 e4 02  	sh	a4, 46(s0)
   16230: 23 20 54 01  	sw	s5, 0(s0)
   16234: 63 5b 90 20  	blez	s1, 0x1644a <emdnorm+0x2b0>
   16238: 03 55 4a 01  	lhu	a0, 20(s4)
   1623c: 13 06 4a 01  	addi	a2, s4, 20
   16240: ad 47        	addi	a5, zero, 11
   16242: 46 15        	slli	a0, a0, 49
   16244: 45 91        	srli	a0, a0, 49
   16246: 35 47        	addi	a4, zero, 13
   16248: 1d 9f        	subw	a4, a4, a5
   1624a: 02 17        	slli	a4, a4, 32
   1624c: 86 07        	slli	a5, a5, 1
   1624e: 01 93        	srli	a4, a4, 32
   16250: d2 97        	add	a5, a5, s4
   16252: f3 25 20 c2  	csrr	a1, vlenb
   16256: d7 76 80 04  	<unknown>
   1625a: 57 3c 00 5e  	<unknown>
   1625e: d7 3c 80 9f  	<unknown>
   16262: d7 76 87 04  	<unknown>
   16266: 07 d0 07 02  	<unknown>
   1626a: 15 8f        	sub	a4, a4, a3
   1626c: a7 dc 07 02  	<unknown>
   16270: 57 30 00 66  	<unknown>
   16274: ae 97        	add	a5, a5, a1
   16276: 57 bd 90 5d  	<unknown>
   1627a: 57 70 80 00  	<unknown>
   1627e: 57 0c 8d 2b  	<unknown>
   16282: 65 f3        	bnez	a4, 0x16262 <emdnorm+0xc8>
   16284: d7 77 80 00  	<unknown>
   16288: 03 57 06 00  	lhu	a4, 0(a2)
   1628c: d7 6c 00 42  	<unknown>
   16290: 83 56 44 01  	lhu	a3, 20(s0)
   16294: d7 ac 8c 0b  	<unknown>
   16298: 57 70 80 04  	<unknown>
   1629c: d7 27 90 43  	<unknown>
   162a0: c2 17        	slli	a5, a5, 48
   162a2: c1 93        	srli	a5, a5, 48
   162a4: 5d 8d        	or	a0, a0, a5
   162a6: 93 c6 f6 ff  	not	a3, a3
   162aa: 75 8f        	and	a4, a4, a3
   162ac: 23 10 e6 00  	sh	a4, 0(a2)
   162b0: 83 57 64 01  	lhu	a5, 22(s0)
   162b4: 33 f7 a7 00  	and	a4, a5, a0
   162b8: 21 c7        	beqz	a4, 0x16300 <emdnorm+0x166>
   162ba: 63 9d a7 00  	bne	a5, a0, 0x162d4 <emdnorm+0x13a>
   162be: 63 16 09 26  	bnez	s2, 0x1652a <emdnorm+0x390>
   162c2: 5c 44        	lw	a5, 12(s0)
   162c4: 03 57 84 01  	lhu	a4, 24(s0)
   162c8: 86 07        	slli	a5, a5, 1
   162ca: d2 97        	add	a5, a5, s4
   162cc: 83 d7 07 00  	lhu	a5, 0(a5)
   162d0: f9 8f        	and	a5, a5, a4
   162d2: 9d c7        	beqz	a5, 0x16300 <emdnorm+0x166>
   162d4: 93 07 24 03  	addi	a5, s0, 50
   162d8: 13 06 8a 01  	addi	a2, s4, 24
   162dc: 13 05 2a 00  	addi	a0, s4, 2
   162e0: 81 45        	mv	a1, zero
   162e2: 03 d7 07 00  	lhu	a4, 0(a5)
   162e6: 83 56 06 00  	lhu	a3, 0(a2)
   162ea: 79 16        	addi	a2, a2, -2
   162ec: f9 17        	addi	a5, a5, -2
   162ee: 36 97        	add	a4, a4, a3
   162f0: 2e 97        	add	a4, a4, a1
   162f2: 93 55 07 01  	srli	a1, a4, 16
   162f6: 23 11 e6 00  	sh	a4, 2(a2)
   162fa: 85 89        	andi	a1, a1, 1
   162fc: e3 13 a6 fe  	bne	a2, a0, 0x162e2 <emdnorm+0x148>
   16300: 63 5c 90 1e  	blez	s1, 0x164f8 <emdnorm+0x35e>
   16304: 83 57 4a 00  	lhu	a5, 4(s4)
   16308: 63 9d 07 1c  	bnez	a5, 0x164e2 <emdnorm+0x348>
   1630c: a1 67        	lui	a5, 8
   1630e: 23 1c 0a 00  	sh	zero, 24(s4)
   16312: f9 17        	addi	a5, a5, -2
   16314: 63 ca 97 0c  	blt	a5, s1, 0x163e8 <emdnorm+0x24e>
   16318: 23 11 9a 00  	sh	s1, 2(s4)
   1631c: e2 70        	ld	ra, 56(sp)
   1631e: 42 74        	ld	s0, 48(sp)
   16320: a2 74        	ld	s1, 40(sp)
   16322: 02 79        	ld	s2, 32(sp)
   16324: e2 69        	ld	s3, 24(sp)
   16326: 42 6a        	ld	s4, 16(sp)
   16328: a2 6a        	ld	s5, 8(sp)
   1632a: 21 61        	addi	sp, sp, 64
   1632c: 82 80        	ret
   1632e: 63 cb 04 02  	bltz	s1, 0x16364 <emdnorm+0x1ca>
   16332: e3 8d 0a fc  	beqz	s5, 0x1630c <emdnorm+0x172>
   16336: 83 2a 44 00  	lw	s5, 4(s0)
   1633a: 1c 40        	lw	a5, 0(s0)
   1633c: e3 9f fa e8  	bne	s5, a5, 0x161da <emdnorm+0x40>
   16340: 63 44 90 26  	bgtz	s1, 0x165a8 <emdnorm+0x40e>
   16344: 13 07 00 09  	addi	a4, zero, 144
   16348: 63 92 e7 26  	bne	a5, a4, 0x165ac <emdnorm+0x412>
   1634c: 10 44        	lw	a2, 8(s0)
   1634e: 83 56 44 01  	lhu	a3, 20(s0)
   16352: 93 0a 00 09  	addi	s5, zero, 144
   16356: 06 06        	slli	a2, a2, 1
   16358: 52 96        	add	a2, a2, s4
   1635a: 03 57 06 00  	lhu	a4, 0(a2)
   1635e: 33 75 d7 00  	and	a0, a4, a3
   16362: 91 b7        	j	0x162a6 <emdnorm+0x10c>
   16364: 93 07 00 f7  	addi	a5, zero, -144
   16368: 63 de f4 14  	bge	s1, a5, 0x164c4 <emdnorm+0x32a>
   1636c: 57 77 86 c4  	<unknown>
   16370: d7 77 80 04  	<unknown>
   16374: 93 07 2a 00  	addi	a5, s4, 2
   16378: 57 3c 00 5e  	<unknown>
   1637c: b1 46        	addi	a3, zero, 12
   1637e: 57 70 86 c4  	<unknown>
   16382: 27 dc 07 02  	<unknown>
   16386: 33 86 e6 40  	sub	a2, a3, a4
   1638a: e3 89 e6 f8  	beq	a3, a4, 0x1631c <emdnorm+0x182>
   1638e: f3 27 20 c2  	csrr	a5, vlenb
   16392: 89 07        	addi	a5, a5, 2
   16394: d2 97        	add	a5, a5, s4
   16396: 57 70 86 04  	<unknown>
   1639a: 27 dc 07 02  	<unknown>
   1639e: bd bf        	j	0x1631c <emdnorm+0x182>
   163a0: e1 47        	addi	a5, zero, 24
   163a2: 63 82 fa 08  	beq	s5, a5, 0x16426 <emdnorm+0x28c>
   163a6: 93 07 50 03  	addi	a5, zero, 53
   163aa: 63 93 fa 18  	bne	s5, a5, 0x16530 <emdnorm+0x396>
   163ae: 85 67        	lui	a5, 1
   163b0: 37 07 00 04  	lui	a4, 16384
   163b4: 99 46        	addi	a3, zero, 6
   163b6: 93 87 07 80  	addi	a5, a5, -2048
   163ba: 13 07 f7 7f  	addi	a4, a4, 2047
   163be: 14 c4        	sw	a3, 8(s0)
   163c0: 58 c8        	sw	a4, 20(s0)
   163c2: 23 1c f4 00  	sh	a5, 24(s0)
   163c6: 54 c4        	sw	a3, 12(s0)
   163c8: 23 13 f4 02  	sh	a5, 38(s0)
   163cc: 23 20 54 01  	sw	s5, 0(s0)
   163d0: 63 5d 90 06  	blez	s1, 0x1644a <emdnorm+0x2b0>
   163d4: 03 55 ca 00  	lhu	a0, 12(s4)
   163d8: 13 06 ca 00  	addi	a2, s4, 12
   163dc: 9d 47        	addi	a5, zero, 7
   163de: 13 75 f5 7f  	andi	a0, a0, 2047
   163e2: 95 b5        	j	0x16246 <emdnorm+0xac>
   163e4: 23 1c 0a 00  	sh	zero, 24(s4)
   163e8: 57 77 85 c4  	<unknown>
   163ec: d7 77 80 04  	<unknown>
   163f0: e1 77        	lui	a5, 1048568
   163f2: 93 c7 f7 ff  	not	a5, a5
   163f6: 23 11 fa 00  	sh	a5, 2(s4)
   163fa: 93 07 4a 00  	addi	a5, s4, 4
   163fe: 57 3c 00 5e  	<unknown>
   16402: a9 46        	addi	a3, zero, 10
   16404: 57 70 85 c4  	<unknown>
   16408: 27 dc 07 02  	<unknown>
   1640c: 33 86 e6 40  	sub	a2, a3, a4
   16410: e3 86 e6 f0  	beq	a3, a4, 0x1631c <emdnorm+0x182>
   16414: f3 27 20 c2  	csrr	a5, vlenb
   16418: 91 07        	addi	a5, a5, 4
   1641a: d2 97        	add	a5, a5, s4
   1641c: 57 70 86 04  	<unknown>
   16420: 27 dc 07 02  	<unknown>
   16424: e5 bd        	j	0x1631c <emdnorm+0x182>
   16426: b7 07 80 00  	lui	a5, 2048
   1642a: 91 46        	addi	a3, zero, 4
   1642c: 13 07 00 10  	addi	a4, zero, 256
   16430: 93 87 f7 0f  	addi	a5, a5, 255
   16434: 14 c4        	sw	a3, 8(s0)
   16436: 5c c8        	sw	a5, 20(s0)
   16438: 23 1c e4 00  	sh	a4, 24(s0)
   1643c: 54 c4        	sw	a3, 12(s0)
   1643e: 23 11 e4 02  	sh	a4, 34(s0)
   16442: 23 20 54 01  	sw	s5, 0(s0)
   16446: 63 46 90 14  	bgtz	s1, 0x16592 <emdnorm+0x3f8>
   1644a: 83 57 8a 01  	lhu	a5, 24(s4)
   1644e: 52 85        	mv	a0, s4
   16450: 85 8b        	andi	a5, a5, 1
   16452: b3 67 f9 00  	or	a5, s2, a5
   16456: 1b 89 07 00  	sext.w	s2, a5
   1645a: ef f0 af f5  	jal	0x15bb4 <eshdn1>
   1645e: 1c 44        	lw	a5, 8(s0)
   16460: 83 56 44 01  	lhu	a3, 20(s0)
   16464: 93 05 f0 08  	addi	a1, zero, 143
   16468: 13 96 17 00  	slli	a2, a5, 1
   1646c: 52 96        	add	a2, a2, s4
   1646e: 03 57 06 00  	lhu	a4, 0(a2)
   16472: 33 75 d7 00  	and	a0, a4, a3
   16476: e3 c8 55 e3  	blt	a1, s5, 0x162a6 <emdnorm+0x10c>
   1647a: 85 27        	addiw	a5, a5, 1
   1647c: b1 45        	addi	a1, zero, 12
   1647e: e3 d4 f5 dc  	bge	a1, a5, 0x16246 <emdnorm+0xac>
   16482: 15 b5        	j	0x162a6 <emdnorm+0x10c>
   16484: f3 27 20 c2  	csrr	a5, vlenb
   16488: e9 07        	addi	a5, a5, 26
   1648a: a2 97        	add	a5, a5, s0
   1648c: 57 70 86 04  	<unknown>
   16490: 27 dc 07 02  	<unknown>
   16494: a5 b3        	j	0x161fc <emdnorm+0x62>
   16496: 9d 47        	addi	a5, zero, 7
   16498: 1c c4        	sw	a5, 8(s0)
   1649a: b7 07 01 80  	lui	a5, 524304
   1649e: fd 17        	addi	a5, a5, -1
   164a0: 05 47        	addi	a4, zero, 1
   164a2: 5c c8        	sw	a5, 20(s0)
   164a4: 99 47        	addi	a5, zero, 6
   164a6: 23 1c e4 00  	sh	a4, 24(s0)
   164aa: 5c c4        	sw	a5, 12(s0)
   164ac: 23 13 e4 02  	sh	a4, 38(s0)
   164b0: 23 20 54 01  	sw	s5, 0(s0)
   164b4: e3 5b 90 f8  	blez	s1, 0x1644a <emdnorm+0x2b0>
   164b8: 03 55 ea 00  	lhu	a0, 14(s4)
   164bc: 13 06 ea 00  	addi	a2, s4, 14
   164c0: a1 47        	addi	a5, zero, 8
   164c2: 51 b3        	j	0x16246 <emdnorm+0xac>
   164c4: 9b 85 04 00  	sext.w	a1, s1
   164c8: 52 85        	mv	a0, s4
   164ca: ef f0 7f 9c  	jal	0x15e90 <eshift.part.0>
   164ce: 55 c9        	beqz	a0, 0x16582 <emdnorm+0x3e8>
   164d0: 63 87 0a 0c  	beqz	s5, 0x1659e <emdnorm+0x404>
   164d4: 83 2a 44 00  	lw	s5, 4(s0)
   164d8: 1c 40        	lw	a5, 0(s0)
   164da: 05 49        	addi	s2, zero, 1
   164dc: e3 9f fa ce  	bne	s5, a5, 0x161da <emdnorm+0x40>
   164e0: 95 b5        	j	0x16344 <emdnorm+0x1aa>
   164e2: 52 85        	mv	a0, s4
   164e4: ef f0 0f ed  	jal	0x15bb4 <eshdn1>
   164e8: a1 67        	lui	a5, 8
   164ea: 85 04        	addi	s1, s1, 1
   164ec: 23 1c 0a 00  	sh	zero, 24(s4)
   164f0: f9 17        	addi	a5, a5, -2
   164f2: e3 cb 97 ee  	blt	a5, s1, 0x163e8 <emdnorm+0x24e>
   164f6: 0d b5        	j	0x16318 <emdnorm+0x17e>
   164f8: 93 07 00 09  	addi	a5, zero, 144
   164fc: 63 8f fa 00  	beq	s5, a5, 0x1651a <emdnorm+0x380>
   16500: 52 85        	mv	a0, s4
   16502: ef f0 0f ef  	jal	0x15bf2 <eshup1>
   16506: 83 57 4a 00  	lhu	a5, 4(s4)
   1650a: 99 eb        	bnez	a5, 0x16520 <emdnorm+0x386>
   1650c: 23 1c 0a 00  	sh	zero, 24(s4)
   16510: e3 d4 04 e0  	bgez	s1, 0x16318 <emdnorm+0x17e>
   16514: 23 11 0a 00  	sh	zero, 2(s4)
   16518: 11 b5        	j	0x1631c <emdnorm+0x182>
   1651a: 83 57 4a 00  	lhu	a5, 4(s4)
   1651e: fd d7        	beqz	a5, 0x1650c <emdnorm+0x372>
   16520: 52 85        	mv	a0, s4
   16522: ef f0 2f e9  	jal	0x15bb4 <eshdn1>
   16526: 85 04        	addi	s1, s1, 1
   16528: d5 b7        	j	0x1650c <emdnorm+0x372>
   1652a: e3 85 09 da  	beqz	s3, 0x162d4 <emdnorm+0x13a>
   1652e: c9 bb        	j	0x16300 <emdnorm+0x166>
   16530: b1 47        	addi	a5, zero, 12
   16532: 1c c4        	sw	a5, 8(s0)
   16534: b7 07 01 80  	lui	a5, 524304
   16538: fd 17        	addi	a5, a5, -1
   1653a: 05 47        	addi	a4, zero, 1
   1653c: 5c c8        	sw	a5, 20(s0)
   1653e: ad 47        	addi	a5, zero, 11
   16540: 5c c4        	sw	a5, 12(s0)
   16542: 23 1c e4 00  	sh	a4, 24(s0)
   16546: 23 18 e4 02  	sh	a4, 48(s0)
   1654a: 23 20 54 01  	sw	s5, 0(s0)
   1654e: d6 87        	mv	a5, s5
   16550: c5 bb        	j	0x16340 <emdnorm+0x1a6>
   16552: b7 07 80 00  	lui	a5, 2048
   16556: 99 46        	addi	a3, zero, 6
   16558: 13 07 00 10  	addi	a4, zero, 256
   1655c: 93 87 f7 0f  	addi	a5, a5, 255
   16560: 14 c4        	sw	a3, 8(s0)
   16562: 5c c8        	sw	a5, 20(s0)
   16564: 23 1c e4 00  	sh	a4, 24(s0)
   16568: 54 c4        	sw	a3, 12(s0)
   1656a: 23 13 e4 02  	sh	a4, 38(s0)
   1656e: 23 20 54 01  	sw	s5, 0(s0)
   16572: e3 5c 90 ec  	blez	s1, 0x1644a <emdnorm+0x2b0>
   16576: 03 45 ca 00  	lbu	a0, 12(s4)
   1657a: 13 06 ca 00  	addi	a2, s4, 12
   1657e: 9d 47        	addi	a5, zero, 7
   16580: d9 b1        	j	0x16246 <emdnorm+0xac>
   16582: 63 8e 0a 00  	beqz	s5, 0x1659e <emdnorm+0x404>
   16586: 83 2a 44 00  	lw	s5, 4(s0)
   1658a: 1c 40        	lw	a5, 0(s0)
   1658c: e3 97 fa c4  	bne	s5, a5, 0x161da <emdnorm+0x40>
   16590: 55 bb        	j	0x16344 <emdnorm+0x1aa>
   16592: 03 45 8a 00  	lbu	a0, 8(s4)
   16596: 13 06 8a 00  	addi	a2, s4, 8
   1659a: 95 47        	addi	a5, zero, 5
   1659c: 6d b1        	j	0x16246 <emdnorm+0xac>
   1659e: 23 1c 0a 00  	sh	zero, 24(s4)
   165a2: 23 11 0a 00  	sh	zero, 2(s4)
   165a6: 9d bb        	j	0x1631c <emdnorm+0x182>
   165a8: be 8a        	mv	s5, a5
   165aa: 55 bd        	j	0x1645e <emdnorm+0x2c4>
   165ac: be 8a        	mv	s5, a5
   165ae: 71 bd        	j	0x1644a <emdnorm+0x2b0>

00000000000165b0 <eiremain>:
   165b0: 1d 71        	addi	sp, sp, -96
   165b2: 4e fc        	sd	s3, 56(sp)
   165b4: 83 59 25 00  	lhu	s3, 2(a0)
   165b8: 86 ec        	sd	ra, 88(sp)
   165ba: a2 e8        	sd	s0, 80(sp)
   165bc: a6 e4        	sd	s1, 72(sp)
   165be: ca e0        	sd	s2, 64(sp)
   165c0: b2 84        	mv	s1, a2
   165c2: 2e 89        	mv	s2, a1
   165c4: 52 f8        	sd	s4, 48(sp)
   165c6: 5e ec        	sd	s7, 24(sp)
   165c8: 56 f4        	sd	s5, 40(sp)
   165ca: 5a f0        	sd	s6, 32(sp)
   165cc: 62 e8        	sd	s8, 16(sp)
   165ce: 66 e4        	sd	s9, 8(sp)
   165d0: 6a e0        	sd	s10, 0(sp)
   165d2: aa 8b        	mv	s7, a0
   165d4: ef f0 cf f8  	jal	0x15d60 <enormlz>
   165d8: aa 87        	mv	a5, a0
   165da: 4a 85        	mv	a0, s2
   165dc: 03 54 29 00  	lhu	s0, 2(s2)
   165e0: b3 89 f9 40  	sub	s3, s3, a5
   165e4: ef f0 cf f7  	jal	0x15d60 <enormlz>
   165e8: d7 f7 86 c4  	<unknown>
   165ec: 57 77 80 04  	<unknown>
   165f0: 13 8a 44 03  	addi	s4, s1, 52
   165f4: 57 3c 00 5e  	<unknown>
   165f8: 35 47        	addi	a4, zero, 13
   165fa: 57 f0 86 c4  	<unknown>
   165fe: 27 5c 0a 02  	<unknown>
   16602: 09 8c        	sub	s0, s0, a0
   16604: b3 06 f7 40  	sub	a3, a4, a5
   16608: 63 0b f7 00  	beq	a4, a5, 0x1661e <eiremain+0x6e>
   1660c: f3 27 20 c2  	csrr	a5, vlenb
   16610: 93 87 47 03  	addi	a5, a5, 52
   16614: a6 97        	add	a5, a5, s1
   16616: 57 f0 86 04  	<unknown>
   1661a: 27 dc 07 02  	<unknown>
   1661e: 63 4a 34 07  	blt	s0, s3, 0x16692 <eiremain+0xe2>
   16622: 13 8b 4b 00  	addi	s6, s7, 4
   16626: 93 0a 49 00  	addi	s5, s2, 4
   1662a: 13 8c ab 01  	addi	s8, s7, 26
   1662e: 13 0d 29 00  	addi	s10, s2, 2
   16632: 56 87        	mv	a4, s5
   16634: da 87        	mv	a5, s6
   16636: 03 d6 07 00  	lhu	a2, 0(a5)
   1663a: 83 56 07 00  	lhu	a3, 0(a4)
   1663e: 89 07        	addi	a5, a5, 2
   16640: 09 07        	addi	a4, a4, 2
   16642: 63 1d d6 06  	bne	a2, a3, 0x166bc <eiremain+0x10c>
   16646: e3 18 fc fe  	bne	s8, a5, 0x16636 <eiremain+0x86>
   1664a: 13 86 8b 01  	addi	a2, s7, 24
   1664e: 13 07 89 01  	addi	a4, s2, 24
   16652: 81 46        	mv	a3, zero
   16654: 83 57 07 00  	lhu	a5, 0(a4)
   16658: 83 55 06 00  	lhu	a1, 0(a2)
   1665c: 79 17        	addi	a4, a4, -2
   1665e: 79 16        	addi	a2, a2, -2
   16660: 8d 8f        	sub	a5, a5, a1
   16662: 95 8f        	sub	a5, a5, a3
   16664: 93 d6 07 01  	srli	a3, a5, 16
   16668: 23 11 f7 00  	sh	a5, 2(a4)
   1666c: 85 8a        	andi	a3, a3, 1
   1666e: e3 13 ed fe  	bne	s10, a4, 0x16654 <eiremain+0xa4>
   16672: 85 4c        	addi	s9, zero, 1
   16674: 52 85        	mv	a0, s4
   16676: ef f0 cf d7  	jal	0x15bf2 <eshup1>
   1667a: 83 d7 c4 04  	lhu	a5, 76(s1)
   1667e: 7d 14        	addi	s0, s0, -1
   16680: 4a 85        	mv	a0, s2
   16682: b3 e7 fc 00  	or	a5, s9, a5
   16686: 23 96 f4 04  	sh	a5, 76(s1)
   1668a: ef f0 8f d6  	jal	0x15bf2 <eshup1>
   1668e: e3 52 34 fb  	bge	s0, s3, 0x16632 <eiremain+0x82>
   16692: a2 86        	mv	a3, s0
   16694: 46 64        	ld	s0, 80(sp)
   16696: e6 60        	ld	ra, 88(sp)
   16698: e2 79        	ld	s3, 56(sp)
   1669a: 42 7a        	ld	s4, 48(sp)
   1669c: a2 7a        	ld	s5, 40(sp)
   1669e: 02 7b        	ld	s6, 32(sp)
   166a0: e2 6b        	ld	s7, 24(sp)
   166a2: 42 6c        	ld	s8, 16(sp)
   166a4: a2 6c        	ld	s9, 8(sp)
   166a6: 02 6d        	ld	s10, 0(sp)
   166a8: a6 87        	mv	a5, s1
   166aa: 4a 85        	mv	a0, s2
   166ac: a6 64        	ld	s1, 72(sp)
   166ae: 06 69        	ld	s2, 64(sp)
   166b0: 01 47        	mv	a4, zero
   166b2: 01 46        	mv	a2, zero
   166b4: 81 45        	mv	a1, zero
   166b6: 25 61        	addi	sp, sp, 96
   166b8: 6f f0 3f ae  	j	0x1619a <emdnorm>
   166bc: 81 4c        	mv	s9, zero
   166be: e3 eb c6 fa  	bltu	a3, a2, 0x16674 <eiremain+0xc4>
   166c2: 61 b7        	j	0x1664a <eiremain+0x9a>

00000000000166c4 <e113toe.isra.0>:
   166c4: 39 71        	addi	sp, sp, -64
   166c6: d7 f7 86 c4  	<unknown>
   166ca: 57 77 80 04  	<unknown>
   166ce: 22 f8        	sd	s0, 48(sp)
   166d0: 26 f4        	sd	s1, 40(sp)
   166d2: 06 fc        	sd	ra, 56(sp)
   166d4: 0a 84        	mv	s0, sp
   166d6: 57 3c 00 5e  	<unknown>
   166da: 35 47        	addi	a4, zero, 13
   166dc: 57 f0 86 c4  	<unknown>
   166e0: 27 5c 04 02  	<unknown>
   166e4: ae 84        	mv	s1, a1
   166e6: b3 06 f7 40  	sub	a3, a4, a5
   166ea: 63 09 f7 00  	beq	a4, a5, 0x166fc <e113toe.isra.0+0x38>
   166ee: f3 27 20 c2  	csrr	a5, vlenb
   166f2: 8a 97        	add	a5, a5, sp
   166f4: 57 f0 86 04  	<unknown>
   166f8: 27 dc 07 02  	<unknown>
   166fc: 83 57 e5 00  	lhu	a5, 14(a0)
   16700: 21 67        	lui	a4, 8
   16702: 7d 17        	addi	a4, a4, -1
   16704: 93 96 07 03  	slli	a3, a5, 48
   16708: fd 96        	srai	a3, a3, 63
   1670a: 23 10 d1 00  	sh	a3, 0(sp)
   1670e: b3 f6 e7 00  	and	a3, a5, a4
   16712: 36 86        	mv	a2, a3
   16714: 63 84 e6 04  	beq	a3, a4, 0x1675c <e113toe.isra.0+0x98>
   16718: 93 07 e5 00  	addi	a5, a0, 14
   1671c: 23 11 d1 00  	sh	a3, 2(sp)
   16720: 13 07 61 00  	addi	a4, sp, 6
   16724: 83 d6 e7 ff  	lhu	a3, -2(a5)
   16728: f9 17        	addi	a5, a5, -2
   1672a: 09 07        	addi	a4, a4, 2
   1672c: 23 1f d7 fe  	sh	a3, -2(a4)
   16730: e3 1a f5 fe  	bne	a0, a5, 0x16724 <e113toe.isra.0+0x60>
   16734: 01 ee        	bnez	a2, 0x1674c <e113toe.isra.0+0x88>
   16736: 23 12 01 00  	sh	zero, 4(sp)
   1673a: a6 85        	mv	a1, s1
   1673c: 22 85        	mv	a0, s0
   1673e: ef f0 af d7  	jal	0x15cb8 <emovo.constprop.0>
   16742: e2 70        	ld	ra, 56(sp)
   16744: 42 74        	ld	s0, 48(sp)
   16746: a2 74        	ld	s1, 40(sp)
   16748: 21 61        	addi	sp, sp, 64
   1674a: 82 80        	ret
   1674c: 85 47        	addi	a5, zero, 1
   1674e: fd 55        	addi	a1, zero, -1
   16750: 22 85        	mv	a0, s0
   16752: 23 12 f1 00  	sh	a5, 4(sp)
   16756: ef f0 af f3  	jal	0x15e90 <eshift.part.0>
   1675a: c5 b7        	j	0x1673a <e113toe.isra.0+0x76>
   1675c: aa 87        	mv	a5, a0
   1675e: 93 06 e5 00  	addi	a3, a0, 14
   16762: 03 d7 07 00  	lhu	a4, 0(a5)
   16766: 89 07        	addi	a5, a5, 2
   16768: 59 e7        	bnez	a4, 0x167f6 <e113toe.isra.0+0x132>
   1676a: e3 9c f6 fe  	bne	a3, a5, 0x16762 <e113toe.isra.0+0x9e>
   1676e: 57 77 85 c4  	<unknown>
   16772: d7 77 80 04  	<unknown>
   16776: a9 46        	addi	a3, zero, 10
   16778: 57 3c 00 5e  	<unknown>
   1677c: f3 27 20 c2  	csrr	a5, vlenb
   16780: 57 70 85 c4  	<unknown>
   16784: 27 dc 04 02  	<unknown>
   16788: a6 97        	add	a5, a5, s1
   1678a: 33 86 e6 40  	sub	a2, a3, a4
   1678e: 63 86 e6 00  	beq	a3, a4, 0x1679a <e113toe.isra.0+0xd6>
   16792: 57 70 86 04  	<unknown>
   16796: 27 dc 07 02  	<unknown>
   1679a: 57 f7 84 c4  	<unknown>
   1679e: d7 76 80 04  	<unknown>
   167a2: a5 46        	addi	a3, zero, 9
   167a4: 57 3c 00 5e  	<unknown>
   167a8: 57 f0 84 c4  	<unknown>
   167ac: 27 dc 04 02  	<unknown>
   167b0: 33 86 e6 40  	sub	a2, a3, a4
   167b4: 63 86 e6 00  	beq	a3, a4, 0x167c0 <e113toe.isra.0+0xfc>
   167b8: 57 70 86 04  	<unknown>
   167bc: 27 dc 07 02  	<unknown>
   167c0: 83 d7 24 01  	lhu	a5, 18(s1)
   167c4: 21 67        	lui	a4, 8
   167c6: 7d 17        	addi	a4, a4, -1
   167c8: d9 8f        	or	a5, a5, a4
   167ca: c2 17        	slli	a5, a5, 48
   167cc: c1 93        	srli	a5, a5, 48
   167ce: 23 99 f4 00  	sh	a5, 18(s1)
   167d2: 03 17 e5 00  	lh	a4, 14(a0)
   167d6: e3 56 07 f6  	bgez	a4, 0x16742 <e113toe.isra.0+0x7e>
   167da: 13 86 24 01  	addi	a2, s1, 18
   167de: 26 87        	mv	a4, s1
   167e0: 83 56 07 00  	lhu	a3, 0(a4)
   167e4: 09 07        	addi	a4, a4, 2
   167e6: b1 fe        	bnez	a3, 0x16742 <e113toe.isra.0+0x7e>
   167e8: e3 1c e6 fe  	bne	a2, a4, 0x167e0 <e113toe.isra.0+0x11c>
   167ec: 61 77        	lui	a4, 1048568
   167ee: b9 8f        	xor	a5, a5, a4
   167f0: 23 99 f4 00  	sh	a5, 18(s1)
   167f4: b9 b7        	j	0x16742 <e113toe.isra.0+0x7e>
   167f6: d7 77 80 04  	<unknown>
   167fa: f1 77        	lui	a5, 1048572
   167fc: 57 3c 00 5e  	<unknown>
   16800: 57 70 84 c4  	<unknown>
   16804: 27 dc 04 02  	<unknown>
   16808: 23 98 f4 00  	sh	a5, 16(s1)
   1680c: e1 77        	lui	a5, 1048568
   1680e: 93 c7 f7 ff  	not	a5, a5
   16812: 23 99 f4 00  	sh	a5, 18(s1)
   16816: 35 b7        	j	0x16742 <e113toe.isra.0+0x7e>

0000000000016818 <ediv>:
   16818: 83 57 25 01  	lhu	a5, 18(a0)
   1681c: 6d 71        	addi	sp, sp, -272
   1681e: 22 e2        	sd	s0, 256(sp)
   16820: 93 c7 f7 ff  	not	a5, a5
   16824: a6 fd        	sd	s1, 248(sp)
   16826: ca f9        	sd	s2, 240(sp)
   16828: 6e f5        	sd	s11, 168(sp)
   1682a: 06 e6        	sd	ra, 264(sp)
   1682c: ce f5        	sd	s3, 232(sp)
   1682e: d2 f1        	sd	s4, 224(sp)
   16830: d6 ed        	sd	s5, 216(sp)
   16832: da e9        	sd	s6, 208(sp)
   16834: de e5        	sd	s7, 200(sp)
   16836: e2 e1        	sd	s8, 192(sp)
   16838: 66 fd        	sd	s9, 184(sp)
   1683a: 6a f9        	sd	s10, 176(sp)
   1683c: 13 97 17 03  	slli	a4, a5, 49
   16840: aa 84        	mv	s1, a0
   16842: 2e 89        	mv	s2, a1
   16844: b2 8d        	mv	s11, a2
   16846: 36 84        	mv	s0, a3
   16848: 11 eb        	bnez	a4, 0x1685c <ediv+0x44>
   1684a: 93 06 25 01  	addi	a3, a0, 18
   1684e: aa 87        	mv	a5, a0
   16850: 03 d7 07 00  	lhu	a4, 0(a5)
   16854: 89 07        	addi	a5, a5, 2
   16856: 55 ef        	bnez	a4, 0x16912 <ediv+0xfa>
   16858: e3 9c d7 fe  	bne	a5, a3, 0x16850 <ediv+0x38>
   1685c: 83 57 29 01  	lhu	a5, 18(s2)
   16860: 93 c7 f7 ff  	not	a5, a5
   16864: 13 97 17 03  	slli	a4, a5, 49
   16868: 11 eb        	bnez	a4, 0x1687c <ediv+0x64>
   1686a: 93 06 29 01  	addi	a3, s2, 18
   1686e: ca 87        	mv	a5, s2
   16870: 03 d7 07 00  	lhu	a4, 0(a5)
   16874: 89 07        	addi	a5, a5, 2
   16876: 4d eb        	bnez	a4, 0x16928 <ediv+0x110>
   16878: e3 9c d7 fe  	bne	a5, a3, 0x16870 <ediv+0x58>
   1687c: b7 29 02 00  	lui	s3, 34
   16880: 93 85 09 18  	addi	a1, s3, 384
   16884: 26 85        	mv	a0, s1
   16886: ef f0 df 82  	jal	0x160b2 <ecmp>
   1688a: 63 05 05 16  	beqz	a0, 0x169f4 <ediv+0x1dc>
   1688e: 03 d6 24 01  	lhu	a2, 18(s1)
   16892: a1 67        	lui	a5, 8
   16894: fd 17        	addi	a5, a5, -1
   16896: 33 75 f6 00  	and	a0, a2, a5
   1689a: 63 12 f5 0a  	bne	a0, a5, 0x1693e <ediv+0x126>
   1689e: 93 86 24 01  	addi	a3, s1, 18
   168a2: a6 87        	mv	a5, s1
   168a4: 03 d7 07 00  	lhu	a4, 0(a5)
   168a8: 89 07        	addi	a5, a5, 2
   168aa: 51 eb        	bnez	a4, 0x1693e <ediv+0x126>
   168ac: e3 9c f6 fe  	bne	a3, a5, 0x168a4 <ediv+0x8c>
   168b0: 03 57 29 01  	lhu	a4, 18(s2)
   168b4: 93 47 f7 ff  	not	a5, a4
   168b8: 93 96 17 03  	slli	a3, a5, 49
   168bc: 63 95 06 10  	bnez	a3, 0x169c6 <ediv+0x1ae>
   168c0: 93 06 29 01  	addi	a3, s2, 18
   168c4: ca 87        	mv	a5, s2
   168c6: 83 d5 07 00  	lhu	a1, 0(a5)
   168ca: 89 07        	addi	a5, a5, 2
   168cc: 63 94 05 44  	bnez	a1, 0x16d14 <ediv+0x4fc>
   168d0: e3 9b d7 fe  	bne	a5, a3, 0x168c6 <ediv+0xae>
   168d4: d7 77 80 04  	<unknown>
   168d8: f1 77        	lui	a5, 1048572
   168da: 57 3c 00 5e  	<unknown>
   168de: 57 70 84 c4  	<unknown>
   168e2: 27 dc 0d 02  	<unknown>
   168e6: 23 98 fd 00  	sh	a5, 16(s11)
   168ea: e1 77        	lui	a5, 1048568
   168ec: 93 c7 f7 ff  	not	a5, a5
   168f0: 23 99 fd 00  	sh	a5, 18(s11)
   168f4: b2 60        	ld	ra, 264(sp)
   168f6: 12 64        	ld	s0, 256(sp)
   168f8: ee 74        	ld	s1, 248(sp)
   168fa: 4e 79        	ld	s2, 240(sp)
   168fc: ae 79        	ld	s3, 232(sp)
   168fe: 0e 7a        	ld	s4, 224(sp)
   16900: ee 6a        	ld	s5, 216(sp)
   16902: 4e 6b        	ld	s6, 208(sp)
   16904: ae 6b        	ld	s7, 200(sp)
   16906: 0e 6c        	ld	s8, 192(sp)
   16908: ea 7c        	ld	s9, 184(sp)
   1690a: 4a 7d        	ld	s10, 176(sp)
   1690c: aa 7d        	ld	s11, 168(sp)
   1690e: 51 61        	addi	sp, sp, 272
   16910: 82 80        	ret
   16912: 13 87 44 01  	addi	a4, s1, 20
   16916: 83 d7 04 00  	lhu	a5, 0(s1)
   1691a: 89 04        	addi	s1, s1, 2
   1691c: 89 0d        	addi	s11, s11, 2
   1691e: 23 9f fd fe  	sh	a5, -2(s11)
   16922: e3 9a e4 fe  	bne	s1, a4, 0x16916 <ediv+0xfe>
   16926: f9 b7        	j	0x168f4 <ediv+0xdc>
   16928: 13 87 4d 01  	addi	a4, s11, 20
   1692c: 83 57 09 00  	lhu	a5, 0(s2)
   16930: 89 0d        	addi	s11, s11, 2
   16932: 09 09        	addi	s2, s2, 2
   16934: 23 9f fd fe  	sh	a5, -2(s11)
   16938: e3 9a ed fe  	bne	s11, a4, 0x1692c <ediv+0x114>
   1693c: 65 bf        	j	0x168f4 <ediv+0xdc>
   1693e: 03 57 29 01  	lhu	a4, 18(s2)
   16942: 93 47 f7 ff  	not	a5, a4
   16946: 93 96 17 03  	slli	a3, a5, 49
   1694a: cd ee        	bnez	a3, 0x16a04 <ediv+0x1ec>
   1694c: 81 48        	mv	a7, zero
   1694e: 93 06 29 01  	addi	a3, s2, 18
   16952: ca 87        	mv	a5, s2
   16954: 83 d5 07 00  	lhu	a1, 0(a5)
   16958: 89 07        	addi	a5, a5, 2
   1695a: a5 e5        	bnez	a1, 0x169c2 <ediv+0x1aa>
   1695c: e3 9c d7 fe  	bne	a5, a3, 0x16954 <ediv+0x13c>
   16960: a1 67        	lui	a5, 8
   16962: fd 17        	addi	a5, a5, -1
   16964: 63 09 f5 0c  	beq	a0, a5, 0x16a36 <ediv+0x21e>
   16968: 3d 82        	srli	a2, a2, 15
   1696a: 83 57 09 00  	lhu	a5, 0(s2)
   1696e: 09 09        	addi	s2, s2, 2
   16970: 63 9e 07 38  	bnez	a5, 0x16d0c <ediv+0x4f4>
   16974: e3 1b d9 fe  	bne	s2, a3, 0x1696a <ediv+0x152>
   16978: 93 57 f7 00  	srli	a5, a4, 15
   1697c: b1 8f        	xor	a5, a5, a2
   1697e: 57 f7 84 c4  	<unknown>
   16982: 9b 97 f7 00  	slliw	a5, a5, 15
   16986: d7 76 80 04  	<unknown>
   1698a: 23 99 fd 00  	sh	a5, 18(s11)
   1698e: 57 3c 00 5e  	<unknown>
   16992: a5 47        	addi	a5, zero, 9
   16994: 57 f0 84 c4  	<unknown>
   16998: 27 dc 0d 02  	<unknown>
   1699c: b3 86 e7 40  	sub	a3, a5, a4
   169a0: 63 89 e7 00  	beq	a5, a4, 0x169b2 <ediv+0x19a>
   169a4: f3 27 20 c2  	csrr	a5, vlenb
   169a8: ee 97        	add	a5, a5, s11
   169aa: 57 f0 86 04  	<unknown>
   169ae: 27 dc 07 02  	<unknown>
   169b2: 83 d7 2d 01  	lhu	a5, 18(s11)
   169b6: 21 67        	lui	a4, 8
   169b8: 7d 17        	addi	a4, a4, -1
   169ba: d9 8f        	or	a5, a5, a4
   169bc: 23 99 fd 00  	sh	a5, 18(s11)
   169c0: 15 bf        	j	0x168f4 <ediv+0xdc>
   169c2: 63 81 08 04  	beqz	a7, 0x16a04 <ediv+0x1ec>
   169c6: d7 77 85 c4  	<unknown>
   169ca: 57 77 80 04  	<unknown>
   169ce: 29 47        	addi	a4, zero, 10
   169d0: 57 3c 00 5e  	<unknown>
   169d4: 57 70 85 c4  	<unknown>
   169d8: 27 dc 0d 02  	<unknown>
   169dc: b3 06 f7 40  	sub	a3, a4, a5
   169e0: e3 0a f7 f0  	beq	a4, a5, 0x168f4 <ediv+0xdc>
   169e4: f3 27 20 c2  	csrr	a5, vlenb
   169e8: ee 97        	add	a5, a5, s11
   169ea: 57 f0 86 04  	<unknown>
   169ee: 27 dc 07 02  	<unknown>
   169f2: 09 b7        	j	0x168f4 <ediv+0xdc>
   169f4: 93 85 09 18  	addi	a1, s3, 384
   169f8: 4a 85        	mv	a0, s2
   169fa: ef f0 8f eb  	jal	0x160b2 <ecmp>
   169fe: e3 18 05 e8  	bnez	a0, 0x1688e <ediv+0x76>
   16a02: c9 bd        	j	0x168d4 <ediv+0xbc>
   16a04: 8c 00        	addi	a1, sp, 64
   16a06: 26 85        	mv	a0, s1
   16a08: ef f0 2f df  	jal	0x15ffa <emovi>
   16a0c: 4a 85        	mv	a0, s2
   16a0e: 8c 10        	addi	a1, sp, 96
   16a10: ef f0 af de  	jal	0x15ffa <emovi>
   16a14: 03 5c 21 06  	lhu	s8, 98(sp)
   16a18: 03 59 21 04  	lhu	s2, 66(sp)
   16a1c: 63 17 0c 02  	bnez	s8, 0x16a4a <ediv+0x232>
   16a20: 93 07 21 06  	addi	a5, sp, 98
   16a24: a4 18        	addi	s1, sp, 120
   16a26: 03 d7 07 00  	lhu	a4, 0(a5)
   16a2a: 89 07        	addi	a5, a5, 2
   16a2c: 63 12 07 2c  	bnez	a4, 0x16cf0 <ediv+0x4d8>
   16a30: e3 9b f4 fe  	bne	s1, a5, 0x16a26 <ediv+0x20e>
   16a34: 49 bf        	j	0x169c6 <ediv+0x1ae>
   16a36: 93 85 24 01  	addi	a1, s1, 18
   16a3a: 83 d7 04 00  	lhu	a5, 0(s1)
   16a3e: 89 04        	addi	s1, s1, 2
   16a40: 63 98 07 2c  	bnez	a5, 0x16d10 <ediv+0x4f8>
   16a44: e3 9b 95 fe  	bne	a1, s1, 0x16a3a <ediv+0x222>
   16a48: 05 b7        	j	0x16968 <ediv+0x150>
   16a4a: 62 f0        	sd	s8, 32(sp)
   16a4c: 63 1a 09 02  	bnez	s2, 0x16a80 <ediv+0x268>
   16a50: 93 07 21 04  	addi	a5, sp, 66
   16a54: b4 08        	addi	a3, sp, 88
   16a56: 03 d7 07 00  	lhu	a4, 0(a5)
   16a5a: 89 07        	addi	a5, a5, 2
   16a5c: 63 11 07 2a  	bnez	a4, 0x16cfe <ediv+0x4e6>
   16a60: e3 9b f6 fe  	bne	a3, a5, 0x16a56 <ediv+0x23e>
   16a64: 83 57 01 04  	lhu	a5, 64(sp)
   16a68: 83 56 01 06  	lhu	a3, 96(sp)
   16a6c: 57 f7 84 c4  	<unknown>
   16a70: 57 76 80 04  	<unknown>
   16a74: 95 8f        	sub	a5, a5, a3
   16a76: b3 37 f0 00  	snez	a5, a5
   16a7a: 9b 97 f7 00  	slliw	a5, a5, 15
   16a7e: 31 b7        	j	0x1698a <ediv+0x172>
   16a80: 4a f4        	sd	s2, 40(sp)
   16a82: 06 57        	lw	a4, 96(sp)
   16a84: d7 f7 85 c4  	<unknown>
   16a88: d7 76 80 04  	<unknown>
   16a8c: 58 d8        	sw	a4, 52(s0)
   16a8e: 93 0b 84 03  	addi	s7, s0, 56
   16a92: 93 06 44 03  	addi	a3, s0, 52
   16a96: 57 3c 00 5e  	<unknown>
   16a9a: 2d 47        	addi	a4, zero, 11
   16a9c: 57 f0 85 c4  	<unknown>
   16aa0: 27 dc 0b 02  	<unknown>
   16aa4: 36 f8        	sd	a3, 48(sp)
   16aa6: b3 06 f7 40  	sub	a3, a4, a5
   16aaa: 36 fc        	sd	a3, 56(sp)
   16aac: 63 0b f7 00  	beq	a4, a5, 0x16ac2 <ediv+0x2aa>
   16ab0: f3 27 20 c2  	csrr	a5, vlenb
   16ab4: 93 87 87 03  	addi	a5, a5, 56
   16ab8: a2 97        	add	a5, a5, s0
   16aba: 57 f0 86 04  	<unknown>
   16abe: 27 dc 07 02  	<unknown>
   16ac2: 88 10        	addi	a0, sp, 96
   16ac4: ef f0 0f 8f  	jal	0x15bb4 <eshdn1>
   16ac8: 83 56 61 04  	lhu	a3, 70(sp)
   16acc: f3 27 20 c2  	csrr	a5, vlenb
   16ad0: 57 77 85 c4  	<unknown>
   16ad4: 93 99 06 01  	slli	s3, a3, 16
   16ad8: 29 4f        	addi	t5, zero, 10
   16ada: 13 8a 47 00  	addi	s4, a5, 4
   16ade: 36 e8        	sd	a3, 16(sp)
   16ae0: 99 07        	addi	a5, a5, 6
   16ae2: b3 89 d9 40  	sub	s3, s3, a3
   16ae6: b3 0c ef 40  	sub	s9, t5, a4
   16aea: 94 10        	addi	a3, sp, 96
   16aec: 98 10        	addi	a4, sp, 96
   16aee: d2 96        	add	a3, a3, s4
   16af0: 41 69        	lui	s2, 16
   16af2: ba 97        	add	a5, a5, a4
   16af4: 36 e4        	sd	a3, 8(sp)
   16af6: 13 0b e4 04  	addi	s6, s0, 78
   16afa: 13 0d 21 06  	addi	s10, sp, 98
   16afe: a4 18        	addi	s1, sp, 120
   16b00: 13 0c 41 06  	addi	s8, sp, 100
   16b04: 7d 19        	addi	s2, s2, -1
   16b06: 13 0a a1 09  	addi	s4, sp, 154
   16b0a: 3e ec        	sd	a5, 24(sp)
   16b0c: 83 57 41 06  	lhu	a5, 100(sp)
   16b10: 03 57 61 06  	lhu	a4, 102(sp)
   16b14: ca 8a        	mv	s5, s2
   16b16: c2 07        	slli	a5, a5, 16
   16b18: ba 97        	add	a5, a5, a4
   16b1a: 63 e9 f9 00  	bltu	s3, a5, 0x16b2c <ediv+0x314>
   16b1e: 42 67        	ld	a4, 16(sp)
   16b20: b3 d7 e7 02  	divu	a5, a5, a4
   16b24: 93 9a 07 03  	slli	s5, a5, 48
   16b28: 93 da 0a 03  	srli	s5, s5, 48
   16b2c: 10 01        	addi	a2, sp, 128
   16b2e: 8c 00        	addi	a1, sp, 64
   16b30: 56 85        	mv	a0, s5
   16b32: ef f0 cf 8f  	jal	0x15c2e <m16m>
   16b36: 62 87        	mv	a4, s8
   16b38: 5c 01        	addi	a5, sp, 132
   16b3a: 03 d6 07 00  	lhu	a2, 0(a5)
   16b3e: 83 56 07 00  	lhu	a3, 0(a4)
   16b42: 89 07        	addi	a5, a5, 2
   16b44: 09 07        	addi	a4, a4, 2
   16b46: 63 1f d6 10  	bne	a2, a3, 0x16c64 <ediv+0x44c>
   16b4a: e3 98 47 ff  	bne	a5, s4, 0x16b3a <ediv+0x322>
   16b4e: 81 46        	mv	a3, zero
   16b50: 30 09        	addi	a2, sp, 152
   16b52: 26 87        	mv	a4, s1
   16b54: 83 57 07 00  	lhu	a5, 0(a4)
   16b58: 83 55 06 00  	lhu	a1, 0(a2)
   16b5c: 79 17        	addi	a4, a4, -2
   16b5e: 79 16        	addi	a2, a2, -2
   16b60: 8d 8f        	sub	a5, a5, a1
   16b62: 95 8f        	sub	a5, a5, a3
   16b64: 93 d6 07 01  	srli	a3, a5, 16
   16b68: 23 11 f7 00  	sh	a5, 2(a4)
   16b6c: 85 8a        	andi	a3, a3, 1
   16b6e: e3 13 a7 ff  	bne	a4, s10, 0x16b54 <ediv+0x33c>
   16b72: 23 90 5b 01  	sh	s5, 0(s7)
   16b76: 93 07 61 06  	addi	a5, sp, 102
   16b7a: 57 70 85 c4  	<unknown>
   16b7e: 07 dc 07 02  	<unknown>
   16b82: 27 5c 0c 02  	<unknown>
   16b86: 63 8a 0c 00  	beqz	s9, 0x16b9a <ediv+0x382>
   16b8a: e2 67        	ld	a5, 24(sp)
   16b8c: 57 f0 8c 04  	<unknown>
   16b90: 07 dc 07 02  	<unknown>
   16b94: a2 67        	ld	a5, 8(sp)
   16b96: 27 dc 07 02  	<unknown>
   16b9a: 23 1c 01 06  	sh	zero, 120(sp)
   16b9e: 89 0b        	addi	s7, s7, 2
   16ba0: e3 16 7b f7  	bne	s6, s7, 0x16b0c <ediv+0x2f4>
   16ba4: 62 77        	ld	a4, 56(sp)
   16ba6: 57 f0 85 c4  	<unknown>
   16baa: 87 5c 0c 02  	<unknown>
   16bae: d7 77 80 04  	<unknown>
   16bb2: 57 3c 00 5e  	<unknown>
   16bb6: 57 f0 85 c0  	<unknown>
   16bba: 57 0c 9c 2b  	<unknown>
   16bbe: 09 cf        	beqz	a4, 0x16bd8 <ediv+0x3c0>
   16bc0: f3 27 20 c2  	csrr	a5, vlenb
   16bc4: 93 87 47 02  	addi	a5, a5, 36
   16bc8: 94 00        	addi	a3, sp, 64
   16bca: b6 97        	add	a5, a5, a3
   16bcc: 57 70 87 00  	<unknown>
   16bd0: 87 dc 07 02  	<unknown>
   16bd4: 57 0c 9c 2b  	<unknown>
   16bd8: c2 76        	ld	a3, 48(sp)
   16bda: d7 77 80 04  	<unknown>
   16bde: d7 6c 00 42  	<unknown>
   16be2: d7 ac 8c 0b  	<unknown>
   16be6: d7 25 90 43  	<unknown>
   16bea: d7 f7 86 c4  	<unknown>
   16bee: c2 15        	slli	a1, a1, 48
   16bf0: 07 dc 06 02  	<unknown>
   16bf4: 94 10        	addi	a3, sp, 96
   16bf6: 35 47        	addi	a4, zero, 13
   16bf8: c1 91        	srli	a1, a1, 48
   16bfa: 27 dc 06 02  	<unknown>
   16bfe: b3 35 b0 00  	snez	a1, a1
   16c02: b3 06 f7 40  	sub	a3, a4, a5
   16c06: 63 01 f7 02  	beq	a4, a5, 0x16c28 <ediv+0x410>
   16c0a: f3 27 20 c2  	csrr	a5, vlenb
   16c0e: 13 87 47 03  	addi	a4, a5, 52
   16c12: 90 00        	addi	a2, sp, 64
   16c14: 93 87 07 02  	addi	a5, a5, 32
   16c18: 22 97        	add	a4, a4, s0
   16c1a: b2 97        	add	a5, a5, a2
   16c1c: 57 f0 86 04  	<unknown>
   16c20: 07 5c 07 02  	<unknown>
   16c24: 27 dc 07 02  	<unknown>
   16c28: 82 77        	ld	a5, 32(sp)
   16c2a: 22 77        	ld	a4, 40(sp)
   16c2c: 91 66        	lui	a3, 4
   16c2e: fd 16        	addi	a3, a3, -1
   16c30: 33 8c e7 40  	sub	s8, a5, a4
   16c34: 88 10        	addi	a0, sp, 96
   16c36: a2 87        	mv	a5, s0
   16c38: 13 07 00 04  	addi	a4, zero, 64
   16c3c: e2 96        	add	a3, a3, s8
   16c3e: 01 46        	mv	a2, zero
   16c40: ef f0 af d5  	jal	0x1619a <emdnorm>
   16c44: 03 57 01 06  	lhu	a4, 96(sp)
   16c48: 83 57 01 04  	lhu	a5, 64(sp)
   16c4c: ee 85        	mv	a1, s11
   16c4e: 88 10        	addi	a0, sp, 96
   16c50: 99 8f        	sub	a5, a5, a4
   16c52: b3 37 f0 00  	snez	a5, a5
   16c56: bb 07 f0 40  	negw	a5, a5
   16c5a: 23 10 f1 06  	sh	a5, 96(sp)
   16c5e: ef f0 af 85  	jal	0x15cb8 <emovo.constprop.0>
   16c62: 49 b9        	j	0x168f4 <ediv+0xdc>
   16c64: e3 f5 c6 ee  	bgeu	a3, a2, 0x16b4e <ediv+0x336>
   16c68: 9b 8f fa ff  	addiw	t6, s5, -1
   16c6c: c2 1f        	slli	t6, t6, 48
   16c6e: 93 df 0f 03  	srli	t6, t6, 48
   16c72: 81 46        	mv	a3, zero
   16c74: b0 08        	addi	a2, sp, 88
   16c76: 38 09        	addi	a4, sp, 152
   16c78: 93 05 21 08  	addi	a1, sp, 130
   16c7c: 83 57 07 00  	lhu	a5, 0(a4)
   16c80: 83 52 06 00  	lhu	t0, 0(a2)
   16c84: 79 17        	addi	a4, a4, -2
   16c86: 79 16        	addi	a2, a2, -2
   16c88: b3 87 57 40  	sub	a5, a5, t0
   16c8c: 95 8f        	sub	a5, a5, a3
   16c8e: 93 d6 07 01  	srli	a3, a5, 16
   16c92: 23 11 f7 00  	sh	a5, 2(a4)
   16c96: 85 8a        	andi	a3, a3, 1
   16c98: e3 12 b7 fe  	bne	a4, a1, 0x16c7c <ediv+0x464>
   16c9c: 62 87        	mv	a4, s8
   16c9e: 5c 01        	addi	a5, sp, 132
   16ca0: 03 d6 07 00  	lhu	a2, 0(a5)
   16ca4: 83 56 07 00  	lhu	a3, 0(a4)
   16ca8: 89 07        	addi	a5, a5, 2
   16caa: 09 07        	addi	a4, a4, 2
   16cac: 63 16 d6 00  	bne	a2, a3, 0x16cb8 <ediv+0x4a0>
   16cb0: e3 98 47 ff  	bne	a5, s4, 0x16ca0 <ediv+0x488>
   16cb4: fe 8a        	mv	s5, t6
   16cb6: 61 bd        	j	0x16b4e <ediv+0x336>
   16cb8: e3 fe c6 fe  	bgeu	a3, a2, 0x16cb4 <ediv+0x49c>
   16cbc: 1b 85 ea ff  	addiw	a0, s5, -2
   16cc0: 93 1a 05 03  	slli	s5, a0, 48
   16cc4: 93 da 0a 03  	srli	s5, s5, 48
   16cc8: 81 46        	mv	a3, zero
   16cca: b0 08        	addi	a2, sp, 88
   16ccc: 38 09        	addi	a4, sp, 152
   16cce: 83 57 07 00  	lhu	a5, 0(a4)
   16cd2: 83 5f 06 00  	lhu	t6, 0(a2)
   16cd6: 79 17        	addi	a4, a4, -2
   16cd8: 79 16        	addi	a2, a2, -2
   16cda: b3 87 f7 41  	sub	a5, a5, t6
   16cde: 95 8f        	sub	a5, a5, a3
   16ce0: 93 d6 07 01  	srli	a3, a5, 16
   16ce4: 23 11 f7 00  	sh	a5, 2(a4)
   16ce8: 85 8a        	andi	a3, a3, 1
   16cea: e3 12 b7 fe  	bne	a4, a1, 0x16cce <ediv+0x4b6>
   16cee: 85 b5        	j	0x16b4e <ediv+0x336>
   16cf0: 88 10        	addi	a0, sp, 96
   16cf2: ef f0 ef 86  	jal	0x15d60 <enormlz>
   16cf6: b3 07 a0 40  	neg	a5, a0
   16cfa: 3e f0        	sd	a5, 32(sp)
   16cfc: 81 bb        	j	0x16a4c <ediv+0x234>
   16cfe: 88 00        	addi	a0, sp, 64
   16d00: ef f0 0f 86  	jal	0x15d60 <enormlz>
   16d04: b3 07 a0 40  	neg	a5, a0
   16d08: 3e f4        	sd	a5, 40(sp)
   16d0a: a5 bb        	j	0x16a82 <ediv+0x26a>
   16d0c: 81 47        	mv	a5, zero
   16d0e: bd b1        	j	0x1697c <ediv+0x164>
   16d10: 01 46        	mv	a2, zero
   16d12: a1 b9        	j	0x1696a <ediv+0x152>
   16d14: 85 48        	addi	a7, zero, 1
   16d16: 35 b9        	j	0x16952 <ediv+0x13a>

0000000000016d18 <emul>:
   16d18: 31 71        	addi	sp, sp, -192
   16d1a: 22 f9        	sd	s0, 176(sp)
   16d1c: 2a 84        	mv	s0, a0
   16d1e: 03 55 25 01  	lhu	a0, 18(a0)
   16d22: 21 67        	lui	a4, 8
   16d24: 7d 17        	addi	a4, a4, -1
   16d26: 26 f5        	sd	s1, 168(sp)
   16d28: 4a f1        	sd	s2, 160(sp)
   16d2a: 56 e5        	sd	s5, 136(sp)
   16d2c: 06 fd        	sd	ra, 184(sp)
   16d2e: 4e ed        	sd	s3, 152(sp)
   16d30: 52 e9        	sd	s4, 144(sp)
   16d32: 5a e1        	sd	s6, 128(sp)
   16d34: de fc        	sd	s7, 120(sp)
   16d36: e2 f8        	sd	s8, 112(sp)
   16d38: e6 f4        	sd	s9, 104(sp)
   16d3a: ea f0        	sd	s10, 96(sp)
   16d3c: 33 78 e5 00  	and	a6, a0, a4
   16d40: 2e 89        	mv	s2, a1
   16d42: b2 84        	mv	s1, a2
   16d44: b6 8a        	mv	s5, a3
   16d46: 63 14 e8 14  	bne	a6, a4, 0x16e8e <emul+0x176>
   16d4a: 93 09 24 01  	addi	s3, s0, 18
   16d4e: 22 87        	mv	a4, s0
   16d50: 03 56 07 00  	lhu	a2, 0(a4)
   16d54: 09 07        	addi	a4, a4, 2
   16d56: 63 17 06 1c  	bnez	a2, 0x16f24 <emul+0x20c>
   16d5a: e3 1b 37 ff  	bne	a4, s3, 0x16d50 <emul+0x38>
   16d5e: 03 56 29 01  	lhu	a2, 18(s2)
   16d62: 93 47 f6 ff  	not	a5, a2
   16d66: 13 97 17 03  	slli	a4, a5, 49
   16d6a: 63 09 07 18  	beqz	a4, 0x16efc <emul+0x1e4>
   16d6e: a2 87        	mv	a5, s0
   16d70: 03 d7 07 00  	lhu	a4, 0(a5)
   16d74: 89 07        	addi	a5, a5, 2
   16d76: 63 13 07 20  	bnez	a4, 0x16f7c <emul+0x264>
   16d7a: e3 9b 37 ff  	bne	a5, s3, 0x16d70 <emul+0x58>
   16d7e: b7 25 02 00  	lui	a1, 34
   16d82: 93 85 05 18  	addi	a1, a1, 384
   16d86: 4a 85        	mv	a0, s2
   16d88: ef f0 af b2  	jal	0x160b2 <ecmp>
   16d8c: 63 0c 05 20  	beqz	a0, 0x16fa4 <emul+0x28c>
   16d90: 03 56 29 01  	lhu	a2, 18(s2)
   16d94: a1 65        	lui	a1, 8
   16d96: fd 15        	addi	a1, a1, -1
   16d98: 33 77 b6 00  	and	a4, a2, a1
   16d9c: ba 86        	mv	a3, a4
   16d9e: 93 07 29 01  	addi	a5, s2, 18
   16da2: 63 13 b7 38  	bne	a4, a1, 0x17128 <emul+0x410>
   16da6: 4a 87        	mv	a4, s2
   16da8: 83 56 07 00  	lhu	a3, 0(a4)
   16dac: 09 07        	addi	a4, a4, 2
   16dae: 63 91 06 1a  	bnez	a3, 0x16f50 <emul+0x238>
   16db2: e3 9b e7 fe  	bne	a5, a4, 0x16da8 <emul+0x90>
   16db6: b7 25 02 00  	lui	a1, 34
   16dba: 93 85 05 18  	addi	a1, a1, 384
   16dbe: 22 85        	mv	a0, s0
   16dc0: ef f0 2f af  	jal	0x160b2 <ecmp>
   16dc4: 63 00 05 1e  	beqz	a0, 0x16fa4 <emul+0x28c>
   16dc8: 03 55 24 01  	lhu	a0, 18(s0)
   16dcc: 03 56 29 01  	lhu	a2, 18(s2)
   16dd0: a1 67        	lui	a5, 8
   16dd2: fd 17        	addi	a5, a5, -1
   16dd4: 33 77 f5 00  	and	a4, a0, a5
   16dd8: b3 76 f6 00  	and	a3, a2, a5
   16ddc: ba 85        	mv	a1, a4
   16dde: 63 02 f7 18  	beq	a4, a5, 0x16f62 <emul+0x24a>
   16de2: 36 87        	mv	a4, a3
   16de4: ae 86        	mv	a3, a1
   16de6: a1 67        	lui	a5, 8
   16de8: 01 27        	sext.w	a4, a4
   16dea: fd 17        	addi	a5, a5, -1
   16dec: 63 17 f7 0a  	bne	a4, a5, 0x16e9a <emul+0x182>
   16df0: 93 07 29 01  	addi	a5, s2, 18
   16df4: 4a 87        	mv	a4, s2
   16df6: 83 55 07 00  	lhu	a1, 0(a4)
   16dfa: 09 07        	addi	a4, a4, 2
   16dfc: d9 ed        	bnez	a1, 0x16e9a <emul+0x182>
   16dfe: e3 9c e7 fe  	bne	a5, a4, 0x16df6 <emul+0xde>
   16e02: a1 65        	lui	a1, 8
   16e04: fd 15        	addi	a1, a1, -1
   16e06: 1b 87 06 00  	sext.w	a4, a3
   16e0a: 93 09 24 01  	addi	s3, s0, 18
   16e0e: 63 99 b6 1c  	bne	a3, a1, 0x16fe0 <emul+0x2c8>
   16e12: 83 57 04 00  	lhu	a5, 0(s0)
   16e16: 09 04        	addi	s0, s0, 2
   16e18: 63 9a 07 2e  	bnez	a5, 0x1710c <emul+0x3f4>
   16e1c: e3 9b 89 fe  	bne	s3, s0, 0x16e12 <emul+0xfa>
   16e20: 3d 81        	srli	a0, a0, 15
   16e22: a1 67        	lui	a5, 8
   16e24: fd 17        	addi	a5, a5, -1
   16e26: 63 05 f7 16  	beq	a4, a5, 0x16f90 <emul+0x278>
   16e2a: 3d 82        	srli	a2, a2, 15
   16e2c: b3 47 c5 00  	xor	a5, a0, a2
   16e30: 9b 97 f7 00  	slliw	a5, a5, 15
   16e34: 57 f7 84 c4  	<unknown>
   16e38: d7 76 80 04  	<unknown>
   16e3c: 23 99 f4 00  	sh	a5, 18(s1)
   16e40: 57 3c 00 5e  	<unknown>
   16e44: a5 47        	addi	a5, zero, 9
   16e46: 57 f0 84 c4  	<unknown>
   16e4a: 27 dc 04 02  	<unknown>
   16e4e: b3 86 e7 40  	sub	a3, a5, a4
   16e52: 63 89 e7 00  	beq	a5, a4, 0x16e64 <emul+0x14c>
   16e56: f3 27 20 c2  	csrr	a5, vlenb
   16e5a: a6 97        	add	a5, a5, s1
   16e5c: 57 f0 86 04  	<unknown>
   16e60: 27 dc 07 02  	<unknown>
   16e64: 83 d7 24 01  	lhu	a5, 18(s1)
   16e68: 21 67        	lui	a4, 8
   16e6a: 7d 17        	addi	a4, a4, -1
   16e6c: d9 8f        	or	a5, a5, a4
   16e6e: 23 99 f4 00  	sh	a5, 18(s1)
   16e72: ea 70        	ld	ra, 184(sp)
   16e74: 4a 74        	ld	s0, 176(sp)
   16e76: aa 74        	ld	s1, 168(sp)
   16e78: 0a 79        	ld	s2, 160(sp)
   16e7a: ea 69        	ld	s3, 152(sp)
   16e7c: 4a 6a        	ld	s4, 144(sp)
   16e7e: aa 6a        	ld	s5, 136(sp)
   16e80: 0a 6b        	ld	s6, 128(sp)
   16e82: e6 7b        	ld	s7, 120(sp)
   16e84: 46 7c        	ld	s8, 112(sp)
   16e86: a6 7c        	ld	s9, 104(sp)
   16e88: 06 7d        	ld	s10, 96(sp)
   16e8a: 29 61        	addi	sp, sp, 192
   16e8c: 82 80        	ret
   16e8e: 03 d6 25 01  	lhu	a2, 18(a1)
   16e92: b3 77 e6 00  	and	a5, a2, a4
   16e96: 63 83 e7 06  	beq	a5, a4, 0x16efc <emul+0x1e4>
   16e9a: 22 85        	mv	a0, s0
   16e9c: 8a 85        	mv	a1, sp
   16e9e: ef f0 cf 95  	jal	0x15ffa <emovi>
   16ea2: 93 09 01 02  	addi	s3, sp, 32
   16ea6: ce 85        	mv	a1, s3
   16ea8: 4a 85        	mv	a0, s2
   16eaa: ef f0 0f 95  	jal	0x15ffa <emovi>
   16eae: 03 54 21 00  	lhu	s0, 2(sp)
   16eb2: 03 5a 21 02  	lhu	s4, 34(sp)
   16eb6: 63 18 04 10  	bnez	s0, 0x16fc6 <emul+0x2ae>
   16eba: 93 07 21 00  	addi	a5, sp, 2
   16ebe: 34 08        	addi	a3, sp, 24
   16ec0: 03 d7 07 00  	lhu	a4, 0(a5)
   16ec4: 89 07        	addi	a5, a5, 2
   16ec6: 63 19 07 24  	bnez	a4, 0x17118 <emul+0x400>
   16eca: e3 9b d7 fe  	bne	a5, a3, 0x16ec0 <emul+0x1a8>
   16ece: d7 77 85 c4  	<unknown>
   16ed2: 57 77 80 04  	<unknown>
   16ed6: 29 47        	addi	a4, zero, 10
   16ed8: 57 3c 00 5e  	<unknown>
   16edc: 57 70 85 c4  	<unknown>
   16ee0: 27 dc 04 02  	<unknown>
   16ee4: b3 06 f7 40  	sub	a3, a4, a5
   16ee8: e3 05 f7 f8  	beq	a4, a5, 0x16e72 <emul+0x15a>
   16eec: f3 27 20 c2  	csrr	a5, vlenb
   16ef0: a6 97        	add	a5, a5, s1
   16ef2: 57 f0 86 04  	<unknown>
   16ef6: 27 dc 07 02  	<unknown>
   16efa: a5 bf        	j	0x16e72 <emul+0x15a>
   16efc: 93 07 29 01  	addi	a5, s2, 18
   16f00: 4a 87        	mv	a4, s2
   16f02: 83 56 07 00  	lhu	a3, 0(a4)
   16f06: 09 07        	addi	a4, a4, 2
   16f08: 8d ea        	bnez	a3, 0x16f3a <emul+0x222>
   16f0a: e3 1c f7 fe  	bne	a4, a5, 0x16f02 <emul+0x1ea>
   16f0e: 21 67        	lui	a4, 8
   16f10: 7d 17        	addi	a4, a4, -1
   16f12: 93 09 24 01  	addi	s3, s0, 18
   16f16: e3 0c e8 e4  	beq	a6, a4, 0x16d6e <emul+0x56>
   16f1a: b3 76 e6 00  	and	a3, a2, a4
   16f1e: e3 84 e6 e8  	beq	a3, a4, 0x16da6 <emul+0x8e>
   16f22: a5 bf        	j	0x16e9a <emul+0x182>
   16f24: 13 07 44 01  	addi	a4, s0, 20
   16f28: 83 57 04 00  	lhu	a5, 0(s0)
   16f2c: 09 04        	addi	s0, s0, 2
   16f2e: 89 04        	addi	s1, s1, 2
   16f30: 23 9f f4 fe  	sh	a5, -2(s1)
   16f34: e3 1a e4 fe  	bne	s0, a4, 0x16f28 <emul+0x210>
   16f38: 2d bf        	j	0x16e72 <emul+0x15a>
   16f3a: 13 07 49 01  	addi	a4, s2, 20
   16f3e: 83 57 09 00  	lhu	a5, 0(s2)
   16f42: 09 09        	addi	s2, s2, 2
   16f44: 89 04        	addi	s1, s1, 2
   16f46: 23 9f f4 fe  	sh	a5, -2(s1)
   16f4a: e3 1a e9 fe  	bne	s2, a4, 0x16f3e <emul+0x226>
   16f4e: 15 b7        	j	0x16e72 <emul+0x15a>
   16f50: 03 55 24 01  	lhu	a0, 18(s0)
   16f54: a1 65        	lui	a1, 8
   16f56: fd 15        	addi	a1, a1, -1
   16f58: 33 77 b5 00  	and	a4, a0, a1
   16f5c: ba 86        	mv	a3, a4
   16f5e: e3 1b b7 e8  	bne	a4, a1, 0x16df4 <emul+0xdc>
   16f62: 93 09 24 01  	addi	s3, s0, 18
   16f66: a2 87        	mv	a5, s0
   16f68: 03 d7 07 00  	lhu	a4, 0(a5)
   16f6c: 89 07        	addi	a5, a5, 2
   16f6e: 63 11 07 1a  	bnez	a4, 0x17110 <emul+0x3f8>
   16f72: e3 9b f9 fe  	bne	s3, a5, 0x16f68 <emul+0x250>
   16f76: 1b 87 06 00  	sext.w	a4, a3
   16f7a: 61 bd        	j	0x16e12 <emul+0xfa>
   16f7c: a1 65        	lui	a1, 8
   16f7e: fd 15        	addi	a1, a1, -1
   16f80: 33 77 b6 00  	and	a4, a2, a1
   16f84: ba 86        	mv	a3, a4
   16f86: 93 07 29 01  	addi	a5, s2, 18
   16f8a: e3 0e b7 e0  	beq	a4, a1, 0x16da6 <emul+0x8e>
   16f8e: e1 bf        	j	0x16f66 <emul+0x24e>
   16f90: 93 07 29 01  	addi	a5, s2, 18
   16f94: 03 57 09 00  	lhu	a4, 0(s2)
   16f98: 09 09        	addi	s2, s2, 2
   16f9a: 63 17 07 16  	bnez	a4, 0x17108 <emul+0x3f0>
   16f9e: e3 9b 27 ff  	bne	a5, s2, 0x16f94 <emul+0x27c>
   16fa2: 61 b5        	j	0x16e2a <emul+0x112>
   16fa4: d7 77 80 04  	<unknown>
   16fa8: f1 77        	lui	a5, 1048572
   16faa: 57 3c 00 5e  	<unknown>
   16fae: 57 70 84 c4  	<unknown>
   16fb2: 27 dc 04 02  	<unknown>
   16fb6: 23 98 f4 00  	sh	a5, 16(s1)
   16fba: e1 77        	lui	a5, 1048568
   16fbc: 93 c7 f7 ff  	not	a5, a5
   16fc0: 23 99 f4 00  	sh	a5, 18(s1)
   16fc4: 7d b5        	j	0x16e72 <emul+0x15a>
   16fc6: d2 87        	mv	a5, s4
   16fc8: 52 89        	mv	s2, s4
   16fca: 85 e7        	bnez	a5, 0x16ff2 <emul+0x2da>
   16fcc: 93 07 21 02  	addi	a5, sp, 34
   16fd0: 34 18        	addi	a3, sp, 56
   16fd2: 03 d7 07 00  	lhu	a4, 0(a5)
   16fd6: 89 07        	addi	a5, a5, 2
   16fd8: 11 e7        	bnez	a4, 0x16fe4 <emul+0x2cc>
   16fda: e3 9c f6 fe  	bne	a3, a5, 0x16fd2 <emul+0x2ba>
   16fde: c5 bd        	j	0x16ece <emul+0x1b6>
   16fe0: 3d 81        	srli	a0, a0, 15
   16fe2: 4d bf        	j	0x16f94 <emul+0x27c>
   16fe4: 4e 85        	mv	a0, s3
   16fe6: ef e0 bf d7  	jal	0x15d60 <enormlz>
   16fea: 83 57 21 02  	lhu	a5, 34(sp)
   16fee: 33 09 aa 40  	sub	s2, s4, a0
   16ff2: 83 56 01 02  	lhu	a3, 32(sp)
   16ff6: 57 f7 85 c4  	<unknown>
   16ffa: 23 9b fa 02  	sh	a5, 54(s5)
   16ffe: 23 9a da 02  	sh	a3, 52(s5)
   17002: 57 76 80 04  	<unknown>
   17006: 13 8a 8a 03  	addi	s4, s5, 56
   1700a: 57 3c 00 5e  	<unknown>
   1700e: ad 47        	addi	a5, zero, 11
   17010: 57 f0 85 c4  	<unknown>
   17014: 27 5c 0a 02  	<unknown>
   17018: b3 86 e7 40  	sub	a3, a5, a4
   1701c: 63 8b e7 00  	beq	a5, a4, 0x17032 <emul+0x31a>
   17020: f3 27 20 c2  	csrr	a5, vlenb
   17024: 93 87 87 03  	addi	a5, a5, 56
   17028: d6 97        	add	a5, a5, s5
   1702a: 57 f0 86 04  	<unknown>
   1702e: 27 dc 07 02  	<unknown>
   17032: 93 8c ca 04  	addi	s9, s5, 76
   17036: 01 4c        	mv	s8, zero
   17038: 13 0b 81 01  	addi	s6, sp, 24
   1703c: 13 0d 41 00  	addi	s10, sp, 4
   17040: 93 0b 21 04  	addi	s7, sp, 66
   17044: 03 55 0b 00  	lhu	a0, 0(s6)
   17048: 79 1b        	addi	s6, s6, -2
   1704a: 41 e9        	bnez	a0, 0x170da <emul+0x3c2>
   1704c: 83 d7 ca 04  	lhu	a5, 76(s5)
   17050: 66 87        	mv	a4, s9
   17052: 33 6c fc 00  	or	s8, s8, a5
   17056: 03 56 e7 ff  	lhu	a2, -2(a4)
   1705a: 79 17        	addi	a4, a4, -2
   1705c: 23 11 c7 00  	sh	a2, 2(a4)
   17060: e3 1b ea fe  	bne	s4, a4, 0x17056 <emul+0x33e>
   17064: 23 9c 0a 02  	sh	zero, 56(s5)
   17068: e3 1e ab fd  	bne	s6, s10, 0x17044 <emul+0x32c>
   1706c: 93 87 4a 03  	addi	a5, s5, 52
   17070: 57 f7 86 c4  	<unknown>
   17074: b5 46        	addi	a3, zero, 13
   17076: 07 dc 07 02  	<unknown>
   1707a: 27 dc 09 02  	<unknown>
   1707e: 33 86 e6 40  	sub	a2, a3, a4
   17082: 63 80 e6 02  	beq	a3, a4, 0x170a2 <emul+0x38a>
   17086: f3 27 20 c2  	csrr	a5, vlenb
   1708a: 13 87 47 03  	addi	a4, a5, 52
   1708e: 93 87 07 02  	addi	a5, a5, 32
   17092: 56 97        	add	a4, a4, s5
   17094: 8a 97        	add	a5, a5, sp
   17096: 57 70 86 04  	<unknown>
   1709a: 07 5c 07 02  	<unknown>
   1709e: 27 dc 07 02  	<unknown>
   170a2: f1 76        	lui	a3, 1048572
   170a4: 4a 94        	add	s0, s0, s2
   170a6: 89 06        	addi	a3, a3, 2
   170a8: e2 85        	mv	a1, s8
   170aa: 4e 85        	mv	a0, s3
   170ac: d6 87        	mv	a5, s5
   170ae: 13 07 00 04  	addi	a4, zero, 64
   170b2: a2 96        	add	a3, a3, s0
   170b4: 01 46        	mv	a2, zero
   170b6: ef f0 4f 8e  	jal	0x1619a <emdnorm>
   170ba: 03 57 01 02  	lhu	a4, 32(sp)
   170be: 83 57 01 00  	lhu	a5, 0(sp)
   170c2: a6 85        	mv	a1, s1
   170c4: 4e 85        	mv	a0, s3
   170c6: 99 8f        	sub	a5, a5, a4
   170c8: b3 37 f0 00  	snez	a5, a5
   170cc: bb 07 f0 40  	negw	a5, a5
   170d0: 23 10 f1 02  	sh	a5, 32(sp)
   170d4: ef e0 5f be  	jal	0x15cb8 <emovo.constprop.0>
   170d8: 69 bb        	j	0x16e72 <emul+0x15a>
   170da: 90 00        	addi	a2, sp, 64
   170dc: ce 85        	mv	a1, s3
   170de: ef e0 1f b5  	jal	0x15c2e <m16m>
   170e2: 66 85        	mv	a0, s9
   170e4: 81 45        	mv	a1, zero
   170e6: b0 08        	addi	a2, sp, 88
   170e8: 83 57 05 00  	lhu	a5, 0(a0)
   170ec: 03 57 06 00  	lhu	a4, 0(a2)
   170f0: 79 15        	addi	a0, a0, -2
   170f2: 79 16        	addi	a2, a2, -2
   170f4: 3e 97        	add	a4, a4, a5
   170f6: 2e 97        	add	a4, a4, a1
   170f8: 93 55 07 01  	srli	a1, a4, 16
   170fc: 23 11 e5 00  	sh	a4, 2(a0)
   17100: 85 89        	andi	a1, a1, 1
   17102: e3 13 76 ff  	bne	a2, s7, 0x170e8 <emul+0x3d0>
   17106: 99 b7        	j	0x1704c <emul+0x334>
   17108: 01 46        	mv	a2, zero
   1710a: 0d b3        	j	0x16e2c <emul+0x114>
   1710c: 01 45        	mv	a0, zero
   1710e: 11 bb        	j	0x16e22 <emul+0x10a>
   17110: 36 87        	mv	a4, a3
   17112: a1 66        	lui	a3, 8
   17114: fd 16        	addi	a3, a3, -1
   17116: c1 b9        	j	0x16de6 <emul+0xce>
   17118: 0a 85        	mv	a0, sp
   1711a: ef e0 7f c4  	jal	0x15d60 <enormlz>
   1711e: 83 57 21 02  	lhu	a5, 34(sp)
   17122: 33 04 a0 40  	neg	s0, a0
   17126: 4d b5        	j	0x16fc8 <emul+0x2b0>
   17128: 03 55 24 01  	lhu	a0, 18(s0)
   1712c: b3 77 b5 00  	and	a5, a0, a1
   17130: e3 8b b7 e2  	beq	a5, a1, 0x16f66 <emul+0x24e>
   17134: 9d b3        	j	0x16e9a <emul+0x182>

0000000000017136 <_ldtoa_r>:
   17136: 13 01 01 d2  	addi	sp, sp, -736
   1713a: 13 03 10 09  	addi	t1, zero, 145
   1713e: b2 ed        	sd	a2, 216(sp)
   17140: 03 3e 05 06  	ld	t3, 96(a0)
   17144: 13 16 03 02  	slli	a2, t1, 32
   17148: 7d 16        	addi	a2, a2, -1
   1714a: 23 38 81 2c  	sd	s0, 720(sp)
   1714e: 23 3c 11 2c  	sd	ra, 728(sp)
   17152: 23 34 91 2c  	sd	s1, 712(sp)
   17156: 23 30 21 2d  	sd	s2, 704(sp)
   1715a: 23 3c 31 2b  	sd	s3, 696(sp)
   1715e: 23 38 41 2b  	sd	s4, 688(sp)
   17162: 23 34 51 2b  	sd	s5, 680(sp)
   17166: 23 30 61 2b  	sd	s6, 672(sp)
   1716a: 23 3c 71 29  	sd	s7, 664(sp)
   1716e: 23 38 81 29  	sd	s8, 656(sp)
   17172: 23 34 91 29  	sd	s9, 648(sp)
   17176: 23 30 a1 29  	sd	s10, 640(sp)
   1717a: 23 3c b1 27  	sd	s11, 632(sp)
   1717e: 2a e8        	sd	a0, 16(sp)
   17180: ae e9        	sd	a1, 208(sp)
   17182: 23 30 c1 22  	sd	a2, 544(sp)
   17186: 36 e4        	sd	a3, 8(sp)
   17188: 3a f0        	sd	a4, 32(sp)
   1718a: 3e f8        	sd	a5, 48(sp)
   1718c: c6 e4        	sd	a7, 72(sp)
   1718e: 42 84        	mv	s0, a6
   17190: 63 00 0e 02  	beqz	t3, 0x171b0 <_ldtoa_r+0x7a>
   17194: 38 55        	lw	a4, 104(a0)
   17196: 85 47        	addi	a5, zero, 1
   17198: aa 84        	mv	s1, a0
   1719a: bb 97 e7 00  	sllw	a5, a5, a4
   1719e: f2 85        	mv	a1, t3
   171a0: 23 24 ee 00  	sw	a4, 8(t3)
   171a4: 23 26 fe 00  	sw	a5, 12(t3)
   171a8: ef 10 10 2c  	jal	0x18c68 <_Bfree>
   171ac: 23 b0 04 06  	sd	zero, 96(s1)
   171b0: 8c 11        	addi	a1, sp, 224
   171b2: 88 09        	addi	a0, sp, 208
   171b4: ef f0 0f d1  	jal	0x166c4 <e113toe.isra.0>
   171b8: 03 57 21 0f  	lhu	a4, 242(sp)
   171bc: a1 67        	lui	a5, 8
   171be: fd 17        	addi	a5, a5, -1
   171c0: b3 76 f7 00  	and	a3, a4, a5
   171c4: 36 f4        	sd	a3, 40(sp)
   171c6: 63 9c f6 00  	bne	a3, a5, 0x171de <_ldtoa_r+0xa8>
   171ca: 9c 11        	addi	a5, sp, 224
   171cc: 13 06 21 0f  	addi	a2, sp, 242
   171d0: 83 d6 07 00  	lhu	a3, 0(a5)
   171d4: 89 07        	addi	a5, a5, 2
   171d6: 63 9f 06 3c  	bnez	a3, 0x175b4 <_ldtoa_r+0x47e>
   171da: e3 9b c7 fe  	bne	a5, a2, 0x171d0 <_ldtoa_r+0x9a>
   171de: 9b 17 07 01  	slliw	a5, a4, 16
   171e2: 9b d7 07 41  	sraiw	a5, a5, 16
   171e6: 22 67        	ld	a4, 8(sp)
   171e8: 9b d7 f7 01  	srliw	a5, a5, 31
   171ec: 1c c0        	sw	a5, 0(s0)
   171ee: 8d 47        	addi	a5, zero, 3
   171f0: 63 0c f7 7a  	beq	a4, a5, 0x179a8 <_ldtoa_r+0x872>
   171f4: d1 47        	addi	a5, zero, 20
   171f6: 3e ec        	sd	a5, 24(sp)
   171f8: a2 67        	ld	a5, 8(sp)
   171fa: 89 cf        	beqz	a5, 0x17214 <_ldtoa_r+0xde>
   171fc: 82 77        	ld	a5, 32(sp)
   171fe: fd 37        	addiw	a5, a5, -1
   17200: 93 06 a0 02  	addi	a3, zero, 42
   17204: 3e 87        	mv	a4, a5
   17206: 63 d4 f6 00  	bge	a3, a5, 0x1720e <_ldtoa_r+0xd8>
   1720a: 13 07 a0 02  	addi	a4, zero, 42
   1720e: 9b 07 07 00  	sext.w	a5, a4
   17212: 3e ec        	sd	a5, 24(sp)
   17214: 03 27 41 22  	lw	a4, 548(sp)
   17218: a1 67        	lui	a5, 8
   1721a: fd 17        	addi	a5, a5, -1
   1721c: 3a fc        	sd	a4, 56(sp)
   1721e: 22 77        	ld	a4, 40(sp)
   17220: 63 0d f7 3a  	beq	a4, a5, 0x175da <_ldtoa_r+0x4a4>
   17224: 93 07 00 09  	addi	a5, zero, 144
   17228: 23 22 f1 22  	sw	a5, 548(sp)
   1722c: 00 02        	addi	s0, sp, 256
   1722e: 90 11        	addi	a2, sp, 224
   17230: d7 77 85 c4  	<unknown>
   17234: 73 27 20 c2  	csrr	a4, vlenb
   17238: a9 46        	addi	a3, zero, 10
   1723a: 07 5c 06 02  	<unknown>
   1723e: 27 5c 04 02  	<unknown>
   17242: 33 0c e4 00  	add	s8, s0, a4
   17246: 33 8d f6 40  	sub	s10, a3, a5
   1724a: 63 8f f6 3e  	beq	a3, a5, 0x17648 <_ldtoa_r+0x512>
   1724e: 94 09        	addi	a3, sp, 208
   17250: 93 07 07 01  	addi	a5, a4, 16
   17254: b6 97        	add	a5, a5, a3
   17256: 57 70 8d 04  	<unknown>
   1725a: 07 dc 07 02  	<unknown>
   1725e: 27 5c 0c 02  	<unknown>
   17262: 83 56 21 11  	lhu	a3, 274(sp)
   17266: b6 e0        	sd	a3, 64(sp)
   17268: 93 d7 f6 00  	srli	a5, a3, 15
   1726c: e3 98 07 22  	bnez	a5, 0x17c9c <_ldtoa_r+0xb66>
   17270: b7 27 02 00  	lui	a5, 34
   17274: 93 8d 87 16  	addi	s11, a5, 360
   17278: 13 09 01 12  	addi	s2, sp, 288
   1727c: 33 86 ed 00  	add	a2, s11, a4
   17280: 4a 97        	add	a4, a4, s2
   17282: b2 f0        	sd	a2, 96(sp)
   17284: ba e8        	sd	a4, 80(sp)
   17286: 57 70 85 c4  	<unknown>
   1728a: 07 dc 0d 02  	<unknown>
   1728e: 27 5c 09 02  	<unknown>
   17292: 57 70 8d 04  	<unknown>
   17296: 07 5c 06 02  	<unknown>
   1729a: 27 5c 07 02  	<unknown>
   1729e: 63 9d 06 3c  	bnez	a3, 0x17678 <_ldtoa_r+0x542>
   172a2: 22 87        	mv	a4, s0
   172a4: 13 06 21 11  	addi	a2, sp, 274
   172a8: 83 56 07 00  	lhu	a3, 0(a4)
   172ac: 63 9a 06 58  	bnez	a3, 0x17840 <_ldtoa_r+0x70a>
   172b0: 09 07        	addi	a4, a4, 2
   172b2: e3 1b e6 fe  	bne	a2, a4, 0x172a8 <_ldtoa_r+0x172>
   172b6: d7 77 86 c4  	<unknown>
   172ba: 81 44        	mv	s1, zero
   172bc: 93 0b 01 1e  	addi	s7, sp, 480
   172c0: 13 0a 01 16  	addi	s4, sp, 352
   172c4: 93 09 01 22  	addi	s3, sp, 544
   172c8: be ec        	sd	a5, 88(sp)
   172ca: d2 85        	mv	a1, s4
   172cc: 4a 85        	mv	a0, s2
   172ce: ef e0 df d2  	jal	0x15ffa <emovi>
   172d2: 57 70 86 c4  	<unknown>
   172d6: 07 5c 0a 02  	<unknown>
   172da: 27 5c 09 02  	<unknown>
   172de: 66 67        	ld	a4, 88(sp)
   172e0: b1 47        	addi	a5, zero, 12
   172e2: b3 8d e7 40  	sub	s11, a5, a4
   172e6: 63 92 e7 68  	bne	a5, a4, 0x1796a <_ldtoa_r+0x834>
   172ea: d2 85        	mv	a1, s4
   172ec: 22 85        	mv	a0, s0
   172ee: 23 1c 01 12  	sh	zero, 312(sp)
   172f2: ef e0 9f d0  	jal	0x15ffa <emovi>
   172f6: 57 70 86 c4  	<unknown>
   172fa: 07 5c 0a 02  	<unknown>
   172fe: 27 5c 04 02  	<unknown>
   17302: 4e 86        	mv	a2, s3
   17304: a2 85        	mv	a1, s0
   17306: 4a 85        	mv	a0, s2
   17308: 23 1c 01 10  	sh	zero, 280(sp)
   1730c: ef f0 4f aa  	jal	0x165b0 <eiremain>
   17310: 83 56 c1 26  	lhu	a3, 620(sp)
   17314: 63 93 06 12  	bnez	a3, 0x1743a <_ldtoa_r+0x304>
   17318: 93 0a 01 14  	addi	s5, sp, 320
   1731c: 73 2a 20 c2  	csrr	s4, vlenb
   17320: 56 9a        	add	s4, s4, s5
   17322: 37 2b 02 00  	lui	s6, 34
   17326: 13 0d 21 10  	addi	s10, sp, 258
   1732a: 93 0c 21 14  	addi	s9, sp, 322
   1732e: b6 e8        	sd	a3, 80(sp)
   17330: 93 05 0b 18  	addi	a1, s6, 384
   17334: 22 85        	mv	a0, s0
   17336: ef e0 df d7  	jal	0x160b2 <ecmp>
   1733a: 63 05 05 66  	beqz	a0, 0x179a4 <_ldtoa_r+0x86e>
   1733e: 01 47        	mv	a4, zero
   17340: 34 0a        	addi	a3, sp, 280
   17342: 83 d7 06 00  	lhu	a5, 0(a3)
   17346: 13 d6 f7 00  	srli	a2, a5, 15
   1734a: 19 c2        	beqz	a2, 0x17350 <_ldtoa_r+0x21a>
   1734c: 13 67 17 00  	ori	a4, a4, 1
   17350: 9b 97 17 00  	slliw	a5, a5, 1
   17354: c2 17        	slli	a5, a5, 48
   17356: 13 76 27 00  	andi	a2, a4, 2
   1735a: c1 93        	srli	a5, a5, 48
   1735c: 19 c2        	beqz	a2, 0x17362 <_ldtoa_r+0x22c>
   1735e: 93 e7 17 00  	ori	a5, a5, 1
   17362: 1b 17 17 00  	slliw	a4, a4, 1
   17366: 23 90 f6 00  	sh	a5, 0(a3)
   1736a: 42 17        	slli	a4, a4, 48
   1736c: f9 16        	addi	a3, a3, -2
   1736e: 41 93        	srli	a4, a4, 48
   17370: e3 99 a6 fd  	bne	a3, s10, 0x17342 <_ldtoa_r+0x20c>
   17374: 57 70 86 c4  	<unknown>
   17378: 07 5c 04 02  	<unknown>
   1737c: 27 dc 0a 02  	<unknown>
   17380: 63 88 0d 00  	beqz	s11, 0x17390 <_ldtoa_r+0x25a>
   17384: 57 f0 8d 04  	<unknown>
   17388: 07 5c 0c 02  	<unknown>
   1738c: 27 5c 0a 02  	<unknown>
   17390: 23 1c 01 14  	sh	zero, 344(sp)
   17394: 01 47        	mv	a4, zero
   17396: b4 0a        	addi	a3, sp, 344
   17398: 83 d7 06 00  	lhu	a5, 0(a3)
   1739c: 13 d6 f7 00  	srli	a2, a5, 15
   173a0: 19 c2        	beqz	a2, 0x173a6 <_ldtoa_r+0x270>
   173a2: 13 67 17 00  	ori	a4, a4, 1
   173a6: 9b 97 17 00  	slliw	a5, a5, 1
   173aa: c2 17        	slli	a5, a5, 48
   173ac: 13 76 27 00  	andi	a2, a4, 2
   173b0: c1 93        	srli	a5, a5, 48
   173b2: 19 c2        	beqz	a2, 0x173b8 <_ldtoa_r+0x282>
   173b4: 93 e7 17 00  	ori	a5, a5, 1
   173b8: 1b 17 17 00  	slliw	a4, a4, 1
   173bc: 23 90 f6 00  	sh	a5, 0(a3)
   173c0: 42 17        	slli	a4, a4, 48
   173c2: f9 16        	addi	a3, a3, -2
   173c4: 41 93        	srli	a4, a4, 48
   173c6: e3 99 96 fd  	bne	a3, s9, 0x17398 <_ldtoa_r+0x262>
   173ca: 01 47        	mv	a4, zero
   173cc: b4 0a        	addi	a3, sp, 344
   173ce: 83 d7 06 00  	lhu	a5, 0(a3)
   173d2: 13 d6 f7 00  	srli	a2, a5, 15
   173d6: 19 c2        	beqz	a2, 0x173dc <_ldtoa_r+0x2a6>
   173d8: 13 67 17 00  	ori	a4, a4, 1
   173dc: 9b 97 17 00  	slliw	a5, a5, 1
   173e0: c2 17        	slli	a5, a5, 48
   173e2: 13 76 27 00  	andi	a2, a4, 2
   173e6: c1 93        	srli	a5, a5, 48
   173e8: 19 c2        	beqz	a2, 0x173ee <_ldtoa_r+0x2b8>
   173ea: 93 e7 17 00  	ori	a5, a5, 1
   173ee: 1b 17 17 00  	slliw	a4, a4, 1
   173f2: 23 90 f6 00  	sh	a5, 0(a3)
   173f6: 42 17        	slli	a4, a4, 48
   173f8: f9 16        	addi	a3, a3, -2
   173fa: 41 93        	srli	a4, a4, 48
   173fc: e3 99 96 fd  	bne	a3, s9, 0x173ce <_ldtoa_r+0x298>
   17400: 01 46        	mv	a2, zero
   17402: 34 0a        	addi	a3, sp, 280
   17404: b8 0a        	addi	a4, sp, 344
   17406: 83 d5 06 00  	lhu	a1, 0(a3)
   1740a: 83 57 07 00  	lhu	a5, 0(a4)
   1740e: f9 16        	addi	a3, a3, -2
   17410: 79 17        	addi	a4, a4, -2
   17412: ae 97        	add	a5, a5, a1
   17414: b2 97        	add	a5, a5, a2
   17416: 13 d6 07 01  	srli	a2, a5, 16
   1741a: 23 91 f6 00  	sh	a5, 2(a3)
   1741e: 05 8a        	andi	a2, a2, 1
   17420: e3 13 97 ff  	bne	a4, s9, 0x17406 <_ldtoa_r+0x2d0>
   17424: 4e 86        	mv	a2, s3
   17426: a2 85        	mv	a1, s0
   17428: 4a 85        	mv	a0, s2
   1742a: ef f0 6f 98  	jal	0x165b0 <eiremain>
   1742e: 83 57 c1 26  	lhu	a5, 620(sp)
   17432: fd 34        	addiw	s1, s1, -1
   17434: e3 8e 07 ee  	beqz	a5, 0x17330 <_ldtoa_r+0x1fa>
   17438: be 86        	mv	a3, a5
   1743a: 06 67        	ld	a4, 64(sp)
   1743c: 93 07 00 02  	addi	a5, zero, 32
   17440: 19 c3        	beqz	a4, 0x17446 <_ldtoa_r+0x310>
   17442: 93 07 d0 02  	addi	a5, zero, 45
   17446: 22 67        	ld	a4, 8(sp)
   17448: 23 00 f1 1e  	sb	a5, 480(sp)
   1744c: 0d 46        	addi	a2, zero, 3
   1744e: e2 67        	ld	a5, 24(sp)
   17450: 63 13 c7 00  	bne	a4, a2, 0x17456 <_ldtoa_r+0x320>
   17454: a5 9f        	addw	a5, a5, s1
   17456: 93 05 a0 02  	addi	a1, zero, 42
   1745a: 1b 86 07 00  	sext.w	a2, a5
   1745e: 63 d4 f5 00  	bge	a1, a5, 0x17466 <_ldtoa_r+0x330>
   17462: 13 06 a0 02  	addi	a2, zero, 42
   17466: a9 45        	addi	a1, zero, 10
   17468: 63 84 b6 56  	beq	a3, a1, 0x179d0 <_ldtoa_r+0x89a>
   1746c: 9b 86 06 03  	addiw	a3, a3, 48
   17470: a3 00 d1 1e  	sb	a3, 481(sp)
   17474: 93 06 e0 02  	addi	a3, zero, 46
   17478: 23 01 d1 1e  	sb	a3, 482(sp)
   1747c: e3 cd 07 1e  	bltz	a5, 0x17e76 <_ldtoa_r+0xd40>
   17480: 13 0a 31 1e  	addi	s4, sp, 483
   17484: 93 0a 01 14  	addi	s5, sp, 320
   17488: f3 27 20 c2  	csrr	a5, vlenb
   1748c: 01 4d        	mv	s10, zero
   1748e: d6 97        	add	a5, a5, s5
   17490: a6 e0        	sd	s1, 64(sp)
   17492: de e8        	sd	s7, 80(sp)
   17494: d2 84        	mv	s1, s4
   17496: ee 8b        	mv	s7, s11
   17498: 6a 8a        	mv	s4, s10
   1749a: e2 8d        	mv	s11, s8
   1749c: 93 0c 21 10  	addi	s9, sp, 258
   174a0: 4e 8c        	mv	s8, s3
   174a2: 13 0b 21 14  	addi	s6, sp, 322
   174a6: 3e 8d        	mv	s10, a5
   174a8: b2 89        	mv	s3, a2
   174aa: 01 46        	mv	a2, zero
   174ac: 2c 0a        	addi	a1, sp, 280
   174ae: 83 d7 05 00  	lhu	a5, 0(a1)
   174b2: 13 d5 f7 00  	srli	a0, a5, 15
   174b6: 19 c1        	beqz	a0, 0x174bc <_ldtoa_r+0x386>
   174b8: 13 66 16 00  	ori	a2, a2, 1
   174bc: 9b 97 17 00  	slliw	a5, a5, 1
   174c0: c2 17        	slli	a5, a5, 48
   174c2: 13 75 26 00  	andi	a0, a2, 2
   174c6: c1 93        	srli	a5, a5, 48
   174c8: 19 c1        	beqz	a0, 0x174ce <_ldtoa_r+0x398>
   174ca: 93 e7 17 00  	ori	a5, a5, 1
   174ce: 1b 16 16 00  	slliw	a2, a2, 1
   174d2: 23 90 f5 00  	sh	a5, 0(a1)
   174d6: 42 16        	slli	a2, a2, 48
   174d8: f9 15        	addi	a1, a1, -2
   174da: 41 92        	srli	a2, a2, 48
   174dc: e3 99 95 fd  	bne	a1, s9, 0x174ae <_ldtoa_r+0x378>
   174e0: 57 70 86 c4  	<unknown>
   174e4: 07 5c 04 02  	<unknown>
   174e8: 27 dc 0a 02  	<unknown>
   174ec: 63 88 0b 00  	beqz	s7, 0x174fc <_ldtoa_r+0x3c6>
   174f0: 57 f0 8b 04  	<unknown>
   174f4: 07 dc 0d 02  	<unknown>
   174f8: 27 5c 0d 02  	<unknown>
   174fc: 23 1c 01 14  	sh	zero, 344(sp)
   17500: 01 46        	mv	a2, zero
   17502: ac 0a        	addi	a1, sp, 344
   17504: 83 d7 05 00  	lhu	a5, 0(a1)
   17508: 13 d5 f7 00  	srli	a0, a5, 15
   1750c: 19 c1        	beqz	a0, 0x17512 <_ldtoa_r+0x3dc>
   1750e: 13 66 16 00  	ori	a2, a2, 1
   17512: 9b 97 17 00  	slliw	a5, a5, 1
   17516: c2 17        	slli	a5, a5, 48
   17518: 13 75 26 00  	andi	a0, a2, 2
   1751c: c1 93        	srli	a5, a5, 48
   1751e: 19 c1        	beqz	a0, 0x17524 <_ldtoa_r+0x3ee>
   17520: 93 e7 17 00  	ori	a5, a5, 1
   17524: 1b 16 16 00  	slliw	a2, a2, 1
   17528: 23 90 f5 00  	sh	a5, 0(a1)
   1752c: 42 16        	slli	a2, a2, 48
   1752e: f9 15        	addi	a1, a1, -2
   17530: 41 92        	srli	a2, a2, 48
   17532: e3 99 65 fd  	bne	a1, s6, 0x17504 <_ldtoa_r+0x3ce>
   17536: 01 46        	mv	a2, zero
   17538: ac 0a        	addi	a1, sp, 344
   1753a: 83 d7 05 00  	lhu	a5, 0(a1)
   1753e: 13 d5 f7 00  	srli	a0, a5, 15
   17542: 19 c1        	beqz	a0, 0x17548 <_ldtoa_r+0x412>
   17544: 13 66 16 00  	ori	a2, a2, 1
   17548: 9b 97 17 00  	slliw	a5, a5, 1
   1754c: c2 17        	slli	a5, a5, 48
   1754e: 13 75 26 00  	andi	a0, a2, 2
   17552: c1 93        	srli	a5, a5, 48
   17554: 19 c1        	beqz	a0, 0x1755a <_ldtoa_r+0x424>
   17556: 93 e7 17 00  	ori	a5, a5, 1
   1755a: 1b 16 16 00  	slliw	a2, a2, 1
   1755e: 23 90 f5 00  	sh	a5, 0(a1)
   17562: 42 16        	slli	a2, a2, 48
   17564: f9 15        	addi	a1, a1, -2
   17566: 41 92        	srli	a2, a2, 48
   17568: e3 99 65 fd  	bne	a1, s6, 0x1753a <_ldtoa_r+0x404>
   1756c: 01 45        	mv	a0, zero
   1756e: 2c 0a        	addi	a1, sp, 280
   17570: b0 0a        	addi	a2, sp, 344
   17572: 03 d8 05 00  	lhu	a6, 0(a1)
   17576: 83 57 06 00  	lhu	a5, 0(a2)
   1757a: f9 15        	addi	a1, a1, -2
   1757c: 79 16        	addi	a2, a2, -2
   1757e: c2 97        	add	a5, a5, a6
   17580: aa 97        	add	a5, a5, a0
   17582: 13 d5 07 01  	srli	a0, a5, 16
   17586: 23 91 f5 00  	sh	a5, 2(a1)
   1758a: 05 89        	andi	a0, a0, 1
   1758c: e3 13 66 ff  	bne	a2, s6, 0x17572 <_ldtoa_r+0x43c>
   17590: 62 86        	mv	a2, s8
   17592: a2 85        	mv	a1, s0
   17594: 4a 85        	mv	a0, s2
   17596: ef f0 af 81  	jal	0x165b0 <eiremain>
   1759a: 83 55 c1 26  	lhu	a1, 620(sp)
   1759e: 93 87 14 00  	addi	a5, s1, 1
   175a2: 05 2a        	addiw	s4, s4, 1
   175a4: 1b 86 05 03  	addiw	a2, a1, 48
   175a8: a3 8f c7 fe  	sb	a2, -1(a5)
   175ac: 63 cf 49 1d  	blt	s3, s4, 0x1778a <_ldtoa_r+0x654>
   175b0: be 84        	mv	s1, a5
   175b2: e5 bd        	j	0x174aa <_ldtoa_r+0x374>
   175b4: 22 67        	ld	a4, 8(sp)
   175b6: 23 20 04 00  	sw	zero, 0(s0)
   175ba: 8d 47        	addi	a5, zero, 3
   175bc: e3 1c f7 c2  	bne	a4, a5, 0x171f4 <_ldtoa_r+0xbe>
   175c0: 82 76        	ld	a3, 32(sp)
   175c2: 13 07 a0 02  	addi	a4, zero, 42
   175c6: 9b 87 06 00  	sext.w	a5, a3
   175ca: 63 54 d7 00  	bge	a4, a3, 0x175d2 <_ldtoa_r+0x49c>
   175ce: 93 07 a0 02  	addi	a5, zero, 42
   175d2: 03 27 41 22  	lw	a4, 548(sp)
   175d6: 3e ec        	sd	a5, 24(sp)
   175d8: 3a fc        	sd	a4, 56(sp)
   175da: 9c 11        	addi	a5, sp, 224
   175dc: 93 06 21 0f  	addi	a3, sp, 242
   175e0: 19 a0        	j	0x175e6 <_ldtoa_r+0x4b0>
   175e2: e3 81 d7 c4  	beq	a5, a3, 0x17224 <_ldtoa_r+0xee>
   175e6: 03 d7 07 00  	lhu	a4, 0(a5)
   175ea: 89 07        	addi	a5, a5, 2
   175ec: 7d db        	beqz	a4, 0x175e2 <_ldtoa_r+0x4ac>
   175ee: 93 0b 01 1e  	addi	s7, sp, 480
   175f2: b7 25 02 00  	lui	a1, 34
   175f6: 93 85 85 f2  	addi	a1, a1, -216
   175fa: 5e 85        	mv	a0, s7
   175fc: ef 20 80 40  	jal	0x19a04 <sprintf>
   17600: e2 77        	ld	a5, 56(sp)
   17602: 89 64        	lui	s1, 2
   17604: 93 84 f4 70  	addi	s1, s1, 1807
   17608: 23 22 f1 22  	sw	a5, 548(sp)
   1760c: 23 28 91 22  	sw	s1, 560(sp)
   17610: 9c 11        	addi	a5, sp, 224
   17612: 13 06 21 0f  	addi	a2, sp, 242
   17616: 03 d7 07 00  	lhu	a4, 0(a5)
   1761a: 89 07        	addi	a5, a5, 2
   1761c: e3 16 07 00  	bnez	a4, 0x17e28 <_ldtoa_r+0xcf2>
   17620: e3 9b c7 fe  	bne	a5, a2, 0x17616 <_ldtoa_r+0x4e0>
   17624: 42 77        	ld	a4, 48(sp)
   17626: 89 67        	lui	a5, 2
   17628: 93 87 f7 70  	addi	a5, a5, 1807
   1762c: 1c c3        	sw	a5, 0(a4)
   1762e: de 87        	mv	a5, s7
   17630: 13 06 00 02  	addi	a2, zero, 32
   17634: 93 05 d0 02  	addi	a1, zero, 45
   17638: 03 c7 07 00  	lbu	a4, 0(a5)
   1763c: 63 04 c7 00  	beq	a4, a2, 0x17644 <_ldtoa_r+0x50e>
   17640: 63 12 b7 06  	bne	a4, a1, 0x176a4 <_ldtoa_r+0x56e>
   17644: 85 07        	addi	a5, a5, 1
   17646: cd bf        	j	0x17638 <_ldtoa_r+0x502>
   17648: 83 56 21 11  	lhu	a3, 274(sp)
   1764c: b6 e0        	sd	a3, 64(sp)
   1764e: 93 d7 f6 00  	srli	a5, a3, 15
   17652: 63 98 07 60  	bnez	a5, 0x17c62 <_ldtoa_r+0xb2c>
   17656: b7 27 02 00  	lui	a5, 34
   1765a: 93 8d 87 16  	addi	s11, a5, 360
   1765e: 13 09 01 12  	addi	s2, sp, 288
   17662: 07 dc 0d 02  	<unknown>
   17666: 33 86 ed 00  	add	a2, s11, a4
   1766a: 27 5c 09 02  	<unknown>
   1766e: 4a 97        	add	a4, a4, s2
   17670: b2 f0        	sd	a2, 96(sp)
   17672: ba e8        	sd	a4, 80(sp)
   17674: e3 87 06 c2  	beqz	a3, 0x172a2 <_ldtoa_r+0x16c>
   17678: 21 67        	lui	a4, 8
   1767a: 7d 17        	addi	a4, a4, -1
   1767c: 63 81 e6 58  	beq	a3, a4, 0x17bfe <_ldtoa_r+0xac8>
   17680: 82 e0        	sd	zero, 64(sp)
   17682: 03 17 01 11  	lh	a4, 272(sp)
   17686: 63 4d 07 1a  	bltz	a4, 0x17840 <_ldtoa_r+0x70a>
   1768a: 93 0b 01 1e  	addi	s7, sp, 480
   1768e: b7 25 02 00  	lui	a1, 34
   17692: 93 85 05 ea  	addi	a1, a1, -352
   17696: 5e 85        	mv	a0, s7
   17698: ef 20 c0 36  	jal	0x19a04 <sprintf>
   1769c: 89 64        	lui	s1, 2
   1769e: 93 84 f4 70  	addi	s1, s1, 1807
   176a2: a9 a2        	j	0x177ec <_ldtoa_r+0x6b6>
   176a4: 5e 86        	mv	a2, s7
   176a6: 03 c7 07 00  	lbu	a4, 0(a5)
   176aa: 32 84        	mv	s0, a2
   176ac: 85 07        	addi	a5, a5, 1
   176ae: 23 00 e6 00  	sb	a4, 0(a2)
   176b2: 05 06        	addi	a2, a2, 1
   176b4: 6d fb        	bnez	a4, 0x176a6 <_ldtoa_r+0x570>
   176b6: 22 67        	ld	a4, 8(sp)
   176b8: 89 47        	addi	a5, zero, 2
   176ba: 03 46 f4 ff  	lbu	a2, -1(s0)
   176be: 63 0c f7 28  	beq	a4, a5, 0x17956 <_ldtoa_r+0x820>
   176c2: 62 67        	ld	a4, 24(sp)
   176c4: a6 87        	mv	a5, s1
   176c6: 63 d3 e4 00  	bge	s1, a4, 0x176cc <_ldtoa_r+0x596>
   176ca: ba 87        	mv	a5, a4
   176cc: 93 05 00 03  	addi	a1, zero, 48
   176d0: 9b 86 07 00  	sext.w	a3, a5
   176d4: 63 14 b6 02  	bne	a2, a1, 0x176fc <_ldtoa_r+0x5c6>
   176d8: b3 07 74 41  	sub	a5, s0, s7
   176dc: 63 d0 f6 02  	bge	a3, a5, 0x176fc <_ldtoa_r+0x5c6>
   176e0: 13 06 00 03  	addi	a2, zero, 48
   176e4: 29 a0        	j	0x176ee <_ldtoa_r+0x5b8>
   176e6: b3 07 74 41  	sub	a5, s0, s7
   176ea: 63 d9 f6 00  	bge	a3, a5, 0x176fc <_ldtoa_r+0x5c6>
   176ee: 83 47 e4 ff  	lbu	a5, -2(s0)
   176f2: a3 0f 04 fe  	sb	zero, -1(s0)
   176f6: 7d 14        	addi	s0, s0, -1
   176f8: e3 87 c7 fe  	beq	a5, a2, 0x176e6 <_ldtoa_r+0x5b0>
   176fc: 22 67        	ld	a4, 8(sp)
   176fe: 8d 47        	addi	a5, zero, 3
   17700: 63 05 f7 22  	beq	a4, a5, 0x1792a <_ldtoa_r+0x7f4>
   17704: 82 77        	ld	a5, 32(sp)
   17706: 7d 46        	addi	a2, zero, 31
   17708: 85 46        	addi	a3, zero, 1
   1770a: 1b 85 97 00  	addiw	a0, a5, 9
   1770e: 91 47        	addi	a5, zero, 4
   17710: 63 7e a6 22  	bgeu	a2, a0, 0x1794c <_ldtoa_r+0x816>
   17714: 9b 97 17 00  	slliw	a5, a5, 1
   17718: 13 87 c7 01  	addi	a4, a5, 28
   1771c: 9b 85 06 00  	sext.w	a1, a3
   17720: 85 26        	addiw	a3, a3, 1
   17722: e3 79 e5 fe  	bgeu	a0, a4, 0x17714 <_ldtoa_r+0x5de>
   17726: c2 67        	ld	a5, 16(sp)
   17728: 81 25        	sext.w	a1, a1
   1772a: ac d7        	sw	a1, 104(a5)
   1772c: 42 65        	ld	a0, 16(sp)
   1772e: ef 10 80 4d  	jal	0x18c06 <_Balloc>
   17732: aa 84        	mv	s1, a0
   17734: e3 00 05 20  	beqz	a0, 0x18134 <_ldtoa_r+0xffe>
   17738: c2 67        	ld	a5, 16(sp)
   1773a: de 85        	mv	a1, s7
   1773c: a8 f3        	sd	a0, 96(a5)
   1773e: ef 20 40 4e  	jal	0x19c22 <strcpy>
   17742: 26 67        	ld	a4, 72(sp)
   17744: 09 c7        	beqz	a4, 0x1774e <_ldtoa_r+0x618>
   17746: b3 07 74 41  	sub	a5, s0, s7
   1774a: a6 97        	add	a5, a5, s1
   1774c: 1c e3        	sd	a5, 0(a4)
   1774e: 83 30 81 2d  	ld	ra, 728(sp)
   17752: 03 34 01 2d  	ld	s0, 720(sp)
   17756: 03 39 01 2c  	ld	s2, 704(sp)
   1775a: 83 39 81 2b  	ld	s3, 696(sp)
   1775e: 03 3a 01 2b  	ld	s4, 688(sp)
   17762: 83 3a 81 2a  	ld	s5, 680(sp)
   17766: 03 3b 01 2a  	ld	s6, 672(sp)
   1776a: 83 3b 81 29  	ld	s7, 664(sp)
   1776e: 03 3c 01 29  	ld	s8, 656(sp)
   17772: 83 3c 81 28  	ld	s9, 648(sp)
   17776: 03 3d 01 28  	ld	s10, 640(sp)
   1777a: 83 3d 81 27  	ld	s11, 632(sp)
   1777e: 26 85        	mv	a0, s1
   17780: 83 34 81 2c  	ld	s1, 712(sp)
   17784: 13 01 01 2e  	addi	sp, sp, 736
   17788: 82 80        	ret
   1778a: 26 8a        	mv	s4, s1
   1778c: c6 6b        	ld	s7, 80(sp)
   1778e: 86 64        	ld	s1, 64(sp)
   17790: 4e 86        	mv	a2, s3
   17792: 11 45        	addi	a0, zero, 4
   17794: 63 54 b5 04  	bge	a0, a1, 0x177dc <_ldtoa_r+0x6a6>
   17798: 15 45        	addi	a0, zero, 5
   1779a: 63 87 a5 72  	beq	a1, a0, 0x17ec8 <_ldtoa_r+0xd92>
   1779e: 83 c5 e7 ff  	lbu	a1, -2(a5)
   177a2: 13 85 e7 ff  	addi	a0, a5, -2
   177a6: 93 f5 f5 07  	andi	a1, a1, 127
   177aa: 63 44 06 02  	bltz	a2, 0x177d2 <_ldtoa_r+0x69c>
   177ae: 93 07 e0 02  	addi	a5, zero, 46
   177b2: 13 06 80 03  	addi	a2, zero, 56
   177b6: 13 08 00 03  	addi	a6, zero, 48
   177ba: 63 8b f5 64  	beq	a1, a5, 0x17e10 <_ldtoa_r+0xcda>
   177be: 63 75 b6 64  	bgeu	a2, a1, 0x17e08 <_ldtoa_r+0xcd2>
   177c2: 83 45 f5 ff  	lbu	a1, -1(a0)
   177c6: 23 00 05 01  	sb	a6, 0(a0)
   177ca: 7d 15        	addi	a0, a0, -1
   177cc: 93 f5 f5 07  	andi	a1, a1, 127
   177d0: ed b7        	j	0x177ba <_ldtoa_r+0x684>
   177d2: 13 06 10 03  	addi	a2, zero, 49
   177d6: 23 8f c7 fe  	sb	a2, -2(a5)
   177da: 85 24        	addiw	s1, s1, 1
   177dc: b7 25 02 00  	lui	a1, 34
   177e0: 26 86        	mv	a2, s1
   177e2: 93 85 85 ea  	addi	a1, a1, -344
   177e6: 52 85        	mv	a0, s4
   177e8: ef 20 c0 21  	jal	0x19a04 <sprintf>
   177ec: e2 77        	ld	a5, 56(sp)
   177ee: 22 77        	ld	a4, 40(sp)
   177f0: 23 28 91 22  	sw	s1, 560(sp)
   177f4: 23 22 f1 22  	sw	a5, 548(sp)
   177f8: a1 67        	lui	a5, 8
   177fa: fd 17        	addi	a5, a5, -1
   177fc: e3 0a f7 e0  	beq	a4, a5, 0x17610 <_ldtoa_r+0x4da>
   17800: 42 77        	ld	a4, 48(sp)
   17802: 83 46 01 1e  	lbu	a3, 480(sp)
   17806: 9b 87 14 00  	addiw	a5, s1, 1
   1780a: 1c c3        	sw	a5, 0(a4)
   1780c: 63 87 06 44  	beqz	a3, 0x17c5a <_ldtoa_r+0xb24>
   17810: de 87        	mv	a5, s7
   17812: 13 06 e0 02  	addi	a2, zero, 46
   17816: 63 8b c6 18  	beq	a3, a2, 0x179ac <_ldtoa_r+0x876>
   1781a: 83 c6 17 00  	lbu	a3, 1(a5)
   1781e: 85 07        	addi	a5, a5, 1
   17820: fd fa        	bnez	a3, 0x17816 <_ldtoa_r+0x6e0>
   17822: 13 06 50 04  	addi	a2, zero, 69
   17826: 63 e5 fb 00  	bltu	s7, a5, 0x17830 <_ldtoa_r+0x6fa>
   1782a: 01 a8        	j	0x1783a <_ldtoa_r+0x704>
   1782c: 63 87 77 43  	beq	a5, s7, 0x17c5a <_ldtoa_r+0xb24>
   17830: 03 c7 f7 ff  	lbu	a4, -1(a5)
   17834: fd 17        	addi	a5, a5, -1
   17836: e3 1b c7 fe  	bne	a4, a2, 0x1782c <_ldtoa_r+0x6f6>
   1783a: 23 80 07 00  	sb	zero, 0(a5)
   1783e: c5 bb        	j	0x1762e <_ldtoa_r+0x4f8>
   17840: a2 85        	mv	a1, s0
   17842: 13 85 87 16  	addi	a0, a5, 360
   17846: be ec        	sd	a5, 88(sp)
   17848: ef e0 bf 86  	jal	0x160b2 <ecmp>
   1784c: e3 05 05 a6  	beqz	a0, 0x172b6 <_ldtoa_r+0x180>
   17850: e6 67        	ld	a5, 88(sp)
   17852: 63 42 05 1a  	bltz	a0, 0x179f6 <_ldtoa_r+0x8c0>
   17856: 03 57 21 11  	lhu	a4, 274(sp)
   1785a: 63 1d 07 6e  	bnez	a4, 0x17f54 <_ldtoa_r+0xe1e>
   1785e: 03 17 01 11  	lh	a4, 272(sp)
   17862: 81 44        	mv	s1, zero
   17864: e3 41 07 02  	bltz	a4, 0x18086 <_ldtoa_r+0xf50>
   17868: 93 09 01 22  	addi	s3, sp, 544
   1786c: b7 2c 02 00  	lui	s9, 34
   17870: 3e 8a        	mv	s4, a5
   17872: ce 86        	mv	a3, s3
   17874: 22 86        	mv	a2, s0
   17876: a2 85        	mv	a1, s0
   17878: 13 85 0c 15  	addi	a0, s9, 336
   1787c: ef f0 cf c9  	jal	0x16d18 <emul>
   17880: 83 17 01 11  	lh	a5, 272(sp)
   17884: fd 34        	addiw	s1, s1, -1
   17886: e3 d6 07 fe  	bgez	a5, 0x17872 <_ldtoa_r+0x73c>
   1788a: d2 87        	mv	a5, s4
   1788c: 57 77 86 c4  	<unknown>
   17890: 13 0a 01 16  	addi	s4, sp, 352
   17894: ba ec        	sd	a4, 88(sp)
   17896: 57 70 85 c4  	<unknown>
   1789a: 07 5c 04 02  	<unknown>
   1789e: 27 5c 0a 02  	<unknown>
   178a2: 63 1f 0d 58  	bnez	s10, 0x17e40 <_ldtoa_r+0xd0a>
   178a6: 07 dc 0d 02  	<unknown>
   178aa: 27 5c 09 02  	<unknown>
   178ae: 37 2d 02 00  	lui	s10, 34
   178b2: b7 2c 02 00  	lui	s9, 34
   178b6: b7 2d 02 00  	lui	s11, 34
   178ba: 7d 7b        	lui	s6, 1048575
   178bc: 13 0d 0d 06  	addi	s10, s10, 96
   178c0: 93 8c 8c f5  	addi	s9, s9, -168
   178c4: 93 8d 8d 04  	addi	s11, s11, 72
   178c8: be 8a        	mv	s5, a5
   178ca: 19 a0        	j	0x178d0 <_ldtoa_r+0x79a>
   178cc: d1 0c        	addi	s9, s9, 20
   178ce: 51 0d        	addi	s10, s10, 20
   178d0: d2 85        	mv	a1, s4
   178d2: 13 85 8a 16  	addi	a0, s5, 360
   178d6: ef e0 cf fd  	jal	0x160b2 <ecmp>
   178da: 63 5e a0 02  	blez	a0, 0x17916 <_ldtoa_r+0x7e0>
   178de: d2 85        	mv	a1, s4
   178e0: 66 85        	mv	a0, s9
   178e2: ef e0 0f fd  	jal	0x160b2 <ecmp>
   178e6: 63 40 05 02  	bltz	a0, 0x17906 <_ldtoa_r+0x7d0>
   178ea: ce 86        	mv	a3, s3
   178ec: 52 86        	mv	a2, s4
   178ee: d2 85        	mv	a1, s4
   178f0: 6a 85        	mv	a0, s10
   178f2: ef f0 6f c2  	jal	0x16d18 <emul>
   178f6: ce 86        	mv	a3, s3
   178f8: 4a 86        	mv	a2, s2
   178fa: ca 85        	mv	a1, s2
   178fc: 6a 85        	mv	a0, s10
   178fe: ef f0 af c1  	jal	0x16d18 <emul>
   17902: bb 04 9b 00  	addw	s1, s6, s1
   17906: 9b 57 fb 01  	srliw	a5, s6, 31
   1790a: bb 87 67 01  	addw	a5, a5, s6
   1790e: 1b db 17 40  	sraiw	s6, a5, 1
   17912: e3 9d bc fb  	bne	s9, s11, 0x178cc <_ldtoa_r+0x796>
   17916: ce 86        	mv	a3, s3
   17918: 4a 86        	mv	a2, s2
   1791a: 93 85 8a 16  	addi	a1, s5, 360
   1791e: 4a 85        	mv	a0, s2
   17920: ef e0 9f ef  	jal	0x16818 <ediv>
   17924: 93 0b 01 1e  	addi	s7, sp, 480
   17928: 4d b2        	j	0x172ca <_ldtoa_r+0x194>
   1792a: e2 67        	ld	a5, 24(sp)
   1792c: bd 9c        	addw	s1, s1, a5
   1792e: 63 c6 04 4c  	bltz	s1, 0x17dfa <_ldtoa_r+0xcc4>
   17932: c2 77        	ld	a5, 48(sp)
   17934: 02 77        	ld	a4, 32(sp)
   17936: 9c 43        	lw	a5, 0(a5)
   17938: b9 9f        	addw	a5, a5, a4
   1793a: 3e f0        	sd	a5, 32(sp)
   1793c: 82 77        	ld	a5, 32(sp)
   1793e: 7d 46        	addi	a2, zero, 31
   17940: 85 46        	addi	a3, zero, 1
   17942: 1b 85 37 00  	addiw	a0, a5, 3
   17946: 91 47        	addi	a5, zero, 4
   17948: e3 66 a6 dc  	bltu	a2, a0, 0x17714 <_ldtoa_r+0x5de>
   1794c: c2 67        	ld	a5, 16(sp)
   1794e: 81 45        	mv	a1, zero
   17950: 23 a4 07 06  	sw	zero, 104(a5)
   17954: e1 bb        	j	0x1772c <_ldtoa_r+0x5f6>
   17956: 93 07 00 03  	addi	a5, zero, 48
   1795a: e3 15 f6 da  	bne	a2, a5, 0x17704 <_ldtoa_r+0x5ce>
   1795e: b3 07 74 41  	sub	a5, s0, s7
   17962: 85 46        	addi	a3, zero, 1
   17964: e3 ce f6 d6  	blt	a3, a5, 0x176e0 <_ldtoa_r+0x5aa>
   17968: 71 bb        	j	0x17704 <_ldtoa_r+0x5ce>
   1796a: c6 67        	ld	a5, 80(sp)
   1796c: 73 2b 20 c2  	csrr	s6, vlenb
   17970: 52 9b        	add	s6, s6, s4
   17972: 57 f0 8d 04  	<unknown>
   17976: d2 85        	mv	a1, s4
   17978: 07 5c 0b 02  	<unknown>
   1797c: 22 85        	mv	a0, s0
   1797e: 27 dc 07 02  	<unknown>
   17982: 23 1c 01 12  	sh	zero, 312(sp)
   17986: ef e0 4f e7  	jal	0x15ffa <emovi>
   1798a: 57 70 86 c4  	<unknown>
   1798e: 07 5c 0a 02  	<unknown>
   17992: 27 5c 04 02  	<unknown>
   17996: 57 f0 8d 04  	<unknown>
   1799a: 07 5c 0b 02  	<unknown>
   1799e: 27 5c 0c 02  	<unknown>
   179a2: 85 b2        	j	0x17302 <_ldtoa_r+0x1cc>
   179a4: c6 66        	ld	a3, 80(sp)
   179a6: 51 bc        	j	0x1743a <_ldtoa_r+0x304>
   179a8: 82 77        	ld	a5, 32(sp)
   179aa: 99 b8        	j	0x17200 <_ldtoa_r+0xca>
   179ac: 83 c6 07 00  	lbu	a3, 0(a5)
   179b0: e3 89 06 e6  	beqz	a3, 0x17822 <_ldtoa_r+0x6ec>
   179b4: 03 c7 17 00  	lbu	a4, 1(a5)
   179b8: 85 07        	addi	a5, a5, 1
   179ba: a3 8f e7 fe  	sb	a4, -1(a5)
   179be: e3 02 07 e6  	beqz	a4, 0x17822 <_ldtoa_r+0x6ec>
   179c2: 03 c7 17 00  	lbu	a4, 1(a5)
   179c6: 85 07        	addi	a5, a5, 1
   179c8: a3 8f e7 fe  	sb	a4, -1(a5)
   179cc: 65 f7        	bnez	a4, 0x179b4 <_ldtoa_r+0x87e>
   179ce: 91 bd        	j	0x17822 <_ldtoa_r+0x6ec>
   179d0: 93 06 10 03  	addi	a3, zero, 49
   179d4: a3 00 d1 1e  	sb	a3, 481(sp)
   179d8: 93 06 e0 02  	addi	a3, zero, 46
   179dc: 23 01 d1 1e  	sb	a3, 482(sp)
   179e0: 85 24        	addiw	s1, s1, 1
   179e2: 63 58 f0 20  	blez	a5, 0x17bf2 <_ldtoa_r+0xabc>
   179e6: 93 07 00 03  	addi	a5, zero, 48
   179ea: a3 01 f1 1e  	sb	a5, 483(sp)
   179ee: 7d 36        	addiw	a2, a2, -1
   179f0: 13 0a 41 1e  	addi	s4, sp, 484
   179f4: 41 bc        	j	0x17484 <_ldtoa_r+0x34e>
   179f6: 93 0a 01 14  	addi	s5, sp, 320
   179fa: 57 70 85 c4  	<unknown>
   179fe: f3 2e 20 c2  	csrr	t4, vlenb
   17a02: 07 5c 04 02  	<unknown>
   17a06: 27 dc 0a 02  	<unknown>
   17a0a: d6 9e        	add	t4, t4, s5
   17a0c: 63 19 0d 52  	bnez	s10, 0x17f3e <_ldtoa_r+0xe08>
   17a10: f3 26 20 c2  	csrr	a3, vlenb
   17a14: 13 0a 01 16  	addi	s4, sp, 352
   17a18: 33 07 da 00  	add	a4, s4, a3
   17a1c: 93 0b 01 1e  	addi	s7, sp, 480
   17a20: 3a e5        	sd	a4, 136(sp)
   17a22: 33 87 db 00  	add	a4, s7, a3
   17a26: 3a e1        	sd	a4, 128(sp)
   17a28: 98 03        	addi	a4, sp, 448
   17a2a: 36 97        	add	a4, a4, a3
   17a2c: 3a f5        	sd	a4, 168(sp)
   17a2e: 18 03        	addi	a4, sp, 384
   17a30: 36 97        	add	a4, a4, a3
   17a32: 3a f1        	sd	a4, 160(sp)
   17a34: 18 13        	addi	a4, sp, 416
   17a36: 36 97        	add	a4, a4, a3
   17a38: b7 26 02 00  	lui	a3, 34
   17a3c: 93 86 46 16  	addi	a3, a3, 356
   17a40: b6 f4        	sd	a3, 104(sp)
   17a42: b7 26 02 00  	lui	a3, 34
   17a46: 93 86 06 f3  	addi	a3, a3, -208
   17a4a: 36 e9        	sd	a3, 144(sp)
   17a4c: a1 66        	lui	a3, 8
   17a4e: fd 16        	addi	a3, a3, -1
   17a50: 11 66        	lui	a2, 4
   17a52: 37 2e 02 00  	lui	t3, 34
   17a56: b6 f8        	sd	a3, 112(sp)
   17a58: f1 76        	lui	a3, 1048572
   17a5a: 81 44        	mv	s1, zero
   17a5c: 13 0b 0e 10  	addi	s6, t3, 256
   17a60: 93 09 01 22  	addi	s3, sp, 544
   17a64: 89 26        	addiw	a3, a3, 2
   17a66: 13 06 e6 08  	addi	a2, a2, 142
   17a6a: 3a ed        	sd	a4, 152(sp)
   17a6c: 41 47        	addi	a4, zero, 16
   17a6e: b6 de        	sw	a3, 124(sp)
   17a70: 62 f9        	sd	s8, 176(sp)
   17a72: 6e fd        	sd	s11, 184(sp)
   17a74: 4a 8c        	mv	s8, s2
   17a76: d2 8d        	mv	s11, s4
   17a78: 26 89        	mv	s2, s1
   17a7a: 4e 8a        	mv	s4, s3
   17a7c: d7 76 86 c4  	<unknown>
   17a80: ea 89        	mv	s3, s10
   17a82: 23 19 c1 14  	sh	a2, 338(sp)
   17a86: 5a 8d        	mv	s10, s6
   17a88: 93 0c 21 13  	addi	s9, sp, 306
   17a8c: b6 ec        	sd	a3, 88(sp)
   17a8e: be e1        	sd	a5, 192(sp)
   17a90: ba 84        	mv	s1, a4
   17a92: 76 8b        	mv	s6, t4
   17a94: d2 86        	mv	a3, s4
   17a96: 62 86        	mv	a2, s8
   17a98: d6 85        	mv	a1, s5
   17a9a: 6a 85        	mv	a0, s10
   17a9c: ef e0 df d7  	jal	0x16818 <ediv>
   17aa0: 57 70 85 c4  	<unknown>
   17aa4: 07 5c 0c 02  	<unknown>
   17aa8: 27 dc 0b 02  	<unknown>
   17aac: 63 93 09 16  	bnez	s3, 0x17c12 <_ldtoa_r+0xadc>
   17ab0: c6 77        	ld	a5, 112(sp)
   17ab2: 03 57 21 1f  	lhu	a4, 498(sp)
   17ab6: b3 76 f7 00  	and	a3, a4, a5
   17aba: f6 57        	lw	a5, 124(sp)
   17abc: b5 9f        	addw	a5, a5, a3
   17abe: be 86        	mv	a3, a5
   17ac0: 63 52 f0 3c  	blez	a5, 0x17e84 <_ldtoa_r+0xd4e>
   17ac4: 93 07 00 09  	addi	a5, zero, 144
   17ac8: 95 9f        	subw	a5, a5, a3
   17aca: 07 dc 0b 02  	<unknown>
   17ace: 27 dc 0d 02  	<unknown>
   17ad2: 63 55 f0 04  	blez	a5, 0x17b1c <_ldtoa_r+0x9e6>
   17ad6: bd 46        	addi	a3, zero, 15
   17ad8: 63 d2 f6 58  	bge	a3, a5, 0x1805c <_ldtoa_r+0xf26>
   17adc: c1 37        	addiw	a5, a5, -16
   17ade: 1b d6 47 00  	srliw	a2, a5, 4
   17ae2: 9b 05 16 00  	addiw	a1, a2, 1
   17ae6: 93 96 15 00  	slli	a3, a1, 1
   17aea: 1b 16 46 00  	slliw	a2, a2, 4
   17aee: 57 75 80 04  	<unknown>
   17af2: ee 96        	add	a3, a3, s11
   17af4: 91 9f        	subw	a5, a5, a2
   17af6: 57 3c 00 5e  	<unknown>
   17afa: 57 f0 85 04  	<unknown>
   17afe: 27 dc 0d 02  	<unknown>
   17b02: 4a 66        	ld	a2, 144(sp)
   17b04: 86 07        	slli	a5, a5, 1
   17b06: b2 97        	add	a5, a5, a2
   17b08: 83 d7 07 00  	lhu	a5, 0(a5)
   17b0c: 03 d6 06 00  	lhu	a2, 0(a3)
   17b10: f1 8f        	and	a5, a5, a2
   17b12: 23 90 f6 00  	sh	a5, 0(a3)
   17b16: 3d 83        	srli	a4, a4, 15
   17b18: 63 15 07 1c  	bnez	a4, 0x17ce2 <_ldtoa_r+0xbac>
   17b1c: e2 87        	mv	a5, s8
   17b1e: 6e 87        	mv	a4, s11
   17b20: 03 d6 07 00  	lhu	a2, 0(a5)
   17b24: 83 56 07 00  	lhu	a3, 0(a4)
   17b28: 63 17 d6 02  	bne	a2, a3, 0x17b56 <_ldtoa_r+0xa20>
   17b2c: 89 07        	addi	a5, a5, 2
   17b2e: 09 07        	addi	a4, a4, 2
   17b30: e3 98 97 ff  	bne	a5, s9, 0x17b20 <_ldtoa_r+0x9ea>
   17b34: 57 70 85 c4  	<unknown>
   17b38: 07 5c 0c 02  	<unknown>
   17b3c: 27 dc 0a 02  	<unknown>
   17b40: 63 89 09 00  	beqz	s3, 0x17b52 <_ldtoa_r+0xa1c>
   17b44: c6 67        	ld	a5, 80(sp)
   17b46: 57 f0 89 04  	<unknown>
   17b4a: 07 dc 07 02  	<unknown>
   17b4e: 27 5c 0b 02  	<unknown>
   17b52: 3b 89 24 01  	addw	s2, s1, s2
   17b56: a6 77        	ld	a5, 104(sp)
   17b58: 51 0d        	addi	s10, s10, 20
   17b5a: 85 80        	srli	s1, s1, 1
   17b5c: e3 1c fd f2  	bne	s10, a5, 0x17a94 <_ldtoa_r+0x95e>
   17b60: 83 57 21 15  	lhu	a5, 338(sp)
   17b64: 83 56 21 11  	lhu	a3, 274(sp)
   17b68: 57 70 85 c4  	<unknown>
   17b6c: ca 84        	mv	s1, s2
   17b6e: b5 9f        	addw	a5, a5, a3
   17b70: f1 76        	lui	a3, 1048572
   17b72: 9b 86 26 f7  	addiw	a3, a3, -142
   17b76: b5 9f        	addw	a5, a5, a3
   17b78: 23 19 f1 14  	sh	a5, 338(sp)
   17b7c: 4e 8d        	mv	s10, s3
   17b7e: 62 89        	mv	s2, s8
   17b80: d2 89        	mv	s3, s4
   17b82: 4a 7c        	ld	s8, 176(sp)
   17b84: 6e 8a        	mv	s4, s11
   17b86: 07 dc 0a 02  	<unknown>
   17b8a: ea 7d        	ld	s11, 184(sp)
   17b8c: 27 5c 04 02  	<unknown>
   17b90: 63 12 0d 38  	bnez	s10, 0x17f14 <_ldtoa_r+0xdde>
   17b94: 07 dc 0d 02  	<unknown>
   17b98: 27 5c 09 02  	<unknown>
   17b9c: 37 2b 02 00  	lui	s6, 34
   17ba0: 05 6d        	lui	s10, 1
   17ba2: 13 0b 0b 06  	addi	s6, s6, 96
   17ba6: b7 2c 02 00  	lui	s9, 34
   17baa: 01 a8        	j	0x17bba <_ldtoa_r+0xa84>
   17bac: 93 87 0c 15  	addi	a5, s9, 336
   17bb0: 13 5d 1d 00  	srli	s10, s10, 1
   17bb4: 63 0b fb f0  	beq	s6, a5, 0x172ca <_ldtoa_r+0x194>
   17bb8: 51 0b        	addi	s6, s6, 20
   17bba: d6 85        	mv	a1, s5
   17bbc: 13 85 0c 15  	addi	a0, s9, 336
   17bc0: ef e0 2f cf  	jal	0x160b2 <ecmp>
   17bc4: 63 43 a0 f0  	bgtz	a0, 0x172ca <_ldtoa_r+0x194>
   17bc8: d6 85        	mv	a1, s5
   17bca: 5a 85        	mv	a0, s6
   17bcc: ef e0 6f ce  	jal	0x160b2 <ecmp>
   17bd0: e3 4e a0 fc  	bgtz	a0, 0x17bac <_ldtoa_r+0xa76>
   17bd4: ce 86        	mv	a3, s3
   17bd6: 56 86        	mv	a2, s5
   17bd8: d6 85        	mv	a1, s5
   17bda: 5a 85        	mv	a0, s6
   17bdc: ef e0 df c3  	jal	0x16818 <ediv>
   17be0: ce 86        	mv	a3, s3
   17be2: 4a 86        	mv	a2, s2
   17be4: ca 85        	mv	a1, s2
   17be6: 5a 85        	mv	a0, s6
   17be8: ef f0 0f 93  	jal	0x16d18 <emul>
   17bec: bb 04 9d 00  	addw	s1, s10, s1
   17bf0: 75 bf        	j	0x17bac <_ldtoa_r+0xa76>
   17bf2: 13 0a 31 1e  	addi	s4, sp, 483
   17bf6: 01 46        	mv	a2, zero
   17bf8: e3 92 07 be  	bnez	a5, 0x177dc <_ldtoa_r+0x6a6>
   17bfc: 61 b0        	j	0x17484 <_ldtoa_r+0x34e>
   17bfe: 93 0b 01 1e  	addi	s7, sp, 480
   17c02: b7 25 02 00  	lui	a1, 34
   17c06: 93 85 05 e9  	addi	a1, a1, -368
   17c0a: 5e 85        	mv	a0, s7
   17c0c: ef 10 90 5f  	jal	0x19a04 <sprintf>
   17c10: 71 b4        	j	0x1769c <_ldtoa_r+0x566>
   17c12: c6 67        	ld	a5, 80(sp)
   17c14: 57 f0 89 04  	<unknown>
   17c18: 07 dc 07 02  	<unknown>
   17c1c: 8a 67        	ld	a5, 128(sp)
   17c1e: 27 dc 07 02  	<unknown>
   17c22: c6 77        	ld	a5, 112(sp)
   17c24: 03 57 21 1f  	lhu	a4, 498(sp)
   17c28: b3 76 f7 00  	and	a3, a4, a5
   17c2c: f6 57        	lw	a5, 124(sp)
   17c2e: b5 9f        	addw	a5, a5, a3
   17c30: be 86        	mv	a3, a5
   17c32: 63 52 f0 26  	blez	a5, 0x17e96 <_ldtoa_r+0xd60>
   17c36: 93 07 00 09  	addi	a5, zero, 144
   17c3a: 57 70 85 c4  	<unknown>
   17c3e: 95 9f        	subw	a5, a5, a3
   17c40: 07 dc 0b 02  	<unknown>
   17c44: 27 dc 0d 02  	<unknown>
   17c48: 8a 66        	ld	a3, 128(sp)
   17c4a: 57 f0 89 04  	<unknown>
   17c4e: 07 dc 06 02  	<unknown>
   17c52: aa 66        	ld	a3, 136(sp)
   17c54: 27 dc 06 02  	<unknown>
   17c58: ad bd        	j	0x17ad2 <_ldtoa_r+0x99c>
   17c5a: de 87        	mv	a5, s7
   17c5c: 23 80 07 00  	sb	zero, 0(a5)
   17c60: f9 b2        	j	0x1762e <_ldtoa_r+0x4f8>
   17c62: 86 67        	ld	a5, 64(sp)
   17c64: a1 66        	lui	a3, 8
   17c66: fd 16        	addi	a3, a3, -1
   17c68: 33 f6 d7 00  	and	a2, a5, a3
   17c6c: b7 27 02 00  	lui	a5, 34
   17c70: 23 19 c1 10  	sh	a2, 274(sp)
   17c74: 93 8d 87 16  	addi	s11, a5, 360
   17c78: 13 09 01 12  	addi	s2, sp, 288
   17c7c: 07 dc 0d 02  	<unknown>
   17c80: b3 85 ed 00  	add	a1, s11, a4
   17c84: 27 5c 09 02  	<unknown>
   17c88: 4a 97        	add	a4, a4, s2
   17c8a: ae f0        	sd	a1, 96(sp)
   17c8c: ba e8        	sd	a4, 80(sp)
   17c8e: 29 c6        	beqz	a2, 0x17cd8 <_ldtoa_r+0xba2>
   17c90: 63 01 d6 22  	beq	a2, a3, 0x17eb2 <_ldtoa_r+0xd7c>
   17c94: 41 67        	lui	a4, 16
   17c96: 7d 17        	addi	a4, a4, -1
   17c98: ba e0        	sd	a4, 64(sp)
   17c9a: e5 b2        	j	0x17682 <_ldtoa_r+0x54c>
   17c9c: 06 66        	ld	a2, 64(sp)
   17c9e: a1 66        	lui	a3, 8
   17ca0: b7 27 02 00  	lui	a5, 34
   17ca4: 93 8d 87 16  	addi	s11, a5, 360
   17ca8: 13 09 01 12  	addi	s2, sp, 288
   17cac: fd 16        	addi	a3, a3, -1
   17cae: b3 85 ed 00  	add	a1, s11, a4
   17cb2: 75 8e        	and	a2, a2, a3
   17cb4: 4a 97        	add	a4, a4, s2
   17cb6: 23 19 c1 10  	sh	a2, 274(sp)
   17cba: ae f0        	sd	a1, 96(sp)
   17cbc: ba e8        	sd	a4, 80(sp)
   17cbe: 57 70 85 c4  	<unknown>
   17cc2: 07 dc 0d 02  	<unknown>
   17cc6: 27 5c 09 02  	<unknown>
   17cca: 57 70 8d 04  	<unknown>
   17cce: 07 dc 05 02  	<unknown>
   17cd2: 27 5c 07 02  	<unknown>
   17cd6: 4d fe        	bnez	a2, 0x17c90 <_ldtoa_r+0xb5a>
   17cd8: 41 67        	lui	a4, 16
   17cda: 7d 17        	addi	a4, a4, -1
   17cdc: ba e0        	sd	a4, 64(sp)
   17cde: 6f f0 4f dc  	j	0x172a2 <_ldtoa_r+0x16c>
   17ce2: de 87        	mv	a5, s7
   17ce4: 6e 87        	mv	a4, s11
   17ce6: 39 a0        	j	0x17cf4 <_ldtoa_r+0xbbe>
   17ce8: 89 07        	addi	a5, a5, 2
   17cea: 93 06 21 1f  	addi	a3, sp, 498
   17cee: 09 07        	addi	a4, a4, 2
   17cf0: e3 86 f6 e2  	beq	a3, a5, 0x17b1c <_ldtoa_r+0x9e6>
   17cf4: 03 d6 07 00  	lhu	a2, 0(a5)
   17cf8: 83 56 07 00  	lhu	a3, 0(a4)
   17cfc: e3 06 d6 fe  	beq	a2, a3, 0x17ce8 <_ldtoa_r+0xbb2>
   17d00: 83 57 21 17  	lhu	a5, 370(sp)
   17d04: 93 c7 f7 ff  	not	a5, a5
   17d08: 13 97 17 03  	slli	a4, a5, 49
   17d0c: 63 0a 07 34  	beqz	a4, 0x18060 <_ldtoa_r+0xf2a>
   17d10: 8e 67        	ld	a5, 192(sp)
   17d12: 0c 03        	addi	a1, sp, 384
   17d14: 13 85 87 16  	addi	a0, a5, 360
   17d18: ef e0 2f ae  	jal	0x15ffa <emovi>
   17d1c: 0c 13        	addi	a1, sp, 416
   17d1e: 6e 85        	mv	a0, s11
   17d20: ef e0 af ad  	jal	0x15ffa <emovi>
   17d24: 83 57 01 18  	lhu	a5, 384(sp)
   17d28: 03 57 21 1a  	lhu	a4, 418(sp)
   17d2c: 03 56 21 18  	lhu	a2, 386(sp)
   17d30: 93 c7 f7 ff  	not	a5, a5
   17d34: 23 10 f1 18  	sh	a5, 384(sp)
   17d38: b3 05 e6 40  	sub	a1, a2, a4
   17d3c: ba 86        	mv	a3, a4
   17d3e: 63 57 b0 04  	blez	a1, 0x17d8c <_ldtoa_r+0xc56>
   17d42: 1c 13        	addi	a5, sp, 416
   17d44: 57 70 86 c4  	<unknown>
   17d48: 07 dc 07 02  	<unknown>
   17d4c: 9c 03        	addi	a5, sp, 448
   17d4e: 27 dc 07 02  	<unknown>
   17d52: e6 65        	ld	a1, 88(sp)
   17d54: b1 46        	addi	a3, zero, 12
   17d56: b3 87 b6 40  	sub	a5, a3, a1
   17d5a: 63 93 b6 38  	bne	a3, a1, 0x180e0 <_ldtoa_r+0xfaa>
   17d5e: 23 1c 01 1c  	sh	zero, 472(sp)
   17d62: 1c 03        	addi	a5, sp, 384
   17d64: 07 dc 07 02  	<unknown>
   17d68: 1c 13        	addi	a5, sp, 416
   17d6a: 27 dc 07 02  	<unknown>
   17d6e: 23 1c 01 1a  	sh	zero, 440(sp)
   17d72: 9c 03        	addi	a5, sp, 448
   17d74: 07 dc 07 02  	<unknown>
   17d78: 1c 03        	addi	a5, sp, 384
   17d7a: 27 dc 07 02  	<unknown>
   17d7e: 83 56 21 1a  	lhu	a3, 418(sp)
   17d82: b3 05 c7 40  	sub	a1, a4, a2
   17d86: 23 1c 01 18  	sh	zero, 408(sp)
   17d8a: 36 87        	mv	a4, a3
   17d8c: 63 86 05 30  	beqz	a1, 0x18098 <_ldtoa_r+0xf62>
   17d90: b6 e5        	sd	a3, 200(sp)
   17d92: 93 07 f0 f6  	addi	a5, zero, -145
   17d96: 63 cd f5 04  	blt	a1, a5, 0x17df0 <_ldtoa_r+0xcba>
   17d9a: 81 25        	sext.w	a1, a1
   17d9c: 08 03        	addi	a0, sp, 384
   17d9e: ef e0 2f 8f  	jal	0x15e90 <eshift.part.0>
   17da2: ae 66        	ld	a3, 200(sp)
   17da4: aa 85        	mv	a1, a0
   17da6: 03 57 01 18  	lhu	a4, 384(sp)
   17daa: 83 57 01 1a  	lhu	a5, 416(sp)
   17dae: 63 01 f7 3a  	beq	a4, a5, 0x18150 <_ldtoa_r+0x101a>
   17db2: 81 47        	mv	a5, zero
   17db4: 93 0e 81 19  	addi	t4, sp, 408
   17db8: 28 1b        	addi	a0, sp, 440
   17dba: 13 06 21 1a  	addi	a2, sp, 418
   17dbe: 03 57 05 00  	lhu	a4, 0(a0)
   17dc2: 03 d8 0e 00  	lhu	a6, 0(t4)
   17dc6: 79 15        	addi	a0, a0, -2
   17dc8: f9 1e        	addi	t4, t4, -2
   17dca: 33 07 07 41  	sub	a4, a4, a6
   17dce: b3 07 f7 40  	sub	a5, a4, a5
   17dd2: 13 d7 07 01  	srli	a4, a5, 16
   17dd6: 23 11 f5 00  	sh	a5, 2(a0)
   17dda: 93 77 17 00  	andi	a5, a4, 1
   17dde: e3 10 c5 fe  	bne	a0, a2, 0x17dbe <_ldtoa_r+0xc88>
   17de2: 05 46        	addi	a2, zero, 1
   17de4: d2 87        	mv	a5, s4
   17de6: 13 07 00 04  	addi	a4, zero, 64
   17dea: 08 13        	addi	a0, sp, 416
   17dec: ef e0 ef ba  	jal	0x1619a <emdnorm>
   17df0: ee 85        	mv	a1, s11
   17df2: 08 13        	addi	a0, sp, 416
   17df4: ef d0 5f ec  	jal	0x15cb8 <emovo.constprop.0>
   17df8: 15 b3        	j	0x17b1c <_ldtoa_r+0x9e6>
   17dfa: c2 77        	ld	a5, 48(sp)
   17dfc: 23 00 01 1e  	sb	zero, 480(sp)
   17e00: 5e 84        	mv	s0, s7
   17e02: 23 a0 07 00  	sw	zero, 0(a5)
   17e06: 1d be        	j	0x1793c <_ldtoa_r+0x806>
   17e08: 85 25        	addiw	a1, a1, 1
   17e0a: 23 00 b5 00  	sb	a1, 0(a0)
   17e0e: f9 b2        	j	0x177dc <_ldtoa_r+0x6a6>
   17e10: 83 47 f5 ff  	lbu	a5, -1(a0)
   17e14: 13 06 80 03  	addi	a2, zero, 56
   17e18: 63 7a f6 12  	bgeu	a2, a5, 0x17f4c <_ldtoa_r+0xe16>
   17e1c: 93 07 10 03  	addi	a5, zero, 49
   17e20: 85 24        	addiw	s1, s1, 1
   17e22: a3 0f f5 fe  	sb	a5, -1(a0)
   17e26: 5d ba        	j	0x177dc <_ldtoa_r+0x6a6>
   17e28: 9c 11        	addi	a5, sp, 224
   17e2a: 13 06 21 0f  	addi	a2, sp, 242
   17e2e: 19 a0        	j	0x17e34 <_ldtoa_r+0xcfe>
   17e30: e3 88 c7 9c  	beq	a5, a2, 0x17800 <_ldtoa_r+0x6ca>
   17e34: 03 d7 07 00  	lhu	a4, 0(a5)
   17e38: 89 07        	addi	a5, a5, 2
   17e3a: 7d db        	beqz	a4, 0x17e30 <_ldtoa_r+0xcfa>
   17e3c: 6f f0 8f fe  	j	0x17624 <_ldtoa_r+0x4ee>
   17e40: 73 27 20 c2  	csrr	a4, vlenb
   17e44: 13 07 07 09  	addi	a4, a4, 144
   17e48: 94 09        	addi	a3, sp, 208
   17e4a: 57 70 8d 04  	<unknown>
   17e4e: 36 97        	add	a4, a4, a3
   17e50: 07 5c 0c 02  	<unknown>
   17e54: 27 5c 07 02  	<unknown>
   17e58: 57 70 85 c4  	<unknown>
   17e5c: 07 dc 0d 02  	<unknown>
   17e60: 27 5c 09 02  	<unknown>
   17e64: 06 77        	ld	a4, 96(sp)
   17e66: 57 70 8d 04  	<unknown>
   17e6a: 07 5c 07 02  	<unknown>
   17e6e: 46 67        	ld	a4, 80(sp)
   17e70: 27 5c 07 02  	<unknown>
   17e74: 2d bc        	j	0x178ae <_ldtoa_r+0x778>
   17e76: 83 55 c1 26  	lhu	a1, 620(sp)
   17e7a: 13 0a 21 1e  	addi	s4, sp, 482
   17e7e: 93 07 31 1e  	addi	a5, sp, 483
   17e82: 01 ba        	j	0x17792 <_ldtoa_r+0x65c>
   17e84: d7 77 80 04  	<unknown>
   17e88: 57 3c 00 5e  	<unknown>
   17e8c: 57 70 85 c4  	<unknown>
   17e90: 27 dc 0d 02  	<unknown>
   17e94: 49 b1        	j	0x17b16 <_ldtoa_r+0x9e0>
   17e96: d7 77 80 04  	<unknown>
   17e9a: 57 3c 00 5e  	<unknown>
   17e9e: 57 70 85 c4  	<unknown>
   17ea2: 27 dc 0d 02  	<unknown>
   17ea6: aa 67        	ld	a5, 136(sp)
   17ea8: 57 f0 89 04  	<unknown>
   17eac: 27 dc 07 02  	<unknown>
   17eb0: 9d b1        	j	0x17b16 <_ldtoa_r+0x9e0>
   17eb2: 93 0b 01 1e  	addi	s7, sp, 480
   17eb6: b7 25 02 00  	lui	a1, 34
   17eba: 93 85 05 e8  	addi	a1, a1, -384
   17ebe: 5e 85        	mv	a0, s7
   17ec0: ef 10 50 34  	jal	0x19a04 <sprintf>
   17ec4: 6f f0 8f fd  	j	0x1769c <_ldtoa_r+0x566>
   17ec8: ca 85        	mv	a1, s2
   17eca: 22 85        	mv	a0, s0
   17ecc: be e8        	sd	a5, 80(sp)
   17ece: b2 e0        	sd	a2, 64(sp)
   17ed0: ef d0 9f de  	jal	0x15cb8 <emovo.constprop.0>
   17ed4: b7 25 02 00  	lui	a1, 34
   17ed8: 93 85 05 18  	addi	a1, a1, 384
   17edc: 4a 85        	mv	a0, s2
   17ede: ef e0 4f 9d  	jal	0x160b2 <ecmp>
   17ee2: 06 66        	ld	a2, 64(sp)
   17ee4: c6 67        	ld	a5, 80(sp)
   17ee6: e3 1c 05 8a  	bnez	a0, 0x1779e <_ldtoa_r+0x668>
   17eea: e3 49 06 8e  	bltz	a2, 0x177dc <_ldtoa_r+0x6a6>
   17eee: 83 c5 e7 ff  	lbu	a1, -2(a5)
   17ef2: 13 85 e7 ff  	addi	a0, a5, -2
   17ef6: 13 86 25 fd  	addi	a2, a1, -46
   17efa: 13 36 16 00  	seqz	a2, a2
   17efe: 13 46 f6 ff  	not	a2, a2
   17f02: 52 96        	add	a2, a2, s4
   17f04: 03 46 06 00  	lbu	a2, 0(a2)
   17f08: 93 f5 f5 07  	andi	a1, a1, 127
   17f0c: 05 8a        	andi	a2, a2, 1
   17f0e: e3 10 06 8a  	bnez	a2, 0x177ae <_ldtoa_r+0x678>
   17f12: e9 b0        	j	0x177dc <_ldtoa_r+0x6a6>
   17f14: 57 70 8d 04  	<unknown>
   17f18: 07 5c 0b 02  	<unknown>
   17f1c: 27 5c 0c 02  	<unknown>
   17f20: 57 70 85 c4  	<unknown>
   17f24: 07 dc 0d 02  	<unknown>
   17f28: 27 5c 09 02  	<unknown>
   17f2c: 86 77        	ld	a5, 96(sp)
   17f2e: 57 70 8d 04  	<unknown>
   17f32: 07 dc 07 02  	<unknown>
   17f36: c6 67        	ld	a5, 80(sp)
   17f38: 27 dc 07 02  	<unknown>
   17f3c: 85 b1        	j	0x17b9c <_ldtoa_r+0xa66>
   17f3e: 57 70 8d 04  	<unknown>
   17f42: 07 5c 0c 02  	<unknown>
   17f46: 27 dc 0e 02  	<unknown>
   17f4a: d9 b4        	j	0x17a10 <_ldtoa_r+0x8da>
   17f4c: 85 27        	addiw	a5, a5, 1
   17f4e: a3 0f f5 fe  	sb	a5, -1(a0)
   17f52: 69 b0        	j	0x177dc <_ldtoa_r+0x6a6>
   17f54: 13 0a 01 16  	addi	s4, sp, 352
   17f58: d2 85        	mv	a1, s4
   17f5a: 22 85        	mv	a0, s0
   17f5c: be f4        	sd	a5, 104(sp)
   17f5e: ef e0 cf 89  	jal	0x15ffa <emovi>
   17f62: 73 27 20 c2  	csrr	a4, vlenb
   17f66: 93 0a 01 14  	addi	s5, sp, 320
   17f6a: b3 86 ea 00  	add	a3, s5, a4
   17f6e: b3 0c ea 00  	add	s9, s4, a4
   17f72: b1 45        	addi	a1, zero, 12
   17f74: 11 67        	lui	a4, 4
   17f76: d7 77 86 c4  	<unknown>
   17f7a: ca f8        	sd	s2, 112(sp)
   17f7c: be ec        	sd	a5, 88(sp)
   17f7e: 81 44        	mv	s1, zero
   17f80: 33 8b f5 40  	sub	s6, a1, a5
   17f84: 93 09 21 16  	addi	s3, sp, 354
   17f88: 93 0b e7 ff  	addi	s7, a4, -2
   17f8c: 36 89        	mv	s2, a3
   17f8e: 83 57 81 17  	lhu	a5, 376(sp)
   17f92: 9d 8b        	andi	a5, a5, 7
   17f94: c5 eb        	bnez	a5, 0x18044 <_ldtoa_r+0xf0e>
   17f96: 57 70 86 c4  	<unknown>
   17f9a: 07 5c 0a 02  	<unknown>
   17f9e: 27 dc 0a 02  	<unknown>
   17fa2: 63 08 0b 00  	beqz	s6, 0x17fb2 <_ldtoa_r+0xe7c>
   17fa6: 57 70 8b 04  	<unknown>
   17faa: 07 dc 0c 02  	<unknown>
   17fae: 27 5c 09 02  	<unknown>
   17fb2: 56 85        	mv	a0, s5
   17fb4: 23 1c 01 14  	sh	zero, 344(sp)
   17fb8: ef d0 df bf  	jal	0x15bb4 <eshdn1>
   17fbc: 56 85        	mv	a0, s5
   17fbe: ef d0 7f bf  	jal	0x15bb4 <eshdn1>
   17fc2: 01 47        	mv	a4, zero
   17fc4: b0 0a        	addi	a2, sp, 344
   17fc6: b4 1a        	addi	a3, sp, 376
   17fc8: 83 55 06 00  	lhu	a1, 0(a2)
   17fcc: 83 d7 06 00  	lhu	a5, 0(a3)
   17fd0: 79 16        	addi	a2, a2, -2
   17fd2: f9 16        	addi	a3, a3, -2
   17fd4: ae 97        	add	a5, a5, a1
   17fd6: ba 97        	add	a5, a5, a4
   17fd8: 13 d7 07 01  	srli	a4, a5, 16
   17fdc: 23 11 f6 00  	sh	a5, 2(a2)
   17fe0: 05 8b        	andi	a4, a4, 1
   17fe2: e3 93 36 ff  	bne	a3, s3, 0x17fc8 <_ldtoa_r+0xe92>
   17fe6: 83 57 21 14  	lhu	a5, 322(sp)
   17fea: 03 57 41 14  	lhu	a4, 324(sp)
   17fee: 8d 27        	addiw	a5, a5, 3
   17ff0: 23 11 f1 14  	sh	a5, 322(sp)
   17ff4: 01 cf        	beqz	a4, 0x1800c <_ldtoa_r+0xed6>
   17ff6: 56 85        	mv	a0, s5
   17ff8: ef d0 df bb  	jal	0x15bb4 <eshdn1>
   17ffc: 83 57 21 14  	lhu	a5, 322(sp)
   18000: 03 57 41 14  	lhu	a4, 324(sp)
   18004: 85 27        	addiw	a5, a5, 1
   18006: 23 11 f1 14  	sh	a5, 322(sp)
   1800a: 75 f7        	bnez	a4, 0x17ff6 <_ldtoa_r+0xec0>
   1800c: 83 57 81 15  	lhu	a5, 344(sp)
   18010: 95 eb        	bnez	a5, 0x18044 <_ldtoa_r+0xf0e>
   18012: 83 57 21 14  	lhu	a5, 322(sp)
   18016: 63 e7 fb 02  	bltu	s7, a5, 0x18044 <_ldtoa_r+0xf0e>
   1801a: 57 70 86 c4  	<unknown>
   1801e: 07 dc 0a 02  	<unknown>
   18022: 27 5c 0a 02  	<unknown>
   18026: 63 08 0b 00  	beqz	s6, 0x18036 <_ldtoa_r+0xf00>
   1802a: 57 70 8b 04  	<unknown>
   1802e: 07 5c 09 02  	<unknown>
   18032: 27 dc 0c 02  	<unknown>
   18036: 23 1c 01 16  	sh	zero, 376(sp)
   1803a: fd 34        	addiw	s1, s1, -1
   1803c: 93 07 50 fd  	addi	a5, zero, -43
   18040: e3 97 f4 f4  	bne	s1, a5, 0x17f8e <_ldtoa_r+0xe58>
   18044: a6 77        	ld	a5, 104(sp)
   18046: a2 85        	mv	a1, s0
   18048: 52 85        	mv	a0, s4
   1804a: be f4        	sd	a5, 104(sp)
   1804c: 46 79        	ld	s2, 112(sp)
   1804e: ef d0 bf c6  	jal	0x15cb8 <emovo.constprop.0>
   18052: a6 77        	ld	a5, 104(sp)
   18054: 93 09 01 22  	addi	s3, sp, 544
   18058: 6f f0 ff 83  	j	0x17896 <_ldtoa_r+0x760>
   1805c: ee 86        	mv	a3, s11
   1805e: 55 b4        	j	0x17b02 <_ldtoa_r+0x9cc>
   18060: ee 87        	mv	a5, s11
   18062: 13 07 21 17  	addi	a4, sp, 370
   18066: 83 d6 07 00  	lhu	a3, 0(a5)
   1806a: 89 07        	addi	a5, a5, 2
   1806c: e3 98 06 aa  	bnez	a3, 0x17b1c <_ldtoa_r+0x9e6>
   18070: e3 9b e7 fe  	bne	a5, a4, 0x18066 <_ldtoa_r+0xf30>
   18074: ee 87        	mv	a5, s11
   18076: 83 d6 07 00  	lhu	a3, 0(a5)
   1807a: 89 07        	addi	a5, a5, 2
   1807c: e3 9a 06 c8  	bnez	a3, 0x17d10 <_ldtoa_r+0xbda>
   18080: e3 9b e7 fe  	bne	a5, a4, 0x18076 <_ldtoa_r+0xf40>
   18084: 61 bc        	j	0x17b1c <_ldtoa_r+0x9e6>
   18086: 57 77 86 c4  	<unknown>
   1808a: 13 0a 01 16  	addi	s4, sp, 352
   1808e: 93 09 01 22  	addi	s3, sp, 544
   18092: ba ec        	sd	a4, 88(sp)
   18094: 6f f0 3f 80  	j	0x17896 <_ldtoa_r+0x760>
   18098: 50 13        	addi	a2, sp, 420
   1809a: 5c 03        	addi	a5, sp, 388
   1809c: 93 0e a1 19  	addi	t4, sp, 410
   180a0: 03 d5 07 00  	lhu	a0, 0(a5)
   180a4: 83 55 06 00  	lhu	a1, 0(a2)
   180a8: 89 07        	addi	a5, a5, 2
   180aa: 09 06        	addi	a2, a2, 2
   180ac: 63 19 b5 0c  	bne	a0, a1, 0x1817e <_ldtoa_r+0x1048>
   180b0: e3 98 d7 ff  	bne	a5, t4, 0x180a0 <_ldtoa_r+0xf6a>
   180b4: 03 56 01 18  	lhu	a2, 384(sp)
   180b8: 83 57 01 1a  	lhu	a5, 416(sp)
   180bc: 63 05 f6 10  	beq	a2, a5, 0x181c6 <_ldtoa_r+0x1090>
   180c0: d7 77 80 04  	<unknown>
   180c4: 57 3c 00 5e  	<unknown>
   180c8: 57 70 85 c4  	<unknown>
   180cc: 27 dc 0d 02  	<unknown>
   180d0: e3 86 09 a4  	beqz	s3, 0x17b1c <_ldtoa_r+0x9e6>
   180d4: aa 67        	ld	a5, 136(sp)
   180d6: 57 f0 89 04  	<unknown>
   180da: 27 dc 07 02  	<unknown>
   180de: 3d bc        	j	0x17b1c <_ldtoa_r+0x9e6>
   180e0: ea 66        	ld	a3, 152(sp)
   180e2: aa 75        	ld	a1, 168(sp)
   180e4: 57 f0 87 04  	<unknown>
   180e8: 07 dc 06 02  	<unknown>
   180ec: 27 dc 05 02  	<unknown>
   180f0: 23 1c 01 1c  	sh	zero, 472(sp)
   180f4: 08 03        	addi	a0, sp, 384
   180f6: 57 70 86 c4  	<unknown>
   180fa: 07 5c 05 02  	<unknown>
   180fe: 08 13        	addi	a0, sp, 416
   18100: 27 5c 05 02  	<unknown>
   18104: 0a 75        	ld	a0, 160(sp)
   18106: 57 f0 87 04  	<unknown>
   1810a: 07 5c 05 02  	<unknown>
   1810e: 27 dc 06 02  	<unknown>
   18112: 23 1c 01 1a  	sh	zero, 440(sp)
   18116: 94 03        	addi	a3, sp, 448
   18118: 57 70 86 c4  	<unknown>
   1811c: 07 dc 06 02  	<unknown>
   18120: 14 03        	addi	a3, sp, 384
   18122: 27 dc 06 02  	<unknown>
   18126: 57 f0 87 04  	<unknown>
   1812a: 07 dc 05 02  	<unknown>
   1812e: 27 5c 05 02  	<unknown>
   18132: b1 b1        	j	0x17d7e <_ldtoa_r+0xc48>
   18134: b7 26 02 00  	lui	a3, 34
   18138: 85 65        	lui	a1, 1
   1813a: 37 25 02 00  	lui	a0, 34
   1813e: 93 86 06 eb  	addi	a3, a3, -336
   18142: 01 46        	mv	a2, zero
   18144: 93 85 e5 b6  	addi	a1, a1, -1170
   18148: 13 05 85 ec  	addi	a0, a0, -312
   1814c: ef 40 e0 6a  	jal	0x1c7fa <__assert_func>
   18150: 81 47        	mv	a5, zero
   18152: 28 1b        	addi	a0, sp, 440
   18154: 30 0b        	addi	a2, sp, 408
   18156: 93 0e 21 18  	addi	t4, sp, 386
   1815a: 03 58 05 00  	lhu	a6, 0(a0)
   1815e: 03 57 06 00  	lhu	a4, 0(a2)
   18162: 79 15        	addi	a0, a0, -2
   18164: 79 16        	addi	a2, a2, -2
   18166: 42 97        	add	a4, a4, a6
   18168: ba 97        	add	a5, a5, a4
   1816a: 13 d7 07 01  	srli	a4, a5, 16
   1816e: 23 11 f5 00  	sh	a5, 2(a0)
   18172: 93 77 17 00  	andi	a5, a4, 1
   18176: e3 12 d6 ff  	bne	a2, t4, 0x1815a <_ldtoa_r+0x1024>
   1817a: 01 46        	mv	a2, zero
   1817c: a5 b1        	j	0x17de4 <_ldtoa_r+0xcae>
   1817e: 63 f2 a5 04  	bgeu	a1, a0, 0x181c2 <_ldtoa_r+0x108c>
   18182: 1c 13        	addi	a5, sp, 416
   18184: 57 70 86 c4  	<unknown>
   18188: 07 dc 07 02  	<unknown>
   1818c: 9c 03        	addi	a5, sp, 448
   1818e: 27 dc 07 02  	<unknown>
   18192: 66 66        	ld	a2, 88(sp)
   18194: 31 47        	addi	a4, zero, 12
   18196: b3 07 c7 40  	sub	a5, a4, a2
   1819a: 63 18 c7 04  	bne	a4, a2, 0x181ea <_ldtoa_r+0x10b4>
   1819e: 23 1c 01 1c  	sh	zero, 472(sp)
   181a2: 1c 03        	addi	a5, sp, 384
   181a4: 07 dc 07 02  	<unknown>
   181a8: 1c 13        	addi	a5, sp, 416
   181aa: 27 dc 07 02  	<unknown>
   181ae: 23 1c 01 1a  	sh	zero, 440(sp)
   181b2: 9c 03        	addi	a5, sp, 448
   181b4: 07 dc 07 02  	<unknown>
   181b8: 1c 03        	addi	a5, sp, 384
   181ba: 27 dc 07 02  	<unknown>
   181be: 23 1c 01 18  	sh	zero, 408(sp)
   181c2: 81 45        	mv	a1, zero
   181c4: cd b6        	j	0x17da6 <_ldtoa_r+0xc70>
   181c6: b6 87        	mv	a5, a3
   181c8: 89 e6        	bnez	a3, 0x181d2 <_ldtoa_r+0x109c>
   181ca: 03 16 61 1a  	lh	a2, 422(sp)
   181ce: 63 5a 06 06  	bgez	a2, 0x18242 <_ldtoa_r+0x110c>
   181d2: 93 07 21 1a  	addi	a5, sp, 418
   181d6: 30 1b        	addi	a2, sp, 440
   181d8: 83 d5 07 00  	lhu	a1, 0(a5)
   181dc: c5 e1        	bnez	a1, 0x1827c <_ldtoa_r+0x1146>
   181de: 89 07        	addi	a5, a5, 2
   181e0: e3 1c f6 fe  	bne	a2, a5, 0x181d8 <_ldtoa_r+0x10a2>
   181e4: 23 11 e1 1a  	sh	a4, 418(sp)
   181e8: 21 b1        	j	0x17df0 <_ldtoa_r+0xcba>
   181ea: 6a 67        	ld	a4, 152(sp)
   181ec: 2a 76        	ld	a2, 168(sp)
   181ee: 57 f0 87 04  	<unknown>
   181f2: 07 5c 07 02  	<unknown>
   181f6: 27 5c 06 02  	<unknown>
   181fa: 23 1c 01 1c  	sh	zero, 472(sp)
   181fe: 0c 03        	addi	a1, sp, 384
   18200: 57 70 86 c4  	<unknown>
   18204: 07 dc 05 02  	<unknown>
   18208: 0c 13        	addi	a1, sp, 416
   1820a: 27 dc 05 02  	<unknown>
   1820e: 8a 75        	ld	a1, 160(sp)
   18210: 57 f0 87 04  	<unknown>
   18214: 07 dc 05 02  	<unknown>
   18218: 27 5c 07 02  	<unknown>
   1821c: 23 1c 01 1a  	sh	zero, 440(sp)
   18220: 98 03        	addi	a4, sp, 448
   18222: 57 70 86 c4  	<unknown>
   18226: 07 5c 07 02  	<unknown>
   1822a: 18 03        	addi	a4, sp, 384
   1822c: 27 5c 07 02  	<unknown>
   18230: 57 f0 87 04  	<unknown>
   18234: 07 5c 06 02  	<unknown>
   18238: 27 dc 05 02  	<unknown>
   1823c: 23 1c 01 18  	sh	zero, 408(sp)
   18240: 49 b7        	j	0x181c2 <_ldtoa_r+0x108c>
   18242: 34 1b        	addi	a3, sp, 440
   18244: 13 06 21 1a  	addi	a2, sp, 418
   18248: 03 d7 06 00  	lhu	a4, 0(a3)
   1824c: 93 55 f7 00  	srli	a1, a4, 15
   18250: 99 c1        	beqz	a1, 0x18256 <_ldtoa_r+0x1120>
   18252: 93 e7 17 00  	ori	a5, a5, 1
   18256: 1b 17 17 00  	slliw	a4, a4, 1
   1825a: 42 17        	slli	a4, a4, 48
   1825c: 93 f5 27 00  	andi	a1, a5, 2
   18260: 41 93        	srli	a4, a4, 48
   18262: 99 c1        	beqz	a1, 0x18268 <_ldtoa_r+0x1132>
   18264: 13 67 17 00  	ori	a4, a4, 1
   18268: 9b 97 17 00  	slliw	a5, a5, 1
   1826c: 23 90 e6 00  	sh	a4, 0(a3)
   18270: c2 17        	slli	a5, a5, 48
   18272: f9 16        	addi	a3, a3, -2
   18274: c1 93        	srli	a5, a5, 48
   18276: e3 99 c6 fc  	bne	a3, a2, 0x18248 <_ldtoa_r+0x1112>
   1827a: 9d be        	j	0x17df0 <_ldtoa_r+0xcba>
   1827c: 13 87 16 00  	addi	a4, a3, 1
   18280: 23 11 e1 1a  	sh	a4, 418(sp)
   18284: b5 b6        	j	0x17df0 <_ldtoa_r+0xcba>

0000000000018286 <_ldcheck>:
   18286: 18 61        	ld	a4, 0(a0)
   18288: 1c 65        	ld	a5, 8(a0)
   1828a: 39 71        	addi	sp, sp, -64
   1828c: 0a 85        	mv	a0, sp
   1828e: 0c 08        	addi	a1, sp, 16
   18290: 3a e0        	sd	a4, 0(sp)
   18292: 3e e4        	sd	a5, 8(sp)
   18294: 06 fc        	sd	ra, 56(sp)
   18296: ef e0 ef c2  	jal	0x166c4 <e113toe.isra.0>
   1829a: 83 57 21 02  	lhu	a5, 34(sp)
   1829e: 01 45        	mv	a0, zero
   182a0: 93 c7 f7 ff  	not	a5, a5
   182a4: 13 97 17 03  	slli	a4, a5, 49
   182a8: 19 eb        	bnez	a4, 0x182be <_ldcheck+0x38>
   182aa: 1c 08        	addi	a5, sp, 16
   182ac: 93 06 21 02  	addi	a3, sp, 34
   182b0: 03 d7 07 00  	lhu	a4, 0(a5)
   182b4: 89 07        	addi	a5, a5, 2
   182b6: 19 e7        	bnez	a4, 0x182c4 <_ldcheck+0x3e>
   182b8: e3 9c d7 fe  	bne	a5, a3, 0x182b0 <_ldcheck+0x2a>
   182bc: 09 45        	addi	a0, zero, 2
   182be: e2 70        	ld	ra, 56(sp)
   182c0: 21 61        	addi	sp, sp, 64
   182c2: 82 80        	ret
   182c4: 05 45        	addi	a0, zero, 1
   182c6: e5 bf        	j	0x182be <_ldcheck+0x38>

00000000000182c8 <__localeconv_l>:
   182c8: 13 05 05 10  	addi	a0, a0, 256
   182cc: 82 80        	ret

00000000000182ce <_localeconv_r>:
   182ce: 37 45 02 00  	lui	a0, 36
   182d2: 13 05 85 86  	addi	a0, a0, -1944
   182d6: 82 80        	ret

00000000000182d8 <localeconv>:
   182d8: 37 45 02 00  	lui	a0, 36
   182dc: 13 05 85 86  	addi	a0, a0, -1944
   182e0: 82 80        	ret

00000000000182e2 <_setlocale_r>:
   182e2: 01 11        	addi	sp, sp, -32
   182e4: 06 ec        	sd	ra, 24(sp)
   182e6: 22 e8        	sd	s0, 16(sp)
   182e8: 26 e4        	sd	s1, 8(sp)
   182ea: 1d c2        	beqz	a2, 0x18310 <_setlocale_r+0x2e>
   182ec: b7 25 02 00  	lui	a1, 34
   182f0: 93 85 05 1a  	addi	a1, a1, 416
   182f4: 32 85        	mv	a0, a2
   182f6: 32 84        	mv	s0, a2
   182f8: ef 10 10 04  	jal	0x19b38 <strcmp>
   182fc: b7 24 02 00  	lui	s1, 34
   18300: 19 e9        	bnez	a0, 0x18316 <_setlocale_r+0x34>
   18302: 13 85 84 19  	addi	a0, s1, 408
   18306: e2 60        	ld	ra, 24(sp)
   18308: 42 64        	ld	s0, 16(sp)
   1830a: a2 64        	ld	s1, 8(sp)
   1830c: 05 61        	addi	sp, sp, 32
   1830e: 82 80        	ret
   18310: b7 24 02 00  	lui	s1, 34
   18314: fd b7        	j	0x18302 <_setlocale_r+0x20>
   18316: 93 85 84 19  	addi	a1, s1, 408
   1831a: 22 85        	mv	a0, s0
   1831c: ef 10 d0 01  	jal	0x19b38 <strcmp>
   18320: 6d d1        	beqz	a0, 0x18302 <_setlocale_r+0x20>
   18322: b7 25 02 00  	lui	a1, 34
   18326: 93 85 05 c0  	addi	a1, a1, -1024
   1832a: 22 85        	mv	a0, s0
   1832c: ef 10 d0 00  	jal	0x19b38 <strcmp>
   18330: 69 d9        	beqz	a0, 0x18302 <_setlocale_r+0x20>
   18332: 01 45        	mv	a0, zero
   18334: c9 bf        	j	0x18306 <_setlocale_r+0x24>

0000000000018336 <__locale_mb_cur_max>:
   18336: b7 47 02 00  	lui	a5, 36
   1833a: 03 c5 87 8c  	lbu	a0, -1848(a5)
   1833e: 82 80        	ret

0000000000018340 <setlocale>:
   18340: 01 11        	addi	sp, sp, -32
   18342: 06 ec        	sd	ra, 24(sp)
   18344: 22 e8        	sd	s0, 16(sp)
   18346: 26 e4        	sd	s1, 8(sp)
   18348: 9d c1        	beqz	a1, 0x1836e <setlocale+0x2e>
   1834a: 2e 84        	mv	s0, a1
   1834c: b7 25 02 00  	lui	a1, 34
   18350: 93 85 05 1a  	addi	a1, a1, 416
   18354: 22 85        	mv	a0, s0
   18356: ef 10 20 7e  	jal	0x19b38 <strcmp>
   1835a: b7 24 02 00  	lui	s1, 34
   1835e: 19 e9        	bnez	a0, 0x18374 <setlocale+0x34>
   18360: 13 85 84 19  	addi	a0, s1, 408
   18364: e2 60        	ld	ra, 24(sp)
   18366: 42 64        	ld	s0, 16(sp)
   18368: a2 64        	ld	s1, 8(sp)
   1836a: 05 61        	addi	sp, sp, 32
   1836c: 82 80        	ret
   1836e: b7 24 02 00  	lui	s1, 34
   18372: fd b7        	j	0x18360 <setlocale+0x20>
   18374: 93 85 84 19  	addi	a1, s1, 408
   18378: 22 85        	mv	a0, s0
   1837a: ef 10 e0 7b  	jal	0x19b38 <strcmp>
   1837e: 6d d1        	beqz	a0, 0x18360 <setlocale+0x20>
   18380: b7 25 02 00  	lui	a1, 34
   18384: 93 85 05 c0  	addi	a1, a1, -1024
   18388: 22 85        	mv	a0, s0
   1838a: ef 10 e0 7a  	jal	0x19b38 <strcmp>
   1838e: 69 d9        	beqz	a0, 0x18360 <setlocale+0x20>
   18390: 01 45        	mv	a0, zero
   18392: c9 bf        	j	0x18364 <setlocale+0x24>

0000000000018394 <__smakebuf_r>:
   18394: 83 97 05 01  	lh	a5, 16(a1)
   18398: 35 71        	addi	sp, sp, -160
   1839a: 22 e9        	sd	s0, 144(sp)
   1839c: 06 ed        	sd	ra, 152(sp)
   1839e: 26 e5        	sd	s1, 136(sp)
   183a0: 4a e1        	sd	s2, 128(sp)
   183a2: ce fc        	sd	s3, 120(sp)
   183a4: d2 f8        	sd	s4, 112(sp)
   183a6: 13 f7 27 00  	andi	a4, a5, 2
   183aa: 2e 84        	mv	s0, a1
   183ac: 19 cf        	beqz	a4, 0x183ca <__smakebuf_r+0x36>
   183ae: 93 87 75 07  	addi	a5, a1, 119
   183b2: 9c e1        	sd	a5, 0(a1)
   183b4: 9c ed        	sd	a5, 24(a1)
   183b6: 85 47        	addi	a5, zero, 1
   183b8: 9c d1        	sw	a5, 32(a1)
   183ba: ea 60        	ld	ra, 152(sp)
   183bc: 4a 64        	ld	s0, 144(sp)
   183be: aa 64        	ld	s1, 136(sp)
   183c0: 0a 69        	ld	s2, 128(sp)
   183c2: e6 79        	ld	s3, 120(sp)
   183c4: 46 7a        	ld	s4, 112(sp)
   183c6: 0d 61        	addi	sp, sp, 160
   183c8: 82 80        	ret
   183ca: 83 95 25 01  	lh	a1, 18(a1)
   183ce: aa 84        	mv	s1, a0
   183d0: 63 c8 05 06  	bltz	a1, 0x18440 <__smakebuf_r+0xac>
   183d4: 30 00        	addi	a2, sp, 8
   183d6: ef 40 40 78  	jal	0x1cb5a <_fstat_r>
   183da: 63 41 05 06  	bltz	a0, 0x1843c <__smakebuf_r+0xa8>
   183de: 32 49        	lw	s2, 12(sp)
   183e0: bd 67        	lui	a5, 15
   183e2: 85 69        	lui	s3, 1
   183e4: 33 79 f9 00  	and	s2, s2, a5
   183e8: 89 67        	lui	a5, 2
   183ea: 33 09 f9 40  	sub	s2, s2, a5
   183ee: 13 39 19 00  	seqz	s2, s2
   183f2: 13 0a 00 40  	addi	s4, zero, 1024
   183f6: 93 89 09 80  	addi	s3, s3, -2048
   183fa: d2 85        	mv	a1, s4
   183fc: 26 85        	mv	a0, s1
   183fe: ef 00 20 13  	jal	0x18530 <_malloc_r>
   18402: 83 17 04 01  	lh	a5, 16(s0)
   18406: 39 c9        	beqz	a0, 0x1845c <__smakebuf_r+0xc8>
   18408: 55 67        	lui	a4, 21
   1840a: 13 07 47 4e  	addi	a4, a4, 1252
   1840e: b8 ec        	sd	a4, 88(s1)
   18410: 93 e7 07 08  	ori	a5, a5, 128
   18414: 23 18 f4 00  	sh	a5, 16(s0)
   18418: 08 e0        	sd	a0, 0(s0)
   1841a: 08 ec        	sd	a0, 24(s0)
   1841c: 23 20 44 03  	sw	s4, 32(s0)
   18420: 63 11 09 06  	bnez	s2, 0x18482 <__smakebuf_r+0xee>
   18424: b3 e7 37 01  	or	a5, a5, s3
   18428: ea 60        	ld	ra, 152(sp)
   1842a: 23 18 f4 00  	sh	a5, 16(s0)
   1842e: 4a 64        	ld	s0, 144(sp)
   18430: aa 64        	ld	s1, 136(sp)
   18432: 0a 69        	ld	s2, 128(sp)
   18434: e6 79        	ld	s3, 120(sp)
   18436: 46 7a        	ld	s4, 112(sp)
   18438: 0d 61        	addi	sp, sp, 160
   1843a: 82 80        	ret
   1843c: 83 17 04 01  	lh	a5, 16(s0)
   18440: 93 f7 07 08  	andi	a5, a5, 128
   18444: 01 49        	mv	s2, zero
   18446: 95 cb        	beqz	a5, 0x1847a <__smakebuf_r+0xe6>
   18448: 13 0a 00 04  	addi	s4, zero, 64
   1844c: d2 85        	mv	a1, s4
   1844e: 26 85        	mv	a0, s1
   18450: ef 00 00 0e  	jal	0x18530 <_malloc_r>
   18454: 83 17 04 01  	lh	a5, 16(s0)
   18458: 81 49        	mv	s3, zero
   1845a: 5d f5        	bnez	a0, 0x18408 <__smakebuf_r+0x74>
   1845c: 13 f7 07 20  	andi	a4, a5, 512
   18460: 29 ff        	bnez	a4, 0x183ba <__smakebuf_r+0x26>
   18462: f1 9b        	andi	a5, a5, -4
   18464: 93 e7 27 00  	ori	a5, a5, 2
   18468: 13 07 74 07  	addi	a4, s0, 119
   1846c: 23 18 f4 00  	sh	a5, 16(s0)
   18470: 85 47        	addi	a5, zero, 1
   18472: 18 e0        	sd	a4, 0(s0)
   18474: 18 ec        	sd	a4, 24(s0)
   18476: 1c d0        	sw	a5, 32(s0)
   18478: 89 b7        	j	0x183ba <__smakebuf_r+0x26>
   1847a: 13 0a 00 40  	addi	s4, zero, 1024
   1847e: 81 49        	mv	s3, zero
   18480: ad bf        	j	0x183fa <__smakebuf_r+0x66>
   18482: 83 15 24 01  	lh	a1, 18(s0)
   18486: 26 85        	mv	a0, s1
   18488: ef 40 b0 23  	jal	0x1cec2 <_isatty_r>
   1848c: 01 e5        	bnez	a0, 0x18494 <__smakebuf_r+0x100>
   1848e: 83 17 04 01  	lh	a5, 16(s0)
   18492: 49 bf        	j	0x18424 <__smakebuf_r+0x90>
   18494: 83 57 04 01  	lhu	a5, 16(s0)
   18498: f1 9b        	andi	a5, a5, -4
   1849a: 93 e7 17 00  	ori	a5, a5, 1
   1849e: 9b 97 07 01  	slliw	a5, a5, 16
   184a2: 9b d7 07 41  	sraiw	a5, a5, 16
   184a6: bd bf        	j	0x18424 <__smakebuf_r+0x90>

00000000000184a8 <__swhatbuf_r>:
   184a8: 75 71        	addi	sp, sp, -144
   184aa: 22 e1        	sd	s0, 128(sp)
   184ac: 2e 84        	mv	s0, a1
   184ae: 83 95 25 01  	lh	a1, 18(a1)
   184b2: a6 fc        	sd	s1, 120(sp)
   184b4: ca f8        	sd	s2, 112(sp)
   184b6: 06 e5        	sd	ra, 136(sp)
   184b8: b2 84        	mv	s1, a2
   184ba: 36 89        	mv	s2, a3
   184bc: 63 cc 05 02  	bltz	a1, 0x184f4 <__swhatbuf_r+0x4c>
   184c0: 30 00        	addi	a2, sp, 8
   184c2: ef 40 80 69  	jal	0x1cb5a <_fstat_r>
   184c6: 63 47 05 02  	bltz	a0, 0x184f4 <__swhatbuf_r+0x4c>
   184ca: b2 47        	lw	a5, 12(sp)
   184cc: 3d 67        	lui	a4, 15
   184ce: aa 60        	ld	ra, 136(sp)
   184d0: f9 8f        	and	a5, a5, a4
   184d2: 09 67        	lui	a4, 2
   184d4: 99 8f        	sub	a5, a5, a4
   184d6: 0a 64        	ld	s0, 128(sp)
   184d8: 93 b7 17 00  	seqz	a5, a5
   184dc: 23 20 f9 00  	sw	a5, 0(s2)
   184e0: 13 07 00 40  	addi	a4, zero, 1024
   184e4: 98 e0        	sd	a4, 0(s1)
   184e6: 05 65        	lui	a0, 1
   184e8: e6 74        	ld	s1, 120(sp)
   184ea: 46 79        	ld	s2, 112(sp)
   184ec: 13 05 05 80  	addi	a0, a0, -2048
   184f0: 49 61        	addi	sp, sp, 144
   184f2: 82 80        	ret
   184f4: 83 57 04 01  	lhu	a5, 16(s0)
   184f8: 93 f7 07 08  	andi	a5, a5, 128
   184fc: 91 cf        	beqz	a5, 0x18518 <__swhatbuf_r+0x70>
   184fe: aa 60        	ld	ra, 136(sp)
   18500: 0a 64        	ld	s0, 128(sp)
   18502: 81 47        	mv	a5, zero
   18504: 23 20 f9 00  	sw	a5, 0(s2)
   18508: 13 07 00 04  	addi	a4, zero, 64
   1850c: 98 e0        	sd	a4, 0(s1)
   1850e: 46 79        	ld	s2, 112(sp)
   18510: e6 74        	ld	s1, 120(sp)
   18512: 01 45        	mv	a0, zero
   18514: 49 61        	addi	sp, sp, 144
   18516: 82 80        	ret
   18518: aa 60        	ld	ra, 136(sp)
   1851a: 0a 64        	ld	s0, 128(sp)
   1851c: 23 20 f9 00  	sw	a5, 0(s2)
   18520: 13 07 00 40  	addi	a4, zero, 1024
   18524: 98 e0        	sd	a4, 0(s1)
   18526: 46 79        	ld	s2, 112(sp)
   18528: e6 74        	ld	s1, 120(sp)
   1852a: 01 45        	mv	a0, zero
   1852c: 49 61        	addi	sp, sp, 144
   1852e: 82 80        	ret

0000000000018530 <_malloc_r>:
   18530: 1d 71        	addi	sp, sp, -96
   18532: a6 e4        	sd	s1, 72(sp)
   18534: ca e0        	sd	s2, 64(sp)
   18536: 86 ec        	sd	ra, 88(sp)
   18538: a2 e8        	sd	s0, 80(sp)
   1853a: 4e fc        	sd	s3, 56(sp)
   1853c: 52 f8        	sd	s4, 48(sp)
   1853e: 56 f4        	sd	s5, 40(sp)
   18540: 5a f0        	sd	s6, 32(sp)
   18542: 5e ec        	sd	s7, 24(sp)
   18544: 62 e8        	sd	s8, 16(sp)
   18546: 66 e4        	sd	s9, 8(sp)
   18548: 93 84 75 01  	addi	s1, a1, 23
   1854c: 93 07 e0 02  	addi	a5, zero, 46
   18550: 2a 89        	mv	s2, a0
   18552: 63 e4 97 06  	bltu	a5, s1, 0x185ba <_malloc_r+0x8a>
   18556: 93 07 00 02  	addi	a5, zero, 32
   1855a: 63 e1 b7 1a  	bltu	a5, a1, 0x186fc <_malloc_r+0x1cc>
   1855e: ef 00 40 6a  	jal	0x18c02 <__malloc_lock>
   18562: 93 04 00 02  	addi	s1, zero, 32
   18566: 93 07 00 05  	addi	a5, zero, 80
   1856a: 11 46        	addi	a2, zero, 4
   1856c: b7 49 02 00  	lui	s3, 36
   18570: 93 89 09 91  	addi	s3, s3, -1776
   18574: ce 97        	add	a5, a5, s3
   18576: 80 67        	ld	s0, 8(a5)
   18578: 13 87 07 ff  	addi	a4, a5, -16
   1857c: 63 0e e4 24  	beq	s0, a4, 0x187d8 <_malloc_r+0x2a8>
   18580: 1c 64        	ld	a5, 8(s0)
   18582: 14 6c        	ld	a3, 24(s0)
   18584: 10 68        	ld	a2, 16(s0)
   18586: f1 9b        	andi	a5, a5, -4
   18588: a2 97        	add	a5, a5, s0
   1858a: 98 67        	ld	a4, 8(a5)
   1858c: 14 ee        	sd	a3, 24(a2)
   1858e: 90 ea        	sd	a2, 16(a3)
   18590: 13 67 17 00  	ori	a4, a4, 1
   18594: 4a 85        	mv	a0, s2
   18596: 98 e7        	sd	a4, 8(a5)
   18598: ef 00 c0 66  	jal	0x18c04 <__malloc_unlock>
   1859c: 13 05 04 01  	addi	a0, s0, 16
   185a0: e6 60        	ld	ra, 88(sp)
   185a2: 46 64        	ld	s0, 80(sp)
   185a4: a6 64        	ld	s1, 72(sp)
   185a6: 06 69        	ld	s2, 64(sp)
   185a8: e2 79        	ld	s3, 56(sp)
   185aa: 42 7a        	ld	s4, 48(sp)
   185ac: a2 7a        	ld	s5, 40(sp)
   185ae: 02 7b        	ld	s6, 32(sp)
   185b0: e2 6b        	ld	s7, 24(sp)
   185b2: 42 6c        	ld	s8, 16(sp)
   185b4: a2 6c        	ld	s9, 8(sp)
   185b6: 25 61        	addi	sp, sp, 96
   185b8: 82 80        	ret
   185ba: b7 07 00 80  	lui	a5, 524288
   185be: c1 98        	andi	s1, s1, -16
   185c0: 93 c7 f7 ff  	not	a5, a5
   185c4: 63 ec 97 12  	bltu	a5, s1, 0x186fc <_malloc_r+0x1cc>
   185c8: 63 ea b4 12  	bltu	s1, a1, 0x186fc <_malloc_r+0x1cc>
   185cc: ef 00 60 63  	jal	0x18c02 <__malloc_lock>
   185d0: 93 07 70 1f  	addi	a5, zero, 503
   185d4: 63 fd 97 2e  	bgeu	a5, s1, 0x188ce <_malloc_r+0x39e>
   185d8: 93 d7 94 00  	srli	a5, s1, 9
   185dc: 63 85 07 12  	beqz	a5, 0x18706 <_malloc_r+0x1d6>
   185e0: 11 47        	addi	a4, zero, 4
   185e2: 63 62 f7 24  	bltu	a4, a5, 0x18826 <_malloc_r+0x2f6>
   185e6: 93 d7 64 00  	srli	a5, s1, 6
   185ea: 1b 86 97 03  	addiw	a2, a5, 57
   185ee: 9b 16 16 00  	slliw	a3, a2, 1
   185f2: 1b 85 87 03  	addiw	a0, a5, 56
   185f6: 8e 06        	slli	a3, a3, 3
   185f8: b7 49 02 00  	lui	s3, 36
   185fc: 93 89 09 91  	addi	s3, s3, -1776
   18600: ce 96        	add	a3, a3, s3
   18602: 80 66        	ld	s0, 8(a3)
   18604: c1 16        	addi	a3, a3, -16
   18606: 63 80 86 02  	beq	a3, s0, 0x18626 <_malloc_r+0xf6>
   1860a: fd 45        	addi	a1, zero, 31
   1860c: 31 a0        	j	0x18618 <_malloc_r+0xe8>
   1860e: 63 52 07 1c  	bgez	a4, 0x187d2 <_malloc_r+0x2a2>
   18612: 00 6c        	ld	s0, 24(s0)
   18614: 63 89 86 00  	beq	a3, s0, 0x18626 <_malloc_r+0xf6>
   18618: 1c 64        	ld	a5, 8(s0)
   1861a: f1 9b        	andi	a5, a5, -4
   1861c: 33 87 97 40  	sub	a4, a5, s1
   18620: e3 d7 e5 fe  	bge	a1, a4, 0x1860e <_malloc_r+0xde>
   18624: 2a 86        	mv	a2, a0
   18626: 03 b4 09 02  	ld	s0, 32(s3)
   1862a: 37 48 02 00  	lui	a6, 36
   1862e: 13 08 08 92  	addi	a6, a6, -1760
   18632: 63 0d 04 19  	beq	s0, a6, 0x187cc <_malloc_r+0x29c>
   18636: 1c 64        	ld	a5, 8(s0)
   18638: fd 46        	addi	a3, zero, 31
   1863a: f1 9b        	andi	a5, a5, -4
   1863c: 33 87 97 40  	sub	a4, a5, s1
   18640: 63 c0 e6 26  	blt	a3, a4, 0x188a0 <_malloc_r+0x370>
   18644: 23 b4 09 03  	sd	a6, 40(s3)
   18648: 23 b0 09 03  	sd	a6, 32(s3)
   1864c: 63 5f 07 22  	bgez	a4, 0x1888a <_malloc_r+0x35a>
   18650: 13 07 f0 1f  	addi	a4, zero, 511
   18654: 03 b5 89 00  	ld	a0, 8(s3)
   18658: 63 65 f7 18  	bltu	a4, a5, 0x187e2 <_malloc_r+0x2b2>
   1865c: 8d 83        	srli	a5, a5, 3
   1865e: 1b 87 17 00  	addiw	a4, a5, 1
   18662: 1b 17 17 00  	slliw	a4, a4, 1
   18666: 0e 07        	slli	a4, a4, 3
   18668: 4e 97        	add	a4, a4, s3
   1866a: 0c 63        	ld	a1, 0(a4)
   1866c: 9b d7 27 40  	sraiw	a5, a5, 2
   18670: 85 46        	addi	a3, zero, 1
   18672: b3 97 f6 00  	sll	a5, a3, a5
   18676: 5d 8d        	or	a0, a0, a5
   18678: 93 07 07 ff  	addi	a5, a4, -16
   1867c: 1c ec        	sd	a5, 24(s0)
   1867e: 0c e8        	sd	a1, 16(s0)
   18680: 23 b4 a9 00  	sd	a0, 8(s3)
   18684: 00 e3        	sd	s0, 0(a4)
   18686: 80 ed        	sd	s0, 24(a1)
   18688: 9b 57 26 40  	sraiw	a5, a2, 2
   1868c: 85 45        	addi	a1, zero, 1
   1868e: b3 95 f5 00  	sll	a1, a1, a5
   18692: 63 61 b5 08  	bltu	a0, a1, 0x18714 <_malloc_r+0x1e4>
   18696: b3 f7 a5 00  	and	a5, a1, a0
   1869a: 81 ef        	bnez	a5, 0x186b2 <_malloc_r+0x182>
   1869c: 86 05        	slli	a1, a1, 1
   1869e: 71 9a        	andi	a2, a2, -4
   186a0: b3 f7 a5 00  	and	a5, a1, a0
   186a4: 11 26        	addiw	a2, a2, 4
   186a6: 91 e7        	bnez	a5, 0x186b2 <_malloc_r+0x182>
   186a8: 86 05        	slli	a1, a1, 1
   186aa: b3 f7 a5 00  	and	a5, a1, a0
   186ae: 11 26        	addiw	a2, a2, 4
   186b0: e5 df        	beqz	a5, 0x186a8 <_malloc_r+0x178>
   186b2: fd 48        	addi	a7, zero, 31
   186b4: 1b 03 16 00  	addiw	t1, a2, 1
   186b8: 1b 13 13 00  	slliw	t1, t1, 1
   186bc: 0e 03        	slli	t1, t1, 3
   186be: 41 13        	addi	t1, t1, -16
   186c0: 4e 93        	add	t1, t1, s3
   186c2: 1a 85        	mv	a0, t1
   186c4: 1c 6d        	ld	a5, 24(a0)
   186c6: 32 8e        	mv	t3, a2
   186c8: 63 00 f5 18  	beq	a0, a5, 0x18848 <_malloc_r+0x318>
   186cc: 98 67        	ld	a4, 8(a5)
   186ce: 3e 84        	mv	s0, a5
   186d0: 9c 6f        	ld	a5, 24(a5)
   186d2: 71 9b        	andi	a4, a4, -4
   186d4: b3 06 97 40  	sub	a3, a4, s1
   186d8: 63 cf d8 16  	blt	a7, a3, 0x18856 <_malloc_r+0x326>
   186dc: e3 c6 06 fe  	bltz	a3, 0x186c8 <_malloc_r+0x198>
   186e0: 22 97        	add	a4, a4, s0
   186e2: 14 67        	ld	a3, 8(a4)
   186e4: 10 68        	ld	a2, 16(s0)
   186e6: 4a 85        	mv	a0, s2
   186e8: 93 e6 16 00  	ori	a3, a3, 1
   186ec: 14 e7        	sd	a3, 8(a4)
   186ee: 1c ee        	sd	a5, 24(a2)
   186f0: 90 eb        	sd	a2, 16(a5)
   186f2: ef 00 20 51  	jal	0x18c04 <__malloc_unlock>
   186f6: 13 05 04 01  	addi	a0, s0, 16
   186fa: 5d b5        	j	0x185a0 <_malloc_r+0x70>
   186fc: b1 47        	addi	a5, zero, 12
   186fe: 23 20 f9 00  	sw	a5, 0(s2)
   18702: 01 45        	mv	a0, zero
   18704: 71 bd        	j	0x185a0 <_malloc_r+0x70>
   18706: 93 06 00 40  	addi	a3, zero, 1024
   1870a: 13 06 00 04  	addi	a2, zero, 64
   1870e: 13 05 f0 03  	addi	a0, zero, 63
   18712: dd b5        	j	0x185f8 <_malloc_r+0xc8>
   18714: 03 b4 09 01  	ld	s0, 16(s3)
   18718: 1c 64        	ld	a5, 8(s0)
   1871a: 93 fa c7 ff  	andi	s5, a5, -4
   1871e: 63 e7 9a 00  	bltu	s5, s1, 0x1872c <_malloc_r+0x1fc>
   18722: 33 87 9a 40  	sub	a4, s5, s1
   18726: fd 47        	addi	a5, zero, 31
   18728: 63 c3 e7 08  	blt	a5, a4, 0x187ae <_malloc_r+0x27e>
   1872c: 03 ba 01 97  	ld	s4, -1680(gp)
   18730: 03 b7 01 94  	ld	a4, -1728(gp)
   18734: fd 57        	addi	a5, zero, -1
   18736: 33 0c 54 01  	add	s8, s0, s5
   1873a: 26 9a        	add	s4, s4, s1
   1873c: 63 05 f7 2e  	beq	a4, a5, 0x18a26 <_malloc_r+0x4f6>
   18740: 85 67        	lui	a5, 1
   18742: fd 07        	addi	a5, a5, 31
   18744: 3e 9a        	add	s4, s4, a5
   18746: fd 77        	lui	a5, 1048575
   18748: 33 7a fa 00  	and	s4, s4, a5
   1874c: d2 85        	mv	a1, s4
   1874e: 4a 85        	mv	a0, s2
   18750: ef 10 20 1c  	jal	0x19912 <_sbrk_r>
   18754: fd 57        	addi	a5, zero, -1
   18756: 2a 8b        	mv	s6, a0
   18758: 63 00 f5 1a  	beq	a0, a5, 0x188f8 <_malloc_r+0x3c8>
   1875c: 63 6c 85 19  	bltu	a0, s8, 0x188f4 <_malloc_r+0x3c4>
   18760: b7 6b 02 00  	lui	s7, 38
   18764: 93 8b 0b 54  	addi	s7, s7, 1344
   18768: 83 a7 0b 00  	lw	a5, 0(s7)
   1876c: 3b 87 47 01  	addw	a4, a5, s4
   18770: 23 a0 eb 00  	sw	a4, 0(s7)
   18774: ba 86        	mv	a3, a4
   18776: 63 12 ac 20  	bne	s8, a0, 0x1897a <_malloc_r+0x44a>
   1877a: 93 17 45 03  	slli	a5, a0, 52
   1877e: 63 9e 07 1e  	bnez	a5, 0x1897a <_malloc_r+0x44a>
   18782: 03 bb 09 01  	ld	s6, 16(s3)
   18786: b3 87 4a 01  	add	a5, s5, s4
   1878a: 93 e7 17 00  	ori	a5, a5, 1
   1878e: 23 34 fb 00  	sd	a5, 8(s6)
   18792: 03 b6 81 96  	ld	a2, -1688(gp)
   18796: 63 74 e6 00  	bgeu	a2, a4, 0x1879e <_malloc_r+0x26e>
   1879a: 23 b4 e1 96  	sd	a4, -1688(gp)
   1879e: 03 b6 01 96  	ld	a2, -1696(gp)
   187a2: 63 74 e6 00  	bgeu	a2, a4, 0x187aa <_malloc_r+0x27a>
   187a6: 23 b0 e1 96  	sd	a4, -1696(gp)
   187aa: 5a 84        	mv	s0, s6
   187ac: 89 aa        	j	0x188fe <_malloc_r+0x3ce>
   187ae: 93 e7 14 00  	ori	a5, s1, 1
   187b2: 1c e4        	sd	a5, 8(s0)
   187b4: a2 94        	add	s1, s1, s0
   187b6: 23 b8 99 00  	sd	s1, 16(s3)
   187ba: 13 67 17 00  	ori	a4, a4, 1
   187be: 4a 85        	mv	a0, s2
   187c0: 98 e4        	sd	a4, 8(s1)
   187c2: ef 00 20 44  	jal	0x18c04 <__malloc_unlock>
   187c6: 13 05 04 01  	addi	a0, s0, 16
   187ca: d9 bb        	j	0x185a0 <_malloc_r+0x70>
   187cc: 03 b5 89 00  	ld	a0, 8(s3)
   187d0: 65 bd        	j	0x18688 <_malloc_r+0x158>
   187d2: 14 6c        	ld	a3, 24(s0)
   187d4: 10 68        	ld	a2, 16(s0)
   187d6: 4d bb        	j	0x18588 <_malloc_r+0x58>
   187d8: 80 6f        	ld	s0, 24(a5)
   187da: 09 26        	addiw	a2, a2, 2
   187dc: e3 85 87 e4  	beq	a5, s0, 0x18626 <_malloc_r+0xf6>
   187e0: 45 b3        	j	0x18580 <_malloc_r+0x50>
   187e2: 13 d7 97 00  	srli	a4, a5, 9
   187e6: 91 46        	addi	a3, zero, 4
   187e8: 63 fc e6 0e  	bgeu	a3, a4, 0x188e0 <_malloc_r+0x3b0>
   187ec: d1 46        	addi	a3, zero, 20
   187ee: 63 e8 e6 26  	bltu	a3, a4, 0x18a5e <_malloc_r+0x52e>
   187f2: 9b 05 c7 05  	addiw	a1, a4, 92
   187f6: 9b 95 15 00  	slliw	a1, a1, 1
   187fa: 9b 06 b7 05  	addiw	a3, a4, 91
   187fe: 8e 05        	slli	a1, a1, 3
   18800: ce 95        	add	a1, a1, s3
   18802: 98 61        	ld	a4, 0(a1)
   18804: c1 15        	addi	a1, a1, -16
   18806: 63 96 e5 00  	bne	a1, a4, 0x18812 <_malloc_r+0x2e2>
   1880a: 0d a4        	j	0x18a2c <_malloc_r+0x4fc>
   1880c: 18 6b        	ld	a4, 16(a4)
   1880e: 63 86 e5 00  	beq	a1, a4, 0x1881a <_malloc_r+0x2ea>
   18812: 14 67        	ld	a3, 8(a4)
   18814: f1 9a        	andi	a3, a3, -4
   18816: e3 eb d7 fe  	bltu	a5, a3, 0x1880c <_malloc_r+0x2dc>
   1881a: 0c 6f        	ld	a1, 24(a4)
   1881c: 0c ec        	sd	a1, 24(s0)
   1881e: 18 e8        	sd	a4, 16(s0)
   18820: 80 e9        	sd	s0, 16(a1)
   18822: 00 ef        	sd	s0, 24(a4)
   18824: 95 b5        	j	0x18688 <_malloc_r+0x158>
   18826: 51 47        	addi	a4, zero, 20
   18828: 63 78 f7 0e  	bgeu	a4, a5, 0x18918 <_malloc_r+0x3e8>
   1882c: 13 07 40 05  	addi	a4, zero, 84
   18830: 63 65 f7 24  	bltu	a4, a5, 0x18a7a <_malloc_r+0x54a>
   18834: 93 d7 c4 00  	srli	a5, s1, 12
   18838: 1b 86 f7 06  	addiw	a2, a5, 111
   1883c: 9b 16 16 00  	slliw	a3, a2, 1
   18840: 1b 85 e7 06  	addiw	a0, a5, 110
   18844: 8e 06        	slli	a3, a3, 3
   18846: 4d bb        	j	0x185f8 <_malloc_r+0xc8>
   18848: 05 2e        	addiw	t3, t3, 1
   1884a: 93 77 3e 00  	andi	a5, t3, 3
   1884e: 41 05        	addi	a0, a0, 16
   18850: ed c3        	beqz	a5, 0x18932 <_malloc_r+0x402>
   18852: 1c 6d        	ld	a5, 24(a0)
   18854: 95 bd        	j	0x186c8 <_malloc_r+0x198>
   18856: 10 68        	ld	a2, 16(s0)
   18858: 93 e5 14 00  	ori	a1, s1, 1
   1885c: 0c e4        	sd	a1, 8(s0)
   1885e: 1c ee        	sd	a5, 24(a2)
   18860: 90 eb        	sd	a2, 16(a5)
   18862: a2 94        	add	s1, s1, s0
   18864: 23 b4 99 02  	sd	s1, 40(s3)
   18868: 23 b0 99 02  	sd	s1, 32(s3)
   1886c: 93 e7 16 00  	ori	a5, a3, 1
   18870: 23 bc 04 01  	sd	a6, 24(s1)
   18874: 23 b8 04 01  	sd	a6, 16(s1)
   18878: 9c e4        	sd	a5, 8(s1)
   1887a: 22 97        	add	a4, a4, s0
   1887c: 4a 85        	mv	a0, s2
   1887e: 14 e3        	sd	a3, 0(a4)
   18880: ef 00 40 38  	jal	0x18c04 <__malloc_unlock>
   18884: 13 05 04 01  	addi	a0, s0, 16
   18888: 21 bb        	j	0x185a0 <_malloc_r+0x70>
   1888a: a2 97        	add	a5, a5, s0
   1888c: 98 67        	ld	a4, 8(a5)
   1888e: 4a 85        	mv	a0, s2
   18890: 13 67 17 00  	ori	a4, a4, 1
   18894: 98 e7        	sd	a4, 8(a5)
   18896: ef 00 e0 36  	jal	0x18c04 <__malloc_unlock>
   1889a: 13 05 04 01  	addi	a0, s0, 16
   1889e: 09 b3        	j	0x185a0 <_malloc_r+0x70>
   188a0: 93 e6 14 00  	ori	a3, s1, 1
   188a4: 14 e4        	sd	a3, 8(s0)
   188a6: a2 94        	add	s1, s1, s0
   188a8: 23 b4 99 02  	sd	s1, 40(s3)
   188ac: 23 b0 99 02  	sd	s1, 32(s3)
   188b0: 93 66 17 00  	ori	a3, a4, 1
   188b4: 23 bc 04 01  	sd	a6, 24(s1)
   188b8: 23 b8 04 01  	sd	a6, 16(s1)
   188bc: 94 e4        	sd	a3, 8(s1)
   188be: a2 97        	add	a5, a5, s0
   188c0: 4a 85        	mv	a0, s2
   188c2: 98 e3        	sd	a4, 0(a5)
   188c4: ef 00 00 34  	jal	0x18c04 <__malloc_unlock>
   188c8: 13 05 04 01  	addi	a0, s0, 16
   188cc: d1 b9        	j	0x185a0 <_malloc_r+0x70>
   188ce: 13 d6 34 00  	srli	a2, s1, 3
   188d2: 9b 07 16 00  	addiw	a5, a2, 1
   188d6: 9b 97 17 00  	slliw	a5, a5, 1
   188da: 01 26        	sext.w	a2, a2
   188dc: 8e 07        	slli	a5, a5, 3
   188de: 79 b1        	j	0x1856c <_malloc_r+0x3c>
   188e0: 13 d7 67 00  	srli	a4, a5, 6
   188e4: 9b 05 97 03  	addiw	a1, a4, 57
   188e8: 9b 95 15 00  	slliw	a1, a1, 1
   188ec: 9b 06 87 03  	addiw	a3, a4, 56
   188f0: 8e 05        	slli	a1, a1, 3
   188f2: 39 b7        	j	0x18800 <_malloc_r+0x2d0>
   188f4: 63 09 34 07  	beq	s0, s3, 0x18966 <_malloc_r+0x436>
   188f8: 03 b4 09 01  	ld	s0, 16(s3)
   188fc: 1c 64        	ld	a5, 8(s0)
   188fe: f1 9b        	andi	a5, a5, -4
   18900: 33 87 97 40  	sub	a4, a5, s1
   18904: 63 e5 97 00  	bltu	a5, s1, 0x1890e <_malloc_r+0x3de>
   18908: fd 47        	addi	a5, zero, 31
   1890a: e3 c2 e7 ea  	blt	a5, a4, 0x187ae <_malloc_r+0x27e>
   1890e: 4a 85        	mv	a0, s2
   18910: ef 00 40 2f  	jal	0x18c04 <__malloc_unlock>
   18914: 01 45        	mv	a0, zero
   18916: 69 b1        	j	0x185a0 <_malloc_r+0x70>
   18918: 1b 86 c7 05  	addiw	a2, a5, 92
   1891c: 9b 16 16 00  	slliw	a3, a2, 1
   18920: 1b 85 b7 05  	addiw	a0, a5, 91
   18924: 8e 06        	slli	a3, a3, 3
   18926: c9 b9        	j	0x185f8 <_malloc_r+0xc8>
   18928: 83 37 03 01  	ld	a5, 16(t1)
   1892c: 7d 36        	addiw	a2, a2, -1
   1892e: 63 97 67 1e  	bne	a5, t1, 0x18b1c <_malloc_r+0x5ec>
   18932: 93 77 36 00  	andi	a5, a2, 3
   18936: 41 13        	addi	t1, t1, -16
   18938: e5 fb        	bnez	a5, 0x18928 <_malloc_r+0x3f8>
   1893a: 03 b7 89 00  	ld	a4, 8(s3)
   1893e: 93 c7 f5 ff  	not	a5, a1
   18942: f9 8f        	and	a5, a5, a4
   18944: 23 b4 f9 00  	sd	a5, 8(s3)
   18948: 86 05        	slli	a1, a1, 1
   1894a: e3 e5 b7 dc  	bltu	a5, a1, 0x18714 <_malloc_r+0x1e4>
   1894e: e3 83 05 dc  	beqz	a1, 0x18714 <_malloc_r+0x1e4>
   18952: 33 f7 f5 00  	and	a4, a1, a5
   18956: 11 e7        	bnez	a4, 0x18962 <_malloc_r+0x432>
   18958: 86 05        	slli	a1, a1, 1
   1895a: 33 f7 f5 00  	and	a4, a1, a5
   1895e: 11 2e        	addiw	t3, t3, 4
   18960: 65 df        	beqz	a4, 0x18958 <_malloc_r+0x428>
   18962: 72 86        	mv	a2, t3
   18964: 81 bb        	j	0x186b4 <_malloc_r+0x184>
   18966: b7 6b 02 00  	lui	s7, 38
   1896a: 93 8b 0b 54  	addi	s7, s7, 1344
   1896e: 83 a7 0b 00  	lw	a5, 0(s7)
   18972: bb 86 47 01  	addw	a3, a5, s4
   18976: 23 a0 db 00  	sw	a3, 0(s7)
   1897a: 03 b7 01 94  	ld	a4, -1728(gp)
   1897e: fd 57        	addi	a5, zero, -1
   18980: 63 0b f7 10  	beq	a4, a5, 0x18a96 <_malloc_r+0x566>
   18984: b3 07 8b 41  	sub	a5, s6, s8
   18988: b5 9f        	addw	a5, a5, a3
   1898a: 23 a0 fb 00  	sw	a5, 0(s7)
   1898e: 93 7c fb 00  	andi	s9, s6, 15
   18992: 63 86 0c 0a  	beqz	s9, 0x18a3e <_malloc_r+0x50e>
   18996: 05 67        	lui	a4, 1
   18998: 33 0b 9b 41  	sub	s6, s6, s9
   1899c: 13 0c 07 01  	addi	s8, a4, 16
   189a0: 41 0b        	addi	s6, s6, 16
   189a2: 33 0c 9c 41  	sub	s8, s8, s9
   189a6: 5a 9a        	add	s4, s4, s6
   189a8: 7d 17        	addi	a4, a4, -1
   189aa: 33 0c 4c 41  	sub	s8, s8, s4
   189ae: 33 7c ec 00  	and	s8, s8, a4
   189b2: e2 85        	mv	a1, s8
   189b4: 4a 85        	mv	a0, s2
   189b6: ef 00 d0 75  	jal	0x19912 <_sbrk_r>
   189ba: fd 57        	addi	a5, zero, -1
   189bc: 63 00 f5 12  	beq	a0, a5, 0x18adc <_malloc_r+0x5ac>
   189c0: 33 05 65 41  	sub	a0, a0, s6
   189c4: 9b 06 0c 00  	sext.w	a3, s8
   189c8: 33 0a 85 01  	add	s4, a0, s8
   189cc: 03 a7 0b 00  	lw	a4, 0(s7)
   189d0: 23 b8 69 01  	sd	s6, 16(s3)
   189d4: 93 67 1a 00  	ori	a5, s4, 1
   189d8: 35 9f        	addw	a4, a4, a3
   189da: 23 a0 eb 00  	sw	a4, 0(s7)
   189de: 23 34 fb 00  	sd	a5, 8(s6)
   189e2: e3 08 34 db  	beq	s0, s3, 0x18792 <_malloc_r+0x262>
   189e6: 7d 46        	addi	a2, zero, 31
   189e8: 63 7a 56 0b  	bgeu	a2, s5, 0x18a9c <_malloc_r+0x56c>
   189ec: 14 64        	ld	a3, 8(s0)
   189ee: 93 87 8a fe  	addi	a5, s5, -24
   189f2: c1 9b        	andi	a5, a5, -16
   189f4: 85 8a        	andi	a3, a3, 1
   189f6: dd 8e        	or	a3, a3, a5
   189f8: 14 e4        	sd	a3, 8(s0)
   189fa: a5 45        	addi	a1, zero, 9
   189fc: b3 06 f4 00  	add	a3, s0, a5
   18a00: 8c e6        	sd	a1, 8(a3)
   18a02: 8c ea        	sd	a1, 16(a3)
   18a04: 63 65 f6 00  	bltu	a2, a5, 0x18a0e <_malloc_r+0x4de>
   18a08: 83 37 8b 00  	ld	a5, 8(s6)
   18a0c: 59 b3        	j	0x18792 <_malloc_r+0x262>
   18a0e: 93 05 04 01  	addi	a1, s0, 16
   18a12: 4a 85        	mv	a0, s2
   18a14: ef c0 3f e5  	jal	0x15866 <_free_r>
   18a18: 03 bb 09 01  	ld	s6, 16(s3)
   18a1c: 03 a7 0b 00  	lw	a4, 0(s7)
   18a20: 83 37 8b 00  	ld	a5, 8(s6)
   18a24: bd b3        	j	0x18792 <_malloc_r+0x262>
   18a26: 13 0a 0a 02  	addi	s4, s4, 32
   18a2a: 0d b3        	j	0x1874c <_malloc_r+0x21c>
   18a2c: 9b d6 26 40  	sraiw	a3, a3, 2
   18a30: 85 47        	addi	a5, zero, 1
   18a32: b3 97 d7 00  	sll	a5, a5, a3
   18a36: 5d 8d        	or	a0, a0, a5
   18a38: 23 b4 a9 00  	sd	a0, 8(s3)
   18a3c: c5 b3        	j	0x1881c <_malloc_r+0x2ec>
   18a3e: 33 0c 4b 01  	add	s8, s6, s4
   18a42: 33 0c 80 41  	neg	s8, s8
   18a46: 52 1c        	slli	s8, s8, 52
   18a48: 13 5c 4c 03  	srli	s8, s8, 52
   18a4c: e2 85        	mv	a1, s8
   18a4e: 4a 85        	mv	a0, s2
   18a50: ef 00 30 6c  	jal	0x19912 <_sbrk_r>
   18a54: fd 57        	addi	a5, zero, -1
   18a56: 81 46        	mv	a3, zero
   18a58: e3 14 f5 f6  	bne	a0, a5, 0x189c0 <_malloc_r+0x490>
   18a5c: 85 bf        	j	0x189cc <_malloc_r+0x49c>
   18a5e: 93 06 40 05  	addi	a3, zero, 84
   18a62: 63 e1 e6 04  	bltu	a3, a4, 0x18aa4 <_malloc_r+0x574>
   18a66: 13 d7 c7 00  	srli	a4, a5, 12
   18a6a: 9b 05 f7 06  	addiw	a1, a4, 111
   18a6e: 9b 95 15 00  	slliw	a1, a1, 1
   18a72: 9b 06 e7 06  	addiw	a3, a4, 110
   18a76: 8e 05        	slli	a1, a1, 3
   18a78: 61 b3        	j	0x18800 <_malloc_r+0x2d0>
   18a7a: 13 07 40 15  	addi	a4, zero, 340
   18a7e: 63 61 f7 04  	bltu	a4, a5, 0x18ac0 <_malloc_r+0x590>
   18a82: 93 d7 f4 00  	srli	a5, s1, 15
   18a86: 1b 86 87 07  	addiw	a2, a5, 120
   18a8a: 9b 16 16 00  	slliw	a3, a2, 1
   18a8e: 1b 85 77 07  	addiw	a0, a5, 119
   18a92: 8e 06        	slli	a3, a3, 3
   18a94: 95 b6        	j	0x185f8 <_malloc_r+0xc8>
   18a96: 23 b0 61 95  	sd	s6, -1728(gp)
   18a9a: d5 bd        	j	0x1898e <_malloc_r+0x45e>
   18a9c: 85 47        	addi	a5, zero, 1
   18a9e: 23 34 fb 00  	sd	a5, 8(s6)
   18aa2: b5 b5        	j	0x1890e <_malloc_r+0x3de>
   18aa4: 93 06 40 15  	addi	a3, zero, 340
   18aa8: 63 e0 e6 04  	bltu	a3, a4, 0x18ae8 <_malloc_r+0x5b8>
   18aac: 13 d7 f7 00  	srli	a4, a5, 15
   18ab0: 9b 05 87 07  	addiw	a1, a4, 120
   18ab4: 9b 95 15 00  	slliw	a1, a1, 1
   18ab8: 9b 06 77 07  	addiw	a3, a4, 119
   18abc: 8e 05        	slli	a1, a1, 3
   18abe: 89 b3        	j	0x18800 <_malloc_r+0x2d0>
   18ac0: 13 07 40 55  	addi	a4, zero, 1364
   18ac4: 63 60 f7 04  	bltu	a4, a5, 0x18b04 <_malloc_r+0x5d4>
   18ac8: 93 d7 24 01  	srli	a5, s1, 18
   18acc: 1b 86 d7 07  	addiw	a2, a5, 125
   18ad0: 9b 16 16 00  	slliw	a3, a2, 1
   18ad4: 1b 85 c7 07  	addiw	a0, a5, 124
   18ad8: 8e 06        	slli	a3, a3, 3
   18ada: 39 be        	j	0x185f8 <_malloc_r+0xc8>
   18adc: c1 1c        	addi	s9, s9, -16
   18ade: 66 9a        	add	s4, s4, s9
   18ae0: 33 0a 6a 41  	sub	s4, s4, s6
   18ae4: 81 46        	mv	a3, zero
   18ae6: dd b5        	j	0x189cc <_malloc_r+0x49c>
   18ae8: 93 06 40 55  	addi	a3, zero, 1364
   18aec: 63 e3 e6 02  	bltu	a3, a4, 0x18b12 <_malloc_r+0x5e2>
   18af0: 13 d7 27 01  	srli	a4, a5, 18
   18af4: 9b 05 d7 07  	addiw	a1, a4, 125
   18af8: 9b 95 15 00  	slliw	a1, a1, 1
   18afc: 9b 06 c7 07  	addiw	a3, a4, 124
   18b00: 8e 05        	slli	a1, a1, 3
   18b02: fd b9        	j	0x18800 <_malloc_r+0x2d0>
   18b04: 93 06 00 7f  	addi	a3, zero, 2032
   18b08: 13 06 f0 07  	addi	a2, zero, 127
   18b0c: 13 05 e0 07  	addi	a0, zero, 126
   18b10: e5 b4        	j	0x185f8 <_malloc_r+0xc8>
   18b12: 93 05 00 7f  	addi	a1, zero, 2032
   18b16: 93 06 e0 07  	addi	a3, zero, 126
   18b1a: dd b1        	j	0x18800 <_malloc_r+0x2d0>
   18b1c: 83 b7 89 00  	ld	a5, 8(s3)
   18b20: 25 b5        	j	0x18948 <_malloc_r+0x418>

0000000000018b22 <_mbtowc_r>:
   18b22: b7 47 02 00  	lui	a5, 36
   18b26: 83 b7 07 85  	ld	a5, -1968(a5)
   18b2a: 82 87        	jr	a5

0000000000018b2c <__ascii_mbtowc>:
   18b2c: 99 c9        	beqz	a1, 0x18b42 <__ascii_mbtowc+0x16>
   18b2e: 1d c6        	beqz	a2, 0x18b5c <__ascii_mbtowc+0x30>
   18b30: 95 ca        	beqz	a3, 0x18b64 <__ascii_mbtowc+0x38>
   18b32: 83 47 06 00  	lbu	a5, 0(a2)
   18b36: 9c c1        	sw	a5, 0(a1)
   18b38: 03 45 06 00  	lbu	a0, 0(a2)
   18b3c: 33 35 a0 00  	snez	a0, a0
   18b40: 82 80        	ret
   18b42: 41 11        	addi	sp, sp, -16
   18b44: 6c 00        	addi	a1, sp, 12
   18b46: 09 ce        	beqz	a2, 0x18b60 <__ascii_mbtowc+0x34>
   18b48: 85 c2        	beqz	a3, 0x18b68 <__ascii_mbtowc+0x3c>
   18b4a: 83 47 06 00  	lbu	a5, 0(a2)
   18b4e: 9c c1        	sw	a5, 0(a1)
   18b50: 03 45 06 00  	lbu	a0, 0(a2)
   18b54: 33 35 a0 00  	snez	a0, a0
   18b58: 41 01        	addi	sp, sp, 16
   18b5a: 82 80        	ret
   18b5c: 01 45        	mv	a0, zero
   18b5e: 82 80        	ret
   18b60: 01 45        	mv	a0, zero
   18b62: dd bf        	j	0x18b58 <__ascii_mbtowc+0x2c>
   18b64: 79 55        	addi	a0, zero, -2
   18b66: 82 80        	ret
   18b68: 79 55        	addi	a0, zero, -2
   18b6a: fd b7        	j	0x18b58 <__ascii_mbtowc+0x2c>

0000000000018b6c <memchr>:
   18b6c: 93 77 75 00  	andi	a5, a0, 7
   18b70: 93 f6 f5 0f  	andi	a3, a1, 255
   18b74: 85 c7        	beqz	a5, 0x18b9c <memchr+0x30>
   18b76: 93 07 f6 ff  	addi	a5, a2, -1
   18b7a: 19 ce        	beqz	a2, 0x18b98 <memchr+0x2c>
   18b7c: 7d 56        	addi	a2, zero, -1
   18b7e: 01 a8        	j	0x18b8e <memchr+0x22>
   18b80: 05 05        	addi	a0, a0, 1
   18b82: 13 77 75 00  	andi	a4, a0, 7
   18b86: 01 cf        	beqz	a4, 0x18b9e <memchr+0x32>
   18b88: fd 17        	addi	a5, a5, -1
   18b8a: 63 87 c7 00  	beq	a5, a2, 0x18b98 <memchr+0x2c>
   18b8e: 03 47 05 00  	lbu	a4, 0(a0)
   18b92: e3 17 d7 fe  	bne	a4, a3, 0x18b80 <memchr+0x14>
   18b96: 82 80        	ret
   18b98: 01 45        	mv	a0, zero
   18b9a: 82 80        	ret
   18b9c: b2 87        	mv	a5, a2
   18b9e: 1d 47        	addi	a4, zero, 7
   18ba0: 63 75 f7 04  	bgeu	a4, a5, 0x18bea <memchr+0x7e>
   18ba4: 93 f5 f5 0f  	andi	a1, a1, 255
   18ba8: 13 97 85 00  	slli	a4, a1, 8
   18bac: 2e 97        	add	a4, a4, a1
   18bae: 93 15 07 01  	slli	a1, a4, 16
   18bb2: ba 95        	add	a1, a1, a4
   18bb4: 37 47 02 00  	lui	a4, 36
   18bb8: 83 38 07 12  	ld	a7, 288(a4)
   18bbc: 37 47 02 00  	lui	a4, 36
   18bc0: 03 38 87 12  	ld	a6, 296(a4)
   18bc4: 13 97 05 02  	slli	a4, a1, 32
   18bc8: ba 95        	add	a1, a1, a4
   18bca: 1d 43        	addi	t1, zero, 7
   18bcc: 29 a0        	j	0x18bd6 <memchr+0x6a>
   18bce: e1 17        	addi	a5, a5, -8
   18bd0: 21 05        	addi	a0, a0, 8
   18bd2: 63 7c f3 00  	bgeu	t1, a5, 0x18bea <memchr+0x7e>
   18bd6: 18 61        	ld	a4, 0(a0)
   18bd8: 2d 8f        	xor	a4, a4, a1
   18bda: 33 06 17 01  	add	a2, a4, a7
   18bde: 13 47 f7 ff  	not	a4, a4
   18be2: 71 8f        	and	a4, a4, a2
   18be4: 33 77 07 01  	and	a4, a4, a6
   18be8: 7d d3        	beqz	a4, 0x18bce <memchr+0x62>
   18bea: 33 07 f5 00  	add	a4, a0, a5
   18bee: 89 e7        	bnez	a5, 0x18bf8 <memchr+0x8c>
   18bf0: 65 b7        	j	0x18b98 <memchr+0x2c>
   18bf2: 05 05        	addi	a0, a0, 1
   18bf4: e3 02 a7 fa  	beq	a4, a0, 0x18b98 <memchr+0x2c>
   18bf8: 83 47 05 00  	lbu	a5, 0(a0)
   18bfc: e3 9b d7 fe  	bne	a5, a3, 0x18bf2 <memchr+0x86>
   18c00: 82 80        	ret

0000000000018c02 <__malloc_lock>:
   18c02: 82 80        	ret

0000000000018c04 <__malloc_unlock>:
   18c04: 82 80        	ret

0000000000018c06 <_Balloc>:
   18c06: 3c 7d        	ld	a5, 120(a0)
   18c08: 01 11        	addi	sp, sp, -32
   18c0a: 22 e8        	sd	s0, 16(sp)
   18c0c: 26 e4        	sd	s1, 8(sp)
   18c0e: 06 ec        	sd	ra, 24(sp)
   18c10: 4a e0        	sd	s2, 0(sp)
   18c12: 2a 84        	mv	s0, a0
   18c14: ae 84        	mv	s1, a1
   18c16: 85 c3        	beqz	a5, 0x18c36 <_Balloc+0x30>
   18c18: 13 97 34 00  	slli	a4, s1, 3
   18c1c: ba 97        	add	a5, a5, a4
   18c1e: 88 63        	ld	a0, 0(a5)
   18c20: 0d c5        	beqz	a0, 0x18c4a <_Balloc+0x44>
   18c22: 18 61        	ld	a4, 0(a0)
   18c24: 98 e3        	sd	a4, 0(a5)
   18c26: 23 38 05 00  	sd	zero, 16(a0)
   18c2a: e2 60        	ld	ra, 24(sp)
   18c2c: 42 64        	ld	s0, 16(sp)
   18c2e: a2 64        	ld	s1, 8(sp)
   18c30: 02 69        	ld	s2, 0(sp)
   18c32: 05 61        	addi	sp, sp, 32
   18c34: 82 80        	ret
   18c36: 13 06 10 04  	addi	a2, zero, 65
   18c3a: a1 45        	addi	a1, zero, 8
   18c3c: ef 30 90 41  	jal	0x1c854 <_calloc_r>
   18c40: 28 fc        	sd	a0, 120(s0)
   18c42: aa 87        	mv	a5, a0
   18c44: 71 f9        	bnez	a0, 0x18c18 <_Balloc+0x12>
   18c46: 01 45        	mv	a0, zero
   18c48: cd b7        	j	0x18c2a <_Balloc+0x24>
   18c4a: 05 49        	addi	s2, zero, 1
   18c4c: 3b 16 99 00  	sllw	a2, s2, s1
   18c50: 32 89        	mv	s2, a2
   18c52: 1d 06        	addi	a2, a2, 7
   18c54: 0a 06        	slli	a2, a2, 2
   18c56: 85 45        	addi	a1, zero, 1
   18c58: 22 85        	mv	a0, s0
   18c5a: ef 30 b0 3f  	jal	0x1c854 <_calloc_r>
   18c5e: 65 d5        	beqz	a0, 0x18c46 <_Balloc+0x40>
   18c60: 04 c5        	sw	s1, 8(a0)
   18c62: 23 26 25 01  	sw	s2, 12(a0)
   18c66: c1 b7        	j	0x18c26 <_Balloc+0x20>

0000000000018c68 <_Bfree>:
   18c68: 81 c9        	beqz	a1, 0x18c78 <_Bfree+0x10>
   18c6a: 98 45        	lw	a4, 8(a1)
   18c6c: 3c 7d        	ld	a5, 120(a0)
   18c6e: 0e 07        	slli	a4, a4, 3
   18c70: ba 97        	add	a5, a5, a4
   18c72: 98 63        	ld	a4, 0(a5)
   18c74: 98 e1        	sd	a4, 0(a1)
   18c76: 8c e3        	sd	a1, 0(a5)
   18c78: 82 80        	ret

0000000000018c7a <__multadd>:
   18c7a: 79 71        	addi	sp, sp, -48
   18c7c: 22 f0        	sd	s0, 32(sp)
   18c7e: c0 49        	lw	s0, 20(a1)
   18c80: 41 63        	lui	t1, 16
   18c82: 4a e8        	sd	s2, 16(sp)
   18c84: 4e e4        	sd	s3, 8(sp)
   18c86: 06 f4        	sd	ra, 40(sp)
   18c88: 26 ec        	sd	s1, 24(sp)
   18c8a: 52 e0        	sd	s4, 0(sp)
   18c8c: 2e 89        	mv	s2, a1
   18c8e: aa 89        	mv	s3, a0
   18c90: 13 88 85 01  	addi	a6, a1, 24
   18c94: 81 48        	mv	a7, zero
   18c96: 7d 13        	addi	t1, t1, -1
   18c98: 83 27 08 00  	lw	a5, 0(a6)
   18c9c: 11 08        	addi	a6, a6, 4
   18c9e: 85 28        	addiw	a7, a7, 1
   18ca0: 33 f7 67 00  	and	a4, a5, t1
   18ca4: 3b 07 c7 02  	mulw	a4, a4, a2
   18ca8: 9b d7 07 01  	srliw	a5, a5, 16
   18cac: bb 87 c7 02  	mulw	a5, a5, a2
   18cb0: 35 9f        	addw	a4, a4, a3
   18cb2: 9b 56 07 01  	srliw	a3, a4, 16
   18cb6: 33 77 e3 00  	and	a4, t1, a4
   18cba: b5 9f        	addw	a5, a5, a3
   18cbc: 9b 96 07 01  	slliw	a3, a5, 16
   18cc0: 35 9f        	addw	a4, a4, a3
   18cc2: 23 2e e8 fe  	sw	a4, -4(a6)
   18cc6: 9b d4 07 01  	srliw	s1, a5, 16
   18cca: 9b d6 07 01  	srliw	a3, a5, 16
   18cce: e3 c5 88 fc  	blt	a7, s0, 0x18c98 <__multadd+0x1e>
   18cd2: 89 ce        	beqz	a3, 0x18cec <__multadd+0x72>
   18cd4: 83 27 c9 00  	lw	a5, 12(s2)
   18cd8: 63 53 f4 02  	bge	s0, a5, 0x18cfe <__multadd+0x84>
   18cdc: 93 07 44 00  	addi	a5, s0, 4
   18ce0: 8a 07        	slli	a5, a5, 2
   18ce2: ca 97        	add	a5, a5, s2
   18ce4: 84 c7        	sw	s1, 8(a5)
   18ce6: 05 24        	addiw	s0, s0, 1
   18ce8: 23 2a 89 00  	sw	s0, 20(s2)
   18cec: a2 70        	ld	ra, 40(sp)
   18cee: 02 74        	ld	s0, 32(sp)
   18cf0: e2 64        	ld	s1, 24(sp)
   18cf2: a2 69        	ld	s3, 8(sp)
   18cf4: 02 6a        	ld	s4, 0(sp)
   18cf6: 4a 85        	mv	a0, s2
   18cf8: 42 69        	ld	s2, 16(sp)
   18cfa: 45 61        	addi	sp, sp, 48
   18cfc: 82 80        	ret
   18cfe: 83 25 89 00  	lw	a1, 8(s2)
   18d02: 4e 85        	mv	a0, s3
   18d04: 85 25        	addiw	a1, a1, 1
   18d06: ef f0 1f f0  	jal	0x18c06 <_Balloc>
   18d0a: 2a 8a        	mv	s4, a0
   18d0c: 1d cd        	beqz	a0, 0x18d4a <__multadd+0xd0>
   18d0e: 03 26 49 01  	lw	a2, 20(s2)
   18d12: 93 05 09 01  	addi	a1, s2, 16
   18d16: 41 05        	addi	a0, a0, 16
   18d18: 09 06        	addi	a2, a2, 2
   18d1a: 0a 06        	slli	a2, a2, 2
   18d1c: ef 40 80 21  	jal	0x1cf34 <memcpy>
   18d20: 03 27 89 00  	lw	a4, 8(s2)
   18d24: 83 b7 89 07  	ld	a5, 120(s3)
   18d28: 0e 07        	slli	a4, a4, 3
   18d2a: ba 97        	add	a5, a5, a4
   18d2c: 98 63        	ld	a4, 0(a5)
   18d2e: 23 30 e9 00  	sd	a4, 0(s2)
   18d32: 23 b0 27 01  	sd	s2, 0(a5)
   18d36: 93 07 44 00  	addi	a5, s0, 4
   18d3a: 52 89        	mv	s2, s4
   18d3c: 8a 07        	slli	a5, a5, 2
   18d3e: ca 97        	add	a5, a5, s2
   18d40: 84 c7        	sw	s1, 8(a5)
   18d42: 05 24        	addiw	s0, s0, 1
   18d44: 23 2a 89 00  	sw	s0, 20(s2)
   18d48: 55 b7        	j	0x18cec <__multadd+0x72>
   18d4a: b7 26 02 00  	lui	a3, 34
   18d4e: 37 25 02 00  	lui	a0, 34
   18d52: 93 86 06 eb  	addi	a3, a3, -336
   18d56: 01 46        	mv	a2, zero
   18d58: 93 05 50 0b  	addi	a1, zero, 181
   18d5c: 13 05 05 1b  	addi	a0, a0, 432
   18d60: ef 30 b0 29  	jal	0x1c7fa <__assert_func>

0000000000018d64 <__s2b>:
   18d64: 39 71        	addi	sp, sp, -64
   18d66: 22 f8        	sd	s0, 48(sp)
   18d68: 26 f4        	sd	s1, 40(sp)
   18d6a: 4a f0        	sd	s2, 32(sp)
   18d6c: 4e ec        	sd	s3, 24(sp)
   18d6e: 52 e8        	sd	s4, 16(sp)
   18d70: b6 84        	mv	s1, a3
   18d72: 2e 84        	mv	s0, a1
   18d74: a1 26        	addiw	a3, a3, 8
   18d76: a5 45        	addi	a1, zero, 9
   18d78: 06 fc        	sd	ra, 56(sp)
   18d7a: 56 e4        	sd	s5, 8(sp)
   18d7c: 5a e0        	sd	s6, 0(sp)
   18d7e: a5 47        	addi	a5, zero, 9
   18d80: bb c6 b6 02  	divw	a3, a3, a1
   18d84: 2a 89        	mv	s2, a0
   18d86: 32 8a        	mv	s4, a2
   18d88: ba 89        	mv	s3, a4
   18d8a: 63 d1 97 0a  	bge	a5, s1, 0x18e2c <__s2b+0xc8>
   18d8e: 85 47        	addi	a5, zero, 1
   18d90: 81 45        	mv	a1, zero
   18d92: 9b 97 17 00  	slliw	a5, a5, 1
   18d96: 85 25        	addiw	a1, a1, 1
   18d98: e3 cd d7 fe  	blt	a5, a3, 0x18d92 <__s2b+0x2e>
   18d9c: 4a 85        	mv	a0, s2
   18d9e: ef f0 9f e6  	jal	0x18c06 <_Balloc>
   18da2: aa 85        	mv	a1, a0
   18da4: 51 c5        	beqz	a0, 0x18e30 <__s2b+0xcc>
   18da6: 85 47        	addi	a5, zero, 1
   18da8: 5c c9        	sw	a5, 20(a0)
   18daa: 23 2c 35 01  	sw	s3, 24(a0)
   18dae: a5 47        	addi	a5, zero, 9
   18db0: 63 db 47 07  	bge	a5, s4, 0x18e26 <__s2b+0xc2>
   18db4: 93 17 0a 02  	slli	a5, s4, 32
   18db8: 13 0b 94 00  	addi	s6, s0, 9
   18dbc: 81 93        	srli	a5, a5, 32
   18dbe: 9b 0a 0a 00  	sext.w	s5, s4
   18dc2: 3e 94        	add	s0, s0, a5
   18dc4: da 89        	mv	s3, s6
   18dc6: 83 c6 09 00  	lbu	a3, 0(s3)
   18dca: 29 46        	addi	a2, zero, 10
   18dcc: 4a 85        	mv	a0, s2
   18dce: 9b 86 06 fd  	addiw	a3, a3, -48
   18dd2: ef f0 9f ea  	jal	0x18c7a <__multadd>
   18dd6: 85 09        	addi	s3, s3, 1
   18dd8: aa 85        	mv	a1, a0
   18dda: e3 96 89 fe  	bne	s3, s0, 0x18dc6 <__s2b+0x62>
   18dde: e1 3a        	addiw	s5, s5, -8
   18de0: 82 1a        	slli	s5, s5, 32
   18de2: 93 da 0a 02  	srli	s5, s5, 32
   18de6: 33 04 5b 01  	add	s0, s6, s5
   18dea: 63 53 9a 02  	bge	s4, s1, 0x18e10 <__s2b+0xac>
   18dee: bb 84 44 41  	subw	s1, s1, s4
   18df2: 82 14        	slli	s1, s1, 32
   18df4: 81 90        	srli	s1, s1, 32
   18df6: a2 94        	add	s1, s1, s0
   18df8: 83 46 04 00  	lbu	a3, 0(s0)
   18dfc: 29 46        	addi	a2, zero, 10
   18dfe: 4a 85        	mv	a0, s2
   18e00: 9b 86 06 fd  	addiw	a3, a3, -48
   18e04: ef f0 7f e7  	jal	0x18c7a <__multadd>
   18e08: 05 04        	addi	s0, s0, 1
   18e0a: aa 85        	mv	a1, a0
   18e0c: e3 16 94 fe  	bne	s0, s1, 0x18df8 <__s2b+0x94>
   18e10: e2 70        	ld	ra, 56(sp)
   18e12: 42 74        	ld	s0, 48(sp)
   18e14: a2 74        	ld	s1, 40(sp)
   18e16: 02 79        	ld	s2, 32(sp)
   18e18: e2 69        	ld	s3, 24(sp)
   18e1a: 42 6a        	ld	s4, 16(sp)
   18e1c: a2 6a        	ld	s5, 8(sp)
   18e1e: 02 6b        	ld	s6, 0(sp)
   18e20: 2e 85        	mv	a0, a1
   18e22: 21 61        	addi	sp, sp, 64
   18e24: 82 80        	ret
   18e26: 29 04        	addi	s0, s0, 10
   18e28: 25 4a        	addi	s4, zero, 9
   18e2a: c1 b7        	j	0x18dea <__s2b+0x86>
   18e2c: 81 45        	mv	a1, zero
   18e2e: bd b7        	j	0x18d9c <__s2b+0x38>
   18e30: b7 26 02 00  	lui	a3, 34
   18e34: 37 25 02 00  	lui	a0, 34
   18e38: 93 86 06 eb  	addi	a3, a3, -336
   18e3c: 01 46        	mv	a2, zero
   18e3e: 93 05 e0 0c  	addi	a1, zero, 206
   18e42: 13 05 05 1b  	addi	a0, a0, 432
   18e46: ef 30 50 1b  	jal	0x1c7fa <__assert_func>

0000000000018e4a <__hi0bits>:
   18e4a: aa 87        	mv	a5, a0
   18e4c: 41 67        	lui	a4, 16
   18e4e: 01 45        	mv	a0, zero
   18e50: 63 f5 e7 00  	bgeu	a5, a4, 0x18e5a <__hi0bits+0x10>
   18e54: 9b 97 07 01  	slliw	a5, a5, 16
   18e58: 41 45        	addi	a0, zero, 16
   18e5a: 37 07 00 01  	lui	a4, 4096
   18e5e: 63 f5 e7 00  	bgeu	a5, a4, 0x18e68 <__hi0bits+0x1e>
   18e62: 9b 97 87 00  	slliw	a5, a5, 8
   18e66: 21 25        	addiw	a0, a0, 8
   18e68: 37 07 00 10  	lui	a4, 65536
   18e6c: 63 f5 e7 00  	bgeu	a5, a4, 0x18e76 <__hi0bits+0x2c>
   18e70: 9b 97 47 00  	slliw	a5, a5, 4
   18e74: 11 25        	addiw	a0, a0, 4
   18e76: 37 07 00 40  	lui	a4, 262144
   18e7a: 63 e6 e7 00  	bltu	a5, a4, 0x18e86 <__hi0bits+0x3c>
   18e7e: 63 c0 07 02  	bltz	a5, 0x18e9e <__hi0bits+0x54>
   18e82: 05 25        	addiw	a0, a0, 1
   18e84: 82 80        	ret
   18e86: 9b 97 27 00  	slliw	a5, a5, 2
   18e8a: 1b 87 07 00  	sext.w	a4, a5
   18e8e: 63 49 07 00  	bltz	a4, 0x18ea0 <__hi0bits+0x56>
   18e92: 37 07 00 40  	lui	a4, 262144
   18e96: f9 8f        	and	a5, a5, a4
   18e98: 91 e7        	bnez	a5, 0x18ea4 <__hi0bits+0x5a>
   18e9a: 13 05 00 02  	addi	a0, zero, 32
   18e9e: 82 80        	ret
   18ea0: 09 25        	addiw	a0, a0, 2
   18ea2: 82 80        	ret
   18ea4: 0d 25        	addiw	a0, a0, 3
   18ea6: 82 80        	ret

0000000000018ea8 <__lo0bits>:
   18ea8: 1c 41        	lw	a5, 0(a0)
   18eaa: 2a 87        	mv	a4, a0
   18eac: 93 f6 77 00  	andi	a3, a5, 7
   18eb0: 3e 86        	mv	a2, a5
   18eb2: 81 ce        	beqz	a3, 0x18eca <__lo0bits+0x22>
   18eb4: 93 f6 17 00  	andi	a3, a5, 1
   18eb8: 01 45        	mv	a0, zero
   18eba: 9d ee        	bnez	a3, 0x18ef8 <__lo0bits+0x50>
   18ebc: 09 8a        	andi	a2, a2, 2
   18ebe: 25 ce        	beqz	a2, 0x18f36 <__lo0bits+0x8e>
   18ec0: 9b d7 17 00  	srliw	a5, a5, 1
   18ec4: 1c c3        	sw	a5, 0(a4)
   18ec6: 05 45        	addi	a0, zero, 1
   18ec8: 82 80        	ret
   18eca: 93 96 07 03  	slli	a3, a5, 48
   18ece: c1 92        	srli	a3, a3, 48
   18ed0: 01 45        	mv	a0, zero
   18ed2: b9 ca        	beqz	a3, 0x18f28 <__lo0bits+0x80>
   18ed4: 93 f6 f7 0f  	andi	a3, a5, 255
   18ed8: a9 c2        	beqz	a3, 0x18f1a <__lo0bits+0x72>
   18eda: 93 f6 f7 00  	andi	a3, a5, 15
   18ede: 9d c6        	beqz	a3, 0x18f0c <__lo0bits+0x64>
   18ee0: 93 f6 37 00  	andi	a3, a5, 3
   18ee4: 89 ce        	beqz	a3, 0x18efe <__lo0bits+0x56>
   18ee6: 93 f6 17 00  	andi	a3, a5, 1
   18eea: 81 ea        	bnez	a3, 0x18efa <__lo0bits+0x52>
   18eec: 9b d7 17 00  	srliw	a5, a5, 1
   18ef0: 05 25        	addiw	a0, a0, 1
   18ef2: 81 e7        	bnez	a5, 0x18efa <__lo0bits+0x52>
   18ef4: 13 05 00 02  	addi	a0, zero, 32
   18ef8: 82 80        	ret
   18efa: 1c c3        	sw	a5, 0(a4)
   18efc: 82 80        	ret
   18efe: 9b d7 27 00  	srliw	a5, a5, 2
   18f02: 93 f6 17 00  	andi	a3, a5, 1
   18f06: 09 25        	addiw	a0, a0, 2
   18f08: ed fa        	bnez	a3, 0x18efa <__lo0bits+0x52>
   18f0a: cd b7        	j	0x18eec <__lo0bits+0x44>
   18f0c: 9b d7 47 00  	srliw	a5, a5, 4
   18f10: 93 f6 37 00  	andi	a3, a5, 3
   18f14: 11 25        	addiw	a0, a0, 4
   18f16: e1 fa        	bnez	a3, 0x18ee6 <__lo0bits+0x3e>
   18f18: dd b7        	j	0x18efe <__lo0bits+0x56>
   18f1a: 9b d7 87 00  	srliw	a5, a5, 8
   18f1e: 93 f6 f7 00  	andi	a3, a5, 15
   18f22: 21 25        	addiw	a0, a0, 8
   18f24: d5 fe        	bnez	a3, 0x18ee0 <__lo0bits+0x38>
   18f26: dd b7        	j	0x18f0c <__lo0bits+0x64>
   18f28: 9b d7 07 01  	srliw	a5, a5, 16
   18f2c: 93 f6 f7 0f  	andi	a3, a5, 255
   18f30: 41 45        	addi	a0, zero, 16
   18f32: c5 f6        	bnez	a3, 0x18eda <__lo0bits+0x32>
   18f34: dd b7        	j	0x18f1a <__lo0bits+0x72>
   18f36: 9b d7 27 00  	srliw	a5, a5, 2
   18f3a: 1c c3        	sw	a5, 0(a4)
   18f3c: 09 45        	addi	a0, zero, 2
   18f3e: 82 80        	ret

0000000000018f40 <__i2b>:
   18f40: 3c 7d        	ld	a5, 120(a0)
   18f42: 01 11        	addi	sp, sp, -32
   18f44: 22 e8        	sd	s0, 16(sp)
   18f46: 26 e4        	sd	s1, 8(sp)
   18f48: 06 ec        	sd	ra, 24(sp)
   18f4a: 2a 84        	mv	s0, a0
   18f4c: ae 84        	mv	s1, a1
   18f4e: 91 cf        	beqz	a5, 0x18f6a <__i2b+0x2a>
   18f50: 88 67        	ld	a0, 8(a5)
   18f52: 29 c1        	beqz	a0, 0x18f94 <__i2b+0x54>
   18f54: 18 61        	ld	a4, 0(a0)
   18f56: 98 e7        	sd	a4, 8(a5)
   18f58: e2 60        	ld	ra, 24(sp)
   18f5a: 42 64        	ld	s0, 16(sp)
   18f5c: 85 47        	addi	a5, zero, 1
   18f5e: 82 17        	slli	a5, a5, 32
   18f60: 04 cd        	sw	s1, 24(a0)
   18f62: 1c e9        	sd	a5, 16(a0)
   18f64: a2 64        	ld	s1, 8(sp)
   18f66: 05 61        	addi	sp, sp, 32
   18f68: 82 80        	ret
   18f6a: 13 06 10 04  	addi	a2, zero, 65
   18f6e: a1 45        	addi	a1, zero, 8
   18f70: ef 30 50 0e  	jal	0x1c854 <_calloc_r>
   18f74: 28 fc        	sd	a0, 120(s0)
   18f76: aa 87        	mv	a5, a0
   18f78: 61 fd        	bnez	a0, 0x18f50 <__i2b+0x10>
   18f7a: b7 26 02 00  	lui	a3, 34
   18f7e: 37 25 02 00  	lui	a0, 34
   18f82: 93 86 06 eb  	addi	a3, a3, -336
   18f86: 01 46        	mv	a2, zero
   18f88: 93 05 00 14  	addi	a1, zero, 320
   18f8c: 13 05 05 1b  	addi	a0, a0, 432
   18f90: ef 30 b0 06  	jal	0x1c7fa <__assert_func>
   18f94: 13 06 40 02  	addi	a2, zero, 36
   18f98: 85 45        	addi	a1, zero, 1
   18f9a: 22 85        	mv	a0, s0
   18f9c: ef 30 90 0b  	jal	0x1c854 <_calloc_r>
   18fa0: 69 dd        	beqz	a0, 0x18f7a <__i2b+0x3a>
   18fa2: 85 47        	addi	a5, zero, 1
   18fa4: 86 17        	slli	a5, a5, 33
   18fa6: 85 07        	addi	a5, a5, 1
   18fa8: 1c e5        	sd	a5, 8(a0)
   18faa: 7d b7        	j	0x18f58 <__i2b+0x18>

0000000000018fac <__multiply>:
   18fac: 79 71        	addi	sp, sp, -48
   18fae: 4a e8        	sd	s2, 16(sp)
   18fb0: 4e e4        	sd	s3, 8(sp)
   18fb2: 03 29 46 01  	lw	s2, 20(a2)
   18fb6: 83 a9 45 01  	lw	s3, 20(a1)
   18fba: 26 ec        	sd	s1, 24(sp)
   18fbc: 52 e0        	sd	s4, 0(sp)
   18fbe: 06 f4        	sd	ra, 40(sp)
   18fc0: 22 f0        	sd	s0, 32(sp)
   18fc2: 2e 8a        	mv	s4, a1
   18fc4: b2 84        	mv	s1, a2
   18fc6: 63 c7 29 01  	blt	s3, s2, 0x18fd4 <__multiply+0x28>
   18fca: ca 87        	mv	a5, s2
   18fcc: ae 84        	mv	s1, a1
   18fce: 4e 89        	mv	s2, s3
   18fd0: 32 8a        	mv	s4, a2
   18fd2: be 89        	mv	s3, a5
   18fd4: dc 44        	lw	a5, 12(s1)
   18fd6: 3b 04 39 01  	addw	s0, s2, s3
   18fda: 8c 44        	lw	a1, 8(s1)
   18fdc: 63 d3 87 00  	bge	a5, s0, 0x18fe2 <__multiply+0x36>
   18fe0: 85 25        	addiw	a1, a1, 1
   18fe2: ef f0 5f c2  	jal	0x18c06 <_Balloc>
   18fe6: 63 09 05 16  	beqz	a0, 0x19158 <__multiply+0x1ac>
   18fea: 13 03 85 01  	addi	t1, a0, 24
   18fee: 93 18 24 00  	slli	a7, s0, 2
   18ff2: 9a 98        	add	a7, a7, t1
   18ff4: 63 75 13 03  	bgeu	t1, a7, 0x1901e <__multiply+0x72>
   18ff8: b3 87 a8 40  	sub	a5, a7, a0
   18ffc: 9d 17        	addi	a5, a5, -25
   18ffe: 89 83        	srli	a5, a5, 2
   19000: 85 07        	addi	a5, a5, 1
   19002: 1a 87        	mv	a4, t1
   19004: 73 26 20 c2  	csrr	a2, vlenb
   19008: d7 76 00 05  	<unknown>
   1900c: 57 3c 00 5e  	<unknown>
   19010: d7 f6 07 05  	<unknown>
   19014: 27 6c 07 02  	<unknown>
   19018: 95 8f        	sub	a5, a5, a3
   1901a: 32 97        	add	a4, a4, a2
   1901c: f5 fb        	bnez	a5, 0x19010 <__multiply+0x64>
   1901e: 13 08 8a 01  	addi	a6, s4, 24
   19022: 13 9e 29 00  	slli	t3, s3, 2
   19026: 93 8e 84 01  	addi	t4, s1, 24
   1902a: 93 15 29 00  	slli	a1, s2, 2
   1902e: 42 9e        	add	t3, t3, a6
   19030: f6 95        	add	a1, a1, t4
   19032: 63 7a c8 0f  	bgeu	a6, t3, 0x19126 <__multiply+0x17a>
   19036: 93 87 94 01  	addi	a5, s1, 25
   1903a: 11 4f        	addi	t5, zero, 4
   1903c: 63 f7 f5 10  	bgeu	a1, a5, 0x1914a <__multiply+0x19e>
   19040: c1 66        	lui	a3, 16
   19042: fd 16        	addi	a3, a3, -1
   19044: 09 a8        	j	0x19056 <__multiply+0xaa>
   19046: 9b df 0f 01  	srliw	t6, t6, 16
   1904a: 63 9d 0f 06  	bnez	t6, 0x190c4 <__multiply+0x118>
   1904e: 11 08        	addi	a6, a6, 4
   19050: 11 03        	addi	t1, t1, 4
   19052: 63 7a c8 0d  	bgeu	a6, t3, 0x19126 <__multiply+0x17a>
   19056: 83 2f 08 00  	lw	t6, 0(a6)
   1905a: b3 f3 df 00  	and	t2, t6, a3
   1905e: e3 84 03 fe  	beqz	t2, 0x19046 <__multiply+0x9a>
   19062: 9a 82        	mv	t0, t1
   19064: f6 8f        	mv	t6, t4
   19066: 81 44        	mv	s1, zero
   19068: 83 a7 0f 00  	lw	a5, 0(t6)
   1906c: 03 a6 02 00  	lw	a2, 0(t0)
   19070: 91 02        	addi	t0, t0, 4
   19072: 33 f7 d7 00  	and	a4, a5, a3
   19076: 3b 07 77 02  	mulw	a4, a4, t2
   1907a: 9b d7 07 01  	srliw	a5, a5, 16
   1907e: 33 79 d6 00  	and	s2, a2, a3
   19082: 1b 56 06 01  	srliw	a2, a2, 16
   19086: 91 0f        	addi	t6, t6, 4
   19088: bb 87 77 02  	mulw	a5, a5, t2
   1908c: 3b 07 27 01  	addw	a4, a4, s2
   19090: 25 9f        	addw	a4, a4, s1
   19092: 9b 54 07 01  	srliw	s1, a4, 16
   19096: 75 8f        	and	a4, a4, a3
   19098: b1 9f        	addw	a5, a5, a2
   1909a: a5 9f        	addw	a5, a5, s1
   1909c: 1b 96 07 01  	slliw	a2, a5, 16
   190a0: 51 8f        	or	a4, a4, a2
   190a2: 23 ae e2 fe  	sw	a4, -4(t0)
   190a6: 9b d4 07 01  	srliw	s1, a5, 16
   190aa: 1b d7 07 01  	srliw	a4, a5, 16
   190ae: e3 ed bf fa  	bltu	t6, a1, 0x19068 <__multiply+0xbc>
   190b2: b3 07 e3 01  	add	a5, t1, t5
   190b6: 98 c3        	sw	a4, 0(a5)
   190b8: 83 2f 08 00  	lw	t6, 0(a6)
   190bc: 9b df 0f 01  	srliw	t6, t6, 16
   190c0: e3 87 0f f8  	beqz	t6, 0x1904e <__multiply+0xa2>
   190c4: 03 27 03 00  	lw	a4, 0(t1)
   190c8: 9a 82        	mv	t0, t1
   190ca: 76 86        	mv	a2, t4
   190cc: ba 83        	mv	t2, a4
   190ce: 81 44        	mv	s1, zero
   190d0: 1c 42        	lw	a5, 0(a2)
   190d2: 9b d9 03 01  	srliw	s3, t2, 16
   190d6: 75 8f        	and	a4, a4, a3
   190d8: f5 8f        	and	a5, a5, a3
   190da: bb 87 f7 03  	mulw	a5, a5, t6
   190de: 83 a3 42 00  	lw	t2, 4(t0)
   190e2: 91 02        	addi	t0, t0, 4
   190e4: 11 06        	addi	a2, a2, 4
   190e6: 33 f9 d3 00  	and	s2, t2, a3
   190ea: a5 9f        	addw	a5, a5, s1
   190ec: bb 87 37 01  	addw	a5, a5, s3
   190f0: 9b 94 07 01  	slliw	s1, a5, 16
   190f4: 45 8f        	or	a4, a4, s1
   190f6: 23 ae e2 fe  	sw	a4, -4(t0)
   190fa: 03 57 e6 ff  	lhu	a4, -2(a2)
   190fe: 9b d7 07 01  	srliw	a5, a5, 16
   19102: 3b 07 f7 03  	mulw	a4, a4, t6
   19106: 3b 07 27 01  	addw	a4, a4, s2
   1910a: b9 9f        	addw	a5, a5, a4
   1910c: 1b 87 07 00  	sext.w	a4, a5
   19110: 9b d4 07 01  	srliw	s1, a5, 16
   19114: e3 6e b6 fa  	bltu	a2, a1, 0x190d0 <__multiply+0x124>
   19118: 33 07 e3 01  	add	a4, t1, t5
   1911c: 1c c3        	sw	a5, 0(a4)
   1911e: 11 08        	addi	a6, a6, 4
   19120: 11 03        	addi	t1, t1, 4
   19122: e3 6a c8 f3  	bltu	a6, t3, 0x19056 <__multiply+0xaa>
   19126: 63 45 80 00  	bgtz	s0, 0x19130 <__multiply+0x184>
   1912a: 39 a0        	j	0x19138 <__multiply+0x18c>
   1912c: 7d 34        	addiw	s0, s0, -1
   1912e: 09 c4        	beqz	s0, 0x19138 <__multiply+0x18c>
   19130: 83 a7 c8 ff  	lw	a5, -4(a7)
   19134: f1 18        	addi	a7, a7, -4
   19136: fd db        	beqz	a5, 0x1912c <__multiply+0x180>
   19138: a2 70        	ld	ra, 40(sp)
   1913a: 40 c9        	sw	s0, 20(a0)
   1913c: 02 74        	ld	s0, 32(sp)
   1913e: e2 64        	ld	s1, 24(sp)
   19140: 42 69        	ld	s2, 16(sp)
   19142: a2 69        	ld	s3, 8(sp)
   19144: 02 6a        	ld	s4, 0(sp)
   19146: 45 61        	addi	sp, sp, 48
   19148: 82 80        	ret
   1914a: b3 87 95 40  	sub	a5, a1, s1
   1914e: 9d 17        	addi	a5, a5, -25
   19150: f1 9b        	andi	a5, a5, -4
   19152: 13 8f 47 00  	addi	t5, a5, 4
   19156: ed b5        	j	0x19040 <__multiply+0x94>
   19158: b7 26 02 00  	lui	a3, 34
   1915c: 37 25 02 00  	lui	a0, 34
   19160: 93 86 06 eb  	addi	a3, a3, -336
   19164: 01 46        	mv	a2, zero
   19166: 93 05 d0 15  	addi	a1, zero, 349
   1916a: 13 05 05 1b  	addi	a0, a0, 432
   1916e: ef 30 c0 68  	jal	0x1c7fa <__assert_func>

0000000000019172 <__pow5mult>:
   19172: 79 71        	addi	sp, sp, -48
   19174: 22 f0        	sd	s0, 32(sp)
   19176: 4a e8        	sd	s2, 16(sp)
   19178: 4e e4        	sd	s3, 8(sp)
   1917a: 06 f4        	sd	ra, 40(sp)
   1917c: 26 ec        	sd	s1, 24(sp)
   1917e: 93 77 36 00  	andi	a5, a2, 3
   19182: 32 84        	mv	s0, a2
   19184: aa 89        	mv	s3, a0
   19186: 2e 89        	mv	s2, a1
   19188: a5 eb        	bnez	a5, 0x191f8 <__pow5mult+0x86>
   1918a: 09 84        	srai	s0, s0, 2
   1918c: 21 c4        	beqz	s0, 0x191d4 <__pow5mult+0x62>
   1918e: 83 b4 09 07  	ld	s1, 112(s3)
   19192: c1 c0        	beqz	s1, 0x19212 <__pow5mult+0xa0>
   19194: 93 77 14 00  	andi	a5, s0, 1
   19198: 89 eb        	bnez	a5, 0x191aa <__pow5mult+0x38>
   1919a: 05 84        	srai	s0, s0, 1
   1919c: 05 cc        	beqz	s0, 0x191d4 <__pow5mult+0x62>
   1919e: 88 60        	ld	a0, 0(s1)
   191a0: 31 c1        	beqz	a0, 0x191e4 <__pow5mult+0x72>
   191a2: aa 84        	mv	s1, a0
   191a4: 93 77 14 00  	andi	a5, s0, 1
   191a8: ed db        	beqz	a5, 0x1919a <__pow5mult+0x28>
   191aa: 26 86        	mv	a2, s1
   191ac: ca 85        	mv	a1, s2
   191ae: 4e 85        	mv	a0, s3
   191b0: ef f0 df df  	jal	0x18fac <__multiply>
   191b4: 63 0d 09 00  	beqz	s2, 0x191ce <__pow5mult+0x5c>
   191b8: 03 27 89 00  	lw	a4, 8(s2)
   191bc: 83 b7 89 07  	ld	a5, 120(s3)
   191c0: 0e 07        	slli	a4, a4, 3
   191c2: ba 97        	add	a5, a5, a4
   191c4: 98 63        	ld	a4, 0(a5)
   191c6: 23 30 e9 00  	sd	a4, 0(s2)
   191ca: 23 b0 27 01  	sd	s2, 0(a5)
   191ce: 05 84        	srai	s0, s0, 1
   191d0: 2a 89        	mv	s2, a0
   191d2: 71 f4        	bnez	s0, 0x1919e <__pow5mult+0x2c>
   191d4: a2 70        	ld	ra, 40(sp)
   191d6: 02 74        	ld	s0, 32(sp)
   191d8: e2 64        	ld	s1, 24(sp)
   191da: a2 69        	ld	s3, 8(sp)
   191dc: 4a 85        	mv	a0, s2
   191de: 42 69        	ld	s2, 16(sp)
   191e0: 45 61        	addi	sp, sp, 48
   191e2: 82 80        	ret
   191e4: 26 86        	mv	a2, s1
   191e6: a6 85        	mv	a1, s1
   191e8: 4e 85        	mv	a0, s3
   191ea: ef f0 3f dc  	jal	0x18fac <__multiply>
   191ee: 88 e0        	sd	a0, 0(s1)
   191f0: 23 30 05 00  	sd	zero, 0(a0)
   191f4: aa 84        	mv	s1, a0
   191f6: 7d b7        	j	0x191a4 <__pow5mult+0x32>
   191f8: fd 37        	addiw	a5, a5, -1
   191fa: 37 27 02 00  	lui	a4, 34
   191fe: 13 07 07 21  	addi	a4, a4, 528
   19202: 8a 07        	slli	a5, a5, 2
   19204: ba 97        	add	a5, a5, a4
   19206: 90 43        	lw	a2, 0(a5)
   19208: 81 46        	mv	a3, zero
   1920a: ef f0 1f a7  	jal	0x18c7a <__multadd>
   1920e: 2a 89        	mv	s2, a0
   19210: ad bf        	j	0x1918a <__pow5mult+0x18>
   19212: 85 45        	addi	a1, zero, 1
   19214: 4e 85        	mv	a0, s3
   19216: ef f0 1f 9f  	jal	0x18c06 <_Balloc>
   1921a: aa 84        	mv	s1, a0
   1921c: 19 c9        	beqz	a0, 0x19232 <__pow5mult+0xc0>
   1921e: 93 07 10 27  	addi	a5, zero, 625
   19222: 1c cd        	sw	a5, 24(a0)
   19224: 85 47        	addi	a5, zero, 1
   19226: 5c c9        	sw	a5, 20(a0)
   19228: 23 b8 a9 06  	sd	a0, 112(s3)
   1922c: 23 30 05 00  	sd	zero, 0(a0)
   19230: 95 b7        	j	0x19194 <__pow5mult+0x22>
   19232: b7 26 02 00  	lui	a3, 34
   19236: 37 25 02 00  	lui	a0, 34
   1923a: 93 86 06 eb  	addi	a3, a3, -336
   1923e: 01 46        	mv	a2, zero
   19240: 93 05 00 14  	addi	a1, zero, 320
   19244: 13 05 05 1b  	addi	a0, a0, 432
   19248: ef 30 20 5b  	jal	0x1c7fa <__assert_func>

000000000001924c <__lshift>:
   1924c: 39 71        	addi	sp, sp, -64
   1924e: 26 f4        	sd	s1, 40(sp)
   19250: c4 49        	lw	s1, 20(a1)
   19252: dc 45        	lw	a5, 12(a1)
   19254: 5a e0        	sd	s6, 0(sp)
   19256: 1b 5b 56 40  	sraiw	s6, a2, 5
   1925a: bb 04 9b 00  	addw	s1, s6, s1
   1925e: 22 f8        	sd	s0, 48(sp)
   19260: 4a f0        	sd	s2, 32(sp)
   19262: 4e ec        	sd	s3, 24(sp)
   19264: 52 e8        	sd	s4, 16(sp)
   19266: 56 e4        	sd	s5, 8(sp)
   19268: 06 fc        	sd	ra, 56(sp)
   1926a: 9b 89 14 00  	addiw	s3, s1, 1
   1926e: 2e 89        	mv	s2, a1
   19270: b2 8a        	mv	s5, a2
   19272: 8c 45        	lw	a1, 8(a1)
   19274: 2a 8a        	mv	s4, a0
   19276: 13 54 56 40  	srai	s0, a2, 5
   1927a: 63 d7 37 01  	bge	a5, s3, 0x19288 <__lshift+0x3c>
   1927e: 9b 97 17 00  	slliw	a5, a5, 1
   19282: 85 25        	addiw	a1, a1, 1
   19284: e3 cd 37 ff  	blt	a5, s3, 0x1927e <__lshift+0x32>
   19288: 52 85        	mv	a0, s4
   1928a: ef f0 df 97  	jal	0x18c06 <_Balloc>
   1928e: 79 cd        	beqz	a0, 0x1936c <__lshift+0x120>
   19290: 93 06 85 01  	addi	a3, a0, 24
   19294: 63 56 80 02  	blez	s0, 0x192c0 <__lshift+0x74>
   19298: a2 87        	mv	a5, s0
   1929a: 36 87        	mv	a4, a3
   1929c: f3 25 20 c2  	csrr	a1, vlenb
   192a0: 57 76 00 05  	<unknown>
   192a4: 57 3c 00 5e  	<unknown>
   192a8: 57 f6 07 05  	<unknown>
   192ac: 27 6c 07 02  	<unknown>
   192b0: 91 8f        	sub	a5, a5, a2
   192b2: 2e 97        	add	a4, a4, a1
   192b4: f5 fb        	bnez	a5, 0x192a8 <__lshift+0x5c>
   192b6: 93 17 0b 02  	slli	a5, s6, 32
   192ba: 13 db e7 01  	srli	s6, a5, 30
   192be: da 96        	add	a3, a3, s6
   192c0: 83 28 49 01  	lw	a7, 20(s2)
   192c4: 93 07 89 01  	addi	a5, s2, 24
   192c8: 13 f6 fa 01  	andi	a2, s5, 31
   192cc: 8a 08        	slli	a7, a7, 2
   192ce: be 98        	add	a7, a7, a5
   192d0: 3d ce        	beqz	a2, 0x1934e <__lshift+0x102>
   192d2: 13 03 00 02  	addi	t1, zero, 32
   192d6: 3b 03 c3 40  	subw	t1, t1, a2
   192da: b6 85        	mv	a1, a3
   192dc: 01 47        	mv	a4, zero
   192de: 03 a8 07 00  	lw	a6, 0(a5)
   192e2: 91 05        	addi	a1, a1, 4
   192e4: 91 07        	addi	a5, a5, 4
   192e6: 3b 18 c8 00  	sllw	a6, a6, a2
   192ea: 33 67 07 01  	or	a4, a4, a6
   192ee: 23 ae e5 fe  	sw	a4, -4(a1)
   192f2: 03 a8 c7 ff  	lw	a6, -4(a5)
   192f6: 3b 57 68 00  	srlw	a4, a6, t1
   192fa: 3a 88        	mv	a6, a4
   192fc: e3 e1 17 ff  	bltu	a5, a7, 0x192de <__lshift+0x92>
   19300: 93 07 99 01  	addi	a5, s2, 25
   19304: 11 46        	addi	a2, zero, 4
   19306: 63 e8 f8 00  	bltu	a7, a5, 0x19316 <__lshift+0xca>
   1930a: b3 87 28 41  	sub	a5, a7, s2
   1930e: 9d 17        	addi	a5, a5, -25
   19310: f1 9b        	andi	a5, a5, -4
   19312: 13 86 47 00  	addi	a2, a5, 4
   19316: b2 96        	add	a3, a3, a2
   19318: 23 a0 06 01  	sw	a6, 0(a3)
   1931c: 11 e3        	bnez	a4, 0x19320 <__lshift+0xd4>
   1931e: a6 89        	mv	s3, s1
   19320: 03 27 89 00  	lw	a4, 8(s2)
   19324: 83 37 8a 07  	ld	a5, 120(s4)
   19328: e2 70        	ld	ra, 56(sp)
   1932a: 0e 07        	slli	a4, a4, 3
   1932c: ba 97        	add	a5, a5, a4
   1932e: 98 63        	ld	a4, 0(a5)
   19330: 23 2a 35 01  	sw	s3, 20(a0)
   19334: 42 74        	ld	s0, 48(sp)
   19336: 23 30 e9 00  	sd	a4, 0(s2)
   1933a: 23 b0 27 01  	sd	s2, 0(a5)
   1933e: a2 74        	ld	s1, 40(sp)
   19340: 02 79        	ld	s2, 32(sp)
   19342: e2 69        	ld	s3, 24(sp)
   19344: 42 6a        	ld	s4, 16(sp)
   19346: a2 6a        	ld	s5, 8(sp)
   19348: 02 6b        	ld	s6, 0(sp)
   1934a: 21 61        	addi	sp, sp, 64
   1934c: 82 80        	ret
   1934e: 98 43        	lw	a4, 0(a5)
   19350: 91 07        	addi	a5, a5, 4
   19352: 91 06        	addi	a3, a3, 4
   19354: 23 ae e6 fe  	sw	a4, -4(a3)
   19358: e3 f3 17 fd  	bgeu	a5, a7, 0x1931e <__lshift+0xd2>
   1935c: 98 43        	lw	a4, 0(a5)
   1935e: 91 07        	addi	a5, a5, 4
   19360: 91 06        	addi	a3, a3, 4
   19362: 23 ae e6 fe  	sw	a4, -4(a3)
   19366: e3 e4 17 ff  	bltu	a5, a7, 0x1934e <__lshift+0x102>
   1936a: 55 bf        	j	0x1931e <__lshift+0xd2>
   1936c: b7 26 02 00  	lui	a3, 34
   19370: 37 25 02 00  	lui	a0, 34
   19374: 93 86 06 eb  	addi	a3, a3, -336
   19378: 01 46        	mv	a2, zero
   1937a: 93 05 90 1d  	addi	a1, zero, 473
   1937e: 13 05 05 1b  	addi	a0, a0, 432
   19382: ef 30 80 47  	jal	0x1c7fa <__assert_func>

0000000000019386 <__mcmp>:
   19386: dc 49        	lw	a5, 20(a1)
   19388: 2a 88        	mv	a6, a0
   1938a: 48 49        	lw	a0, 20(a0)
   1938c: 1d 9d        	subw	a0, a0, a5
   1938e: 1d e5        	bnez	a0, 0x193bc <__mcmp+0x36>
   19390: 8a 07        	slli	a5, a5, 2
   19392: 61 08        	addi	a6, a6, 24
   19394: e1 05        	addi	a1, a1, 24
   19396: 33 07 f8 00  	add	a4, a6, a5
   1939a: ae 97        	add	a5, a5, a1
   1939c: 19 a0        	j	0x193a2 <__mcmp+0x1c>
   1939e: 63 7f e8 00  	bgeu	a6, a4, 0x193bc <__mcmp+0x36>
   193a2: 03 26 c7 ff  	lw	a2, -4(a4)
   193a6: 83 a6 c7 ff  	lw	a3, -4(a5)
   193aa: 71 17        	addi	a4, a4, -4
   193ac: f1 17        	addi	a5, a5, -4
   193ae: e3 08 d6 fe  	beq	a2, a3, 0x1939e <__mcmp+0x18>
   193b2: 05 45        	addi	a0, zero, 1
   193b4: 63 74 d6 00  	bgeu	a2, a3, 0x193bc <__mcmp+0x36>
   193b8: 7d 55        	addi	a0, zero, -1
   193ba: 82 80        	ret
   193bc: 82 80        	ret

00000000000193be <__mdiff>:
   193be: 01 11        	addi	sp, sp, -32
   193c0: 5c 4a        	lw	a5, 20(a2)
   193c2: 4a e0        	sd	s2, 0(sp)
   193c4: 03 a9 45 01  	lw	s2, 20(a1)
   193c8: 22 e8        	sd	s0, 16(sp)
   193ca: 26 e4        	sd	s1, 8(sp)
   193cc: 06 ec        	sd	ra, 24(sp)
   193ce: 3b 09 f9 40  	subw	s2, s2, a5
   193d2: ae 84        	mv	s1, a1
   193d4: 32 84        	mv	s0, a2
   193d6: 63 1c 09 02  	bnez	s2, 0x1940e <__mdiff+0x50>
   193da: 93 96 27 00  	slli	a3, a5, 2
   193de: e1 05        	addi	a1, a1, 24
   193e0: 13 07 86 01  	addi	a4, a2, 24
   193e4: b3 87 d5 00  	add	a5, a1, a3
   193e8: 36 97        	add	a4, a4, a3
   193ea: 19 a0        	j	0x193f0 <__mdiff+0x32>
   193ec: 63 f6 f5 12  	bgeu	a1, a5, 0x19518 <__mdiff+0x15a>
   193f0: 03 a6 c7 ff  	lw	a2, -4(a5)
   193f4: 83 26 c7 ff  	lw	a3, -4(a4)
   193f8: f1 17        	addi	a5, a5, -4
   193fa: 71 17        	addi	a4, a4, -4
   193fc: e3 08 d6 fe  	beq	a2, a3, 0x193ec <__mdiff+0x2e>
   19400: 63 7a d6 00  	bgeu	a2, a3, 0x19414 <__mdiff+0x56>
   19404: a6 87        	mv	a5, s1
   19406: 05 49        	addi	s2, zero, 1
   19408: a2 84        	mv	s1, s0
   1940a: 3e 84        	mv	s0, a5
   1940c: 21 a0        	j	0x19414 <__mdiff+0x56>
   1940e: e3 4b 09 fe  	bltz	s2, 0x19404 <__mdiff+0x46>
   19412: 01 49        	mv	s2, zero
   19414: 8c 44        	lw	a1, 8(s1)
   19416: ef f0 0f ff  	jal	0x18c06 <_Balloc>
   1941a: 63 03 05 14  	beqz	a0, 0x19560 <__mdiff+0x1a2>
   1941e: 03 a3 44 01  	lw	t1, 20(s1)
   19422: 03 2f 44 01  	lw	t5, 20(s0)
   19426: 93 82 84 01  	addi	t0, s1, 24
   1942a: 13 1e 23 00  	slli	t3, t1, 2
   1942e: 13 08 84 01  	addi	a6, s0, 24
   19432: 0a 0f        	slli	t5, t5, 2
   19434: 93 0f 85 01  	addi	t6, a0, 24
   19438: c1 68        	lui	a7, 16
   1943a: 23 28 25 01  	sw	s2, 16(a0)
   1943e: 16 9e        	add	t3, t3, t0
   19440: 42 9f        	add	t5, t5, a6
   19442: 7e 86        	mv	a2, t6
   19444: 96 8e        	mv	t4, t0
   19446: 81 45        	mv	a1, zero
   19448: fd 18        	addi	a7, a7, -1
   1944a: 03 a7 0e 00  	lw	a4, 0(t4)
   1944e: 83 26 08 00  	lw	a3, 0(a6)
   19452: 11 06        	addi	a2, a2, 4
   19454: b3 77 17 01  	and	a5, a4, a7
   19458: b3 f3 16 01  	and	t2, a3, a7
   1945c: bb 87 77 40  	subw	a5, a5, t2
   19460: ad 9f        	addw	a5, a5, a1
   19462: 9b d6 06 01  	srliw	a3, a3, 16
   19466: 1b 57 07 01  	srliw	a4, a4, 16
   1946a: 15 9f        	subw	a4, a4, a3
   1946c: 9b d6 07 41  	sraiw	a3, a5, 16
   19470: 35 9f        	addw	a4, a4, a3
   19472: b3 f7 f8 00  	and	a5, a7, a5
   19476: 9b 16 07 01  	slliw	a3, a4, 16
   1947a: d5 8f        	or	a5, a5, a3
   1947c: 81 27        	sext.w	a5, a5
   1947e: 11 08        	addi	a6, a6, 4
   19480: 23 2e f6 fe  	sw	a5, -4(a2)
   19484: 91 0e        	addi	t4, t4, 4
   19486: 9b 55 07 41  	sraiw	a1, a4, 16
   1948a: e3 60 e8 fd  	bltu	a6, t5, 0x1944a <__mdiff+0x8c>
   1948e: 33 07 8f 40  	sub	a4, t5, s0
   19492: 1d 17        	addi	a4, a4, -25
   19494: 65 04        	addi	s0, s0, 25
   19496: 93 5e 27 00  	srli	t4, a4, 2
   1949a: 11 47        	addi	a4, zero, 4
   1949c: 63 65 8f 00  	bltu	t5, s0, 0x194a6 <__mdiff+0xe8>
   194a0: 13 87 1e 00  	addi	a4, t4, 1
   194a4: 0a 07        	slli	a4, a4, 2
   194a6: ba 92        	add	t0, t0, a4
   194a8: 33 88 ef 00  	add	a6, t6, a4
   194ac: c1 68        	lui	a7, 16
   194ae: 42 86        	mv	a2, a6
   194b0: 96 86        	mv	a3, t0
   194b2: fd 18        	addi	a7, a7, -1
   194b4: 63 f0 c2 09  	bgeu	t0, t3, 0x19534 <__mdiff+0x176>
   194b8: 9c 42        	lw	a5, 0(a3)
   194ba: 9b 8e 05 00  	sext.w	t4, a1
   194be: 11 06        	addi	a2, a2, 4
   194c0: 33 f7 17 01  	and	a4, a5, a7
   194c4: 2d 9f        	addw	a4, a4, a1
   194c6: 1b 57 07 41  	sraiw	a4, a4, 16
   194ca: 9b d5 07 01  	srliw	a1, a5, 16
   194ce: 2d 9f        	addw	a4, a4, a1
   194d0: bb 87 d7 01  	addw	a5, a5, t4
   194d4: 9b 15 07 01  	slliw	a1, a4, 16
   194d8: b3 f7 17 01  	and	a5, a5, a7
   194dc: cd 8f        	or	a5, a5, a1
   194de: 81 27        	sext.w	a5, a5
   194e0: 91 06        	addi	a3, a3, 4
   194e2: 23 2e f6 fe  	sw	a5, -4(a2)
   194e6: 9b 55 07 41  	sraiw	a1, a4, 16
   194ea: e3 e7 c6 fd  	bltu	a3, t3, 0x194b8 <__mdiff+0xfa>
   194ee: 7d 1e        	addi	t3, t3, -1
   194f0: 33 0e 5e 40  	sub	t3, t3, t0
   194f4: 13 7e ce ff  	andi	t3, t3, -4
   194f8: 33 07 c8 01  	add	a4, a6, t3
   194fc: 91 e7        	bnez	a5, 0x19508 <__mdiff+0x14a>
   194fe: 83 27 c7 ff  	lw	a5, -4(a4)
   19502: 7d 33        	addiw	t1, t1, -1
   19504: 71 17        	addi	a4, a4, -4
   19506: e5 df        	beqz	a5, 0x194fe <__mdiff+0x140>
   19508: e2 60        	ld	ra, 24(sp)
   1950a: 42 64        	ld	s0, 16(sp)
   1950c: 23 2a 65 00  	sw	t1, 20(a0)
   19510: a2 64        	ld	s1, 8(sp)
   19512: 02 69        	ld	s2, 0(sp)
   19514: 05 61        	addi	sp, sp, 32
   19516: 82 80        	ret
   19518: 81 45        	mv	a1, zero
   1951a: ef f0 cf ee  	jal	0x18c06 <_Balloc>
   1951e: 05 c5        	beqz	a0, 0x19546 <__mdiff+0x188>
   19520: e2 60        	ld	ra, 24(sp)
   19522: 42 64        	ld	s0, 16(sp)
   19524: 85 47        	addi	a5, zero, 1
   19526: 5c c9        	sw	a5, 20(a0)
   19528: 23 2c 05 00  	sw	zero, 24(a0)
   1952c: a2 64        	ld	s1, 8(sp)
   1952e: 02 69        	ld	s2, 0(sp)
   19530: 05 61        	addi	sp, sp, 32
   19532: 82 80        	ret
   19534: 01 47        	mv	a4, zero
   19536: 63 74 8f 00  	bgeu	t5, s0, 0x1953e <__mdiff+0x180>
   1953a: 7e 97        	add	a4, a4, t6
   1953c: c1 b7        	j	0x194fc <__mdiff+0x13e>
   1953e: 13 97 2e 00  	slli	a4, t4, 2
   19542: 7e 97        	add	a4, a4, t6
   19544: 65 bf        	j	0x194fc <__mdiff+0x13e>
   19546: b7 26 02 00  	lui	a3, 34
   1954a: 37 25 02 00  	lui	a0, 34
   1954e: 93 86 06 eb  	addi	a3, a3, -336
   19552: 01 46        	mv	a2, zero
   19554: 93 05 20 23  	addi	a1, zero, 562
   19558: 13 05 05 1b  	addi	a0, a0, 432
   1955c: ef 30 e0 29  	jal	0x1c7fa <__assert_func>
   19560: b7 26 02 00  	lui	a3, 34
   19564: 37 25 02 00  	lui	a0, 34
   19568: 93 86 06 eb  	addi	a3, a3, -336
   1956c: 01 46        	mv	a2, zero
   1956e: 93 05 00 24  	addi	a1, zero, 576
   19572: 13 05 05 1b  	addi	a0, a0, 432
   19576: ef 30 40 28  	jal	0x1c7fa <__assert_func>

000000000001957a <__ulp>:
   1957a: d3 07 05 e2  	fmv.x.d	a5, fa0
   1957e: 37 07 f0 7f  	lui	a4, 524032
   19582: 81 46        	mv	a3, zero
   19584: 81 93        	srli	a5, a5, 32
   19586: f9 8f        	and	a5, a5, a4
   19588: 37 07 c0 fc  	lui	a4, 1035264
   1958c: b9 9f        	addw	a5, a5, a4
   1958e: 63 59 f0 00  	blez	a5, 0x195a0 <__ulp+0x26>
   19592: 82 16        	slli	a3, a3, 32
   19594: 81 92        	srli	a3, a3, 32
   19596: 82 17        	slli	a5, a5, 32
   19598: d5 8f        	or	a5, a5, a3
   1959a: 53 85 07 f2  	fmv.d.x	fa0, a5
   1959e: 82 80        	ret
   195a0: 3b 07 f0 40  	negw	a4, a5
   195a4: 9b 56 47 41  	sraiw	a3, a4, 20
   195a8: cd 47        	addi	a5, zero, 19
   195aa: 63 ce d7 00  	blt	a5, a3, 0x195c6 <__ulp+0x4c>
   195ae: b7 07 08 00  	lui	a5, 128
   195b2: bb d7 d7 40  	sraw	a5, a5, a3
   195b6: 81 46        	mv	a3, zero
   195b8: 82 16        	slli	a3, a3, 32
   195ba: 81 92        	srli	a3, a3, 32
   195bc: 82 17        	slli	a5, a5, 32
   195be: d5 8f        	or	a5, a5, a3
   195c0: 53 85 07 f2  	fmv.d.x	fa0, a5
   195c4: 82 80        	ret
   195c6: 1b 87 c6 fe  	addiw	a4, a3, -20
   195ca: f9 47        	addi	a5, zero, 30
   195cc: 85 46        	addi	a3, zero, 1
   195ce: 63 c6 e7 00  	blt	a5, a4, 0x195da <__ulp+0x60>
   195d2: b7 06 00 80  	lui	a3, 524288
   195d6: bb d6 e6 00  	srlw	a3, a3, a4
   195da: 81 47        	mv	a5, zero
   195dc: 82 16        	slli	a3, a3, 32
   195de: 81 92        	srli	a3, a3, 32
   195e0: 82 17        	slli	a5, a5, 32
   195e2: d5 8f        	or	a5, a5, a3
   195e4: 53 85 07 f2  	fmv.d.x	fa0, a5
   195e8: 82 80        	ret

00000000000195ea <__b2d>:
   195ea: 79 71        	addi	sp, sp, -48
   195ec: 26 ec        	sd	s1, 24(sp)
   195ee: 44 49        	lw	s1, 20(a0)
   195f0: 22 f0        	sd	s0, 32(sp)
   195f2: 13 04 85 01  	addi	s0, a0, 24
   195f6: 8a 04        	slli	s1, s1, 2
   195f8: a2 94        	add	s1, s1, s0
   195fa: 4a e8        	sd	s2, 16(sp)
   195fc: 03 a9 c4 ff  	lw	s2, -4(s1)
   19600: 4e e4        	sd	s3, 8(sp)
   19602: 52 e0        	sd	s4, 0(sp)
   19604: 4a 85        	mv	a0, s2
   19606: ae 89        	mv	s3, a1
   19608: 06 f4        	sd	ra, 40(sp)
   1960a: ef f0 1f 84  	jal	0x18e4a <__hi0bits>
   1960e: 93 06 00 02  	addi	a3, zero, 32
   19612: bb 87 a6 40  	subw	a5, a3, a0
   19616: 23 a0 f9 00  	sw	a5, 0(s3)
   1961a: a9 47        	addi	a5, zero, 10
   1961c: 13 8a c4 ff  	addi	s4, s1, -4
   19620: 63 d5 a7 08  	bge	a5, a0, 0x196aa <__b2d+0xc0>
   19624: 9b 07 55 ff  	addiw	a5, a0, -11
   19628: 63 71 44 05  	bgeu	s0, s4, 0x1966a <__b2d+0x80>
   1962c: 03 a7 84 ff  	lw	a4, -8(s1)
   19630: a9 cb        	beqz	a5, 0x19682 <__b2d+0x98>
   19632: 9d 9e        	subw	a3, a3, a5
   19634: 3b 56 d7 00  	srlw	a2, a4, a3
   19638: 3b 19 f9 00  	sllw	s2, s2, a5
   1963c: 33 69 c9 00  	or	s2, s2, a2
   19640: 37 06 f0 3f  	lui	a2, 261888
   19644: 33 69 c9 00  	or	s2, s2, a2
   19648: bb 15 f7 00  	sllw	a1, a4, a5
   1964c: 13 86 84 ff  	addi	a2, s1, -8
   19650: 93 17 09 02  	slli	a5, s2, 32
   19654: 1b 87 05 00  	sext.w	a4, a1
   19658: 63 7a c4 02  	bgeu	s0, a2, 0x1968c <__b2d+0xa2>
   1965c: 03 a7 44 ff  	lw	a4, -12(s1)
   19660: 3b 57 d7 00  	srlw	a4, a4, a3
   19664: 4d 8f        	or	a4, a4, a1
   19666: 01 27        	sext.w	a4, a4
   19668: 15 a0        	j	0x1968c <__b2d+0xa2>
   1966a: 2d 47        	addi	a4, zero, 11
   1966c: 63 0a e5 00  	beq	a0, a4, 0x19680 <__b2d+0x96>
   19670: 37 07 f0 3f  	lui	a4, 261888
   19674: bb 17 f9 00  	sllw	a5, s2, a5
   19678: d9 8f        	or	a5, a5, a4
   1967a: 82 17        	slli	a5, a5, 32
   1967c: 01 47        	mv	a4, zero
   1967e: 39 a0        	j	0x1968c <__b2d+0xa2>
   19680: 01 47        	mv	a4, zero
   19682: b7 07 f0 3f  	lui	a5, 261888
   19686: b3 67 f9 00  	or	a5, s2, a5
   1968a: 82 17        	slli	a5, a5, 32
   1968c: a2 70        	ld	ra, 40(sp)
   1968e: 02 74        	ld	s0, 32(sp)
   19690: 81 93        	srli	a5, a5, 32
   19692: 02 17        	slli	a4, a4, 32
   19694: 01 93        	srli	a4, a4, 32
   19696: 82 17        	slli	a5, a5, 32
   19698: d9 8f        	or	a5, a5, a4
   1969a: e2 64        	ld	s1, 24(sp)
   1969c: 42 69        	ld	s2, 16(sp)
   1969e: a2 69        	ld	s3, 8(sp)
   196a0: 02 6a        	ld	s4, 0(sp)
   196a2: 53 85 07 f2  	fmv.d.x	fa0, a5
   196a6: 45 61        	addi	sp, sp, 48
   196a8: 82 80        	ret
   196aa: ad 46        	addi	a3, zero, 11
   196ac: 89 9e        	subw	a3, a3, a0
   196ae: 37 07 f0 3f  	lui	a4, 261888
   196b2: bb 57 d9 00  	srlw	a5, s2, a3
   196b6: d9 8f        	or	a5, a5, a4
   196b8: 82 17        	slli	a5, a5, 32
   196ba: 01 47        	mv	a4, zero
   196bc: 63 76 44 01  	bgeu	s0, s4, 0x196c8 <__b2d+0xde>
   196c0: 03 a7 84 ff  	lw	a4, -8(s1)
   196c4: 3b 57 d7 00  	srlw	a4, a4, a3
   196c8: 55 25        	addiw	a0, a0, 21
   196ca: 3b 15 a9 00  	sllw	a0, s2, a0
   196ce: 49 8f        	or	a4, a4, a0
   196d0: 01 27        	sext.w	a4, a4
   196d2: 6d bf        	j	0x1968c <__b2d+0xa2>

00000000000196d4 <__d2b>:
   196d4: 39 71        	addi	sp, sp, -64
   196d6: 52 e8        	sd	s4, 16(sp)
   196d8: 2e 8a        	mv	s4, a1
   196da: 85 45        	addi	a1, zero, 1
   196dc: 22 f8        	sd	s0, 48(sp)
   196de: 4a f0        	sd	s2, 32(sp)
   196e0: 06 fc        	sd	ra, 56(sp)
   196e2: 26 f4        	sd	s1, 40(sp)
   196e4: 4e ec        	sd	s3, 24(sp)
   196e6: 53 04 05 e2  	fmv.x.d	s0, fa0
   196ea: 32 89        	mv	s2, a2
   196ec: ef f0 af d1  	jal	0x18c06 <_Balloc>
   196f0: 55 c9        	beqz	a0, 0x197a4 <__d2b+0xd0>
   196f2: 93 57 04 42  	srai	a5, s0, 32
   196f6: 9b d9 47 01  	srliw	s3, a5, 20
   196fa: 93 f9 f9 7f  	andi	s3, s3, 2047
   196fe: b2 17        	slli	a5, a5, 44
   19700: aa 84        	mv	s1, a0
   19702: b1 93        	srli	a5, a5, 44
   19704: 63 85 09 00  	beqz	s3, 0x1970e <__d2b+0x3a>
   19708: 37 07 10 00  	lui	a4, 256
   1970c: d9 8f        	or	a5, a5, a4
   1970e: 3e c6        	sw	a5, 12(sp)
   19710: 01 24        	sext.w	s0, s0
   19712: 29 e0        	bnez	s0, 0x19754 <__d2b+0x80>
   19714: 68 00        	addi	a0, sp, 12
   19716: ef f0 2f f9  	jal	0x18ea8 <__lo0bits>
   1971a: b2 47        	lw	a5, 12(sp)
   1971c: 05 44        	addi	s0, zero, 1
   1971e: c0 c8        	sw	s0, 20(s1)
   19720: 9c cc        	sw	a5, 24(s1)
   19722: 1b 05 05 02  	addiw	a0, a0, 32
   19726: 63 80 09 06  	beqz	s3, 0x19786 <__d2b+0xb2>
   1972a: 9b 89 d9 bc  	addiw	s3, s3, -1075
   1972e: bb 89 a9 00  	addw	s3, s3, a0
   19732: 13 04 50 03  	addi	s0, zero, 53
   19736: 23 20 3a 01  	sw	s3, 0(s4)
   1973a: 3b 05 a4 40  	subw	a0, s0, a0
   1973e: e2 70        	ld	ra, 56(sp)
   19740: 42 74        	ld	s0, 48(sp)
   19742: 23 20 a9 00  	sw	a0, 0(s2)
   19746: e2 69        	ld	s3, 24(sp)
   19748: 02 79        	ld	s2, 32(sp)
   1974a: 42 6a        	ld	s4, 16(sp)
   1974c: 26 85        	mv	a0, s1
   1974e: a2 74        	ld	s1, 40(sp)
   19750: 21 61        	addi	sp, sp, 64
   19752: 82 80        	ret
   19754: 28 00        	addi	a0, sp, 8
   19756: 22 c4        	sw	s0, 8(sp)
   19758: ef f0 0f f5  	jal	0x18ea8 <__lo0bits>
   1975c: b2 47        	lw	a5, 12(sp)
   1975e: 22 47        	lw	a4, 8(sp)
   19760: 19 c9        	beqz	a0, 0x19776 <__d2b+0xa2>
   19762: 93 06 00 02  	addi	a3, zero, 32
   19766: 89 9e        	subw	a3, a3, a0
   19768: bb 96 d7 00  	sllw	a3, a5, a3
   1976c: 55 8f        	or	a4, a4, a3
   1976e: bb d7 a7 00  	srlw	a5, a5, a0
   19772: 01 27        	sext.w	a4, a4
   19774: 3e c6        	sw	a5, 12(sp)
   19776: 33 34 f0 00  	snez	s0, a5
   1977a: 05 04        	addi	s0, s0, 1
   1977c: 98 cc        	sw	a4, 24(s1)
   1977e: dc cc        	sw	a5, 28(s1)
   19780: c0 c8        	sw	s0, 20(s1)
   19782: e3 94 09 fa  	bnez	s3, 0x1972a <__d2b+0x56>
   19786: 1b 05 e5 bc  	addiw	a0, a0, -1074
   1978a: 93 17 24 00  	slli	a5, s0, 2
   1978e: 23 20 aa 00  	sw	a0, 0(s4)
   19792: a6 97        	add	a5, a5, s1
   19794: c8 4b        	lw	a0, 20(a5)
   19796: 1b 14 54 00  	slliw	s0, s0, 5
   1979a: ef f0 0f eb  	jal	0x18e4a <__hi0bits>
   1979e: 3b 05 a4 40  	subw	a0, s0, a0
   197a2: 71 bf        	j	0x1973e <__d2b+0x6a>
   197a4: b7 26 02 00  	lui	a3, 34
   197a8: 37 25 02 00  	lui	a0, 34
   197ac: 93 86 06 eb  	addi	a3, a3, -336
   197b0: 01 46        	mv	a2, zero
   197b2: 93 05 a0 30  	addi	a1, zero, 778
   197b6: 13 05 05 1b  	addi	a0, a0, 432
   197ba: ef 30 00 04  	jal	0x1c7fa <__assert_func>

00000000000197be <__ratio>:
   197be: 79 71        	addi	sp, sp, -48
   197c0: 26 ec        	sd	s1, 24(sp)
   197c2: ae 84        	mv	s1, a1
   197c4: 2c 00        	addi	a1, sp, 8
   197c6: 06 f4        	sd	ra, 40(sp)
   197c8: 22 f0        	sd	s0, 32(sp)
   197ca: 4a e8        	sd	s2, 16(sp)
   197cc: 2a 89        	mv	s2, a0
   197ce: ef f0 df e1  	jal	0x195ea <__b2d>
   197d2: 6c 00        	addi	a1, sp, 12
   197d4: 26 85        	mv	a0, s1
   197d6: 53 04 05 e2  	fmv.x.d	s0, fa0
   197da: ef f0 1f e1  	jal	0x195ea <__b2d>
   197de: d8 48        	lw	a4, 20(s1)
   197e0: 83 27 49 01  	lw	a5, 20(s2)
   197e4: a2 46        	lw	a3, 8(sp)
   197e6: 32 46        	lw	a2, 12(sp)
   197e8: 99 9f        	subw	a5, a5, a4
   197ea: 9b 97 57 00  	slliw	a5, a5, 5
   197ee: 91 9e        	subw	a3, a3, a2
   197f0: bb 85 d7 00  	addw	a1, a5, a3
   197f4: 53 07 05 e2  	fmv.x.d	a4, fa0
   197f8: 63 59 b0 02  	blez	a1, 0x1982a <__ratio+0x6c>
   197fc: 13 56 04 02  	srli	a2, s0, 32
   19800: 9b 97 45 01  	slliw	a5, a1, 20
   19804: 93 16 04 02  	slli	a3, s0, 32
   19808: b1 9f        	addw	a5, a5, a2
   1980a: 82 17        	slli	a5, a5, 32
   1980c: 81 92        	srli	a3, a3, 32
   1980e: 33 e4 f6 00  	or	s0, a3, a5
   19812: d3 07 04 f2  	fmv.d.x	fa5, s0
   19816: a2 70        	ld	ra, 40(sp)
   19818: 02 74        	ld	s0, 32(sp)
   1981a: 53 07 07 f2  	fmv.d.x	fa4, a4
   1981e: e2 64        	ld	s1, 24(sp)
   19820: 42 69        	ld	s2, 16(sp)
   19822: 53 f5 e7 1a  	fdiv.d	fa0, fa5, fa4
   19826: 45 61        	addi	sp, sp, 48
   19828: 82 80        	ret
   1982a: 3a 86        	mv	a2, a4
   1982c: 9b 97 45 01  	slliw	a5, a1, 20
   19830: 01 93        	srli	a4, a4, 32
   19832: 1d 9f        	subw	a4, a4, a5
   19834: 02 16        	slli	a2, a2, 32
   19836: 02 17        	slli	a4, a4, 32
   19838: 01 92        	srli	a2, a2, 32
   1983a: 51 8f        	or	a4, a4, a2
   1983c: d9 bf        	j	0x19812 <__ratio+0x54>

000000000001983e <_mprec_log10>:
   1983e: dd 47        	addi	a5, zero, 23
   19840: 63 db a7 00  	bge	a5, a0, 0x19856 <_mprec_log10+0x18>
   19844: 07 b5 01 81  	fld	fa0, -2032(gp)
   19848: 87 b7 81 81  	fld	fa5, -2024(gp)
   1984c: 7d 35        	addiw	a0, a0, -1
   1984e: 53 75 f5 12  	fmul.d	fa0, fa0, fa5
   19852: 6d fd        	bnez	a0, 0x1984c <_mprec_log10+0xe>
   19854: 82 80        	ret
   19856: b7 27 02 00  	lui	a5, 34
   1985a: 0e 05        	slli	a0, a0, 3
   1985c: 93 87 07 27  	addi	a5, a5, 624
   19860: aa 97        	add	a5, a5, a0
   19862: 88 23        	fld	fa0, 0(a5)
   19864: 82 80        	ret

0000000000019866 <__copybits>:
   19866: 54 4a        	lw	a3, 20(a2)
   19868: 9b 87 f5 ff  	addiw	a5, a1, -1
   1986c: 9b d7 57 40  	sraiw	a5, a5, 5
   19870: 85 27        	addiw	a5, a5, 1
   19872: 13 07 86 01  	addi	a4, a2, 24
   19876: 93 95 26 00  	slli	a1, a3, 2
   1987a: 8a 07        	slli	a5, a5, 2
   1987c: ba 95        	add	a1, a1, a4
   1987e: aa 97        	add	a5, a5, a0
   19880: 63 71 b7 02  	bgeu	a4, a1, 0x198a2 <__copybits+0x3c>
   19884: aa 86        	mv	a3, a0
   19886: 03 28 07 00  	lw	a6, 0(a4)
   1988a: 11 07        	addi	a4, a4, 4
   1988c: 91 06        	addi	a3, a3, 4
   1988e: 23 ae 06 ff  	sw	a6, -4(a3)
   19892: e3 6a b7 fe  	bltu	a4, a1, 0x19886 <__copybits+0x20>
   19896: 33 87 c5 40  	sub	a4, a1, a2
   1989a: 1d 17        	addi	a4, a4, -25
   1989c: 71 9b        	andi	a4, a4, -4
   1989e: 11 07        	addi	a4, a4, 4
   198a0: 3a 95        	add	a0, a0, a4
   198a2: 63 73 f5 02  	bgeu	a0, a5, 0x198c8 <__copybits+0x62>
   198a6: fd 17        	addi	a5, a5, -1
   198a8: 89 8f        	sub	a5, a5, a0
   198aa: 89 83        	srli	a5, a5, 2
   198ac: 85 07        	addi	a5, a5, 1
   198ae: f3 26 20 c2  	csrr	a3, vlenb
   198b2: 57 77 00 05  	<unknown>
   198b6: 57 3c 00 5e  	<unknown>
   198ba: 57 f7 07 05  	<unknown>
   198be: 27 6c 05 02  	<unknown>
   198c2: 99 8f        	sub	a5, a5, a4
   198c4: 36 95        	add	a0, a0, a3
   198c6: f5 fb        	bnez	a5, 0x198ba <__copybits+0x54>
   198c8: 82 80        	ret

00000000000198ca <__any_on>:
   198ca: 58 49        	lw	a4, 20(a0)
   198cc: 13 d6 55 40  	srai	a2, a1, 5
   198d0: 93 06 85 01  	addi	a3, a0, 24
   198d4: 63 5d c7 00  	bge	a4, a2, 0x198ee <__any_on+0x24>
   198d8: 0a 07        	slli	a4, a4, 2
   198da: b3 87 e6 00  	add	a5, a3, a4
   198de: 63 f8 f6 02  	bgeu	a3, a5, 0x1990e <__any_on+0x44>
   198e2: 03 a7 c7 ff  	lw	a4, -4(a5)
   198e6: f1 17        	addi	a5, a5, -4
   198e8: 7d db        	beqz	a4, 0x198de <__any_on+0x14>
   198ea: 05 45        	addi	a0, zero, 1
   198ec: 82 80        	ret
   198ee: 93 17 26 00  	slli	a5, a2, 2
   198f2: b6 97        	add	a5, a5, a3
   198f4: e3 55 e6 fe  	bge	a2, a4, 0x198de <__any_on+0x14>
   198f8: fd 89        	andi	a1, a1, 31
   198fa: f5 d1        	beqz	a1, 0x198de <__any_on+0x14>
   198fc: 90 43        	lw	a2, 0(a5)
   198fe: 05 45        	addi	a0, zero, 1
   19900: 3b 57 b6 00  	srlw	a4, a2, a1
   19904: 3b 17 b7 00  	sllw	a4, a4, a1
   19908: e3 0b c7 fc  	beq	a4, a2, 0x198de <__any_on+0x14>
   1990c: 82 80        	ret
   1990e: 01 45        	mv	a0, zero
   19910: 82 80        	ret

0000000000019912 <_sbrk_r>:
   19912: 01 11        	addi	sp, sp, -32
   19914: 22 e8        	sd	s0, 16(sp)
   19916: 26 e4        	sd	s1, 8(sp)
   19918: 2a 84        	mv	s0, a0
   1991a: 2e 85        	mv	a0, a1
   1991c: 06 ec        	sd	ra, 24(sp)
   1991e: 23 ac 01 96  	sw	zero, -1672(gp)
   19922: ef 50 40 12  	jal	0x1ea46 <_sbrk>
   19926: fd 57        	addi	a5, zero, -1
   19928: 63 07 f5 00  	beq	a0, a5, 0x19936 <_sbrk_r+0x24>
   1992c: e2 60        	ld	ra, 24(sp)
   1992e: 42 64        	ld	s0, 16(sp)
   19930: a2 64        	ld	s1, 8(sp)
   19932: 05 61        	addi	sp, sp, 32
   19934: 82 80        	ret
   19936: 83 a7 81 97  	lw	a5, -1672(gp)
   1993a: ed db        	beqz	a5, 0x1992c <_sbrk_r+0x1a>
   1993c: e2 60        	ld	ra, 24(sp)
   1993e: 1c c0        	sw	a5, 0(s0)
   19940: 42 64        	ld	s0, 16(sp)
   19942: a2 64        	ld	s1, 8(sp)
   19944: 05 61        	addi	sp, sp, 32
   19946: 82 80        	ret

0000000000019948 <frexp>:
   19948: d3 07 05 e2  	fmv.x.d	a5, fa0
   1994c: b7 05 00 80  	lui	a1, 524288
   19950: 93 c5 f5 ff  	not	a1, a1
   19954: 93 d6 07 42  	srai	a3, a5, 32
   19958: 33 f7 b6 00  	and	a4, a3, a1
   1995c: 23 20 05 00  	sw	zero, 0(a0)
   19960: 37 08 f0 7f  	lui	a6, 524032
   19964: 63 7f 07 03  	bgeu	a4, a6, 0x199a2 <frexp+0x5a>
   19968: 33 e6 e7 00  	or	a2, a5, a4
   1996c: 01 26        	sext.w	a2, a2
   1996e: 15 ca        	beqz	a2, 0x199a2 <frexp+0x5a>
   19970: 37 06 f0 7f  	lui	a2, 524032
   19974: 75 8e        	and	a2, a2, a3
   19976: 0d ca        	beqz	a2, 0x199a8 <frexp+0x60>
   19978: 81 26        	sext.w	a3, a3
   1997a: 01 27        	sext.w	a4, a4
   1997c: 01 48        	mv	a6, zero
   1997e: 37 06 10 80  	lui	a2, 524544
   19982: 7d 16        	addi	a2, a2, -1
   19984: f1 8e        	and	a3, a3, a2
   19986: 1b 57 47 41  	sraiw	a4, a4, 20
   1998a: 37 06 e0 3f  	lui	a2, 261632
   1998e: 1b 07 27 c0  	addiw	a4, a4, -1022
   19992: d1 8e        	or	a3, a3, a2
   19994: 82 17        	slli	a5, a5, 32
   19996: 3b 07 07 01  	addw	a4, a4, a6
   1999a: 82 16        	slli	a3, a3, 32
   1999c: 81 93        	srli	a5, a5, 32
   1999e: 18 c1        	sw	a4, 0(a0)
   199a0: d5 8f        	or	a5, a5, a3
   199a2: 53 85 07 f2  	fmv.d.x	fa0, a5
   199a6: 82 80        	ret
   199a8: 87 b7 01 82  	fld	fa5, -2016(gp)
   199ac: 13 08 a0 fc  	addi	a6, zero, -54
   199b0: d3 77 f5 12  	fmul.d	fa5, fa0, fa5
   199b4: d3 87 07 e2  	fmv.x.d	a5, fa5
   199b8: 13 d7 07 42  	srai	a4, a5, 32
   199bc: ba 86        	mv	a3, a4
   199be: 6d 8f        	and	a4, a4, a1
   199c0: 7d bf        	j	0x1997e <frexp+0x36>

00000000000199c2 <_sprintf_r>:
   199c2: 11 71        	addi	sp, sp, -256
   199c4: 93 0e 81 0d  	addi	t4, sp, 216
   199c8: be f5        	sd	a5, 232(sp)
   199ca: 37 03 00 80  	lui	t1, 524288
   199ce: c1 77        	lui	a5, 1048560
   199d0: 2e 8e        	mv	t3, a1
   199d2: 13 43 f3 ff  	not	t1, t1
   199d6: b6 ed        	sd	a3, 216(sp)
   199d8: 93 87 87 20  	addi	a5, a5, 520
   199dc: 0c 08        	addi	a1, sp, 16
   199de: f6 86        	mv	a3, t4
   199e0: 86 e5        	sd	ra, 200(sp)
   199e2: 3e d0        	sw	a5, 32(sp)
   199e4: ba f1        	sd	a4, 224(sp)
   199e6: c2 f9        	sd	a6, 240(sp)
   199e8: c6 fd        	sd	a7, 248(sp)
   199ea: 72 e8        	sd	t3, 16(sp)
   199ec: 72 f4        	sd	t3, 40(sp)
   199ee: 1a d8        	sw	t1, 48(sp)
   199f0: 1a ce        	sw	t1, 28(sp)
   199f2: 76 e4        	sd	t4, 8(sp)
   199f4: ef 00 80 3e  	jal	0x19ddc <_svfprintf_r>
   199f8: c2 67        	ld	a5, 16(sp)
   199fa: 23 80 07 00  	sb	zero, 0(a5)
   199fe: ae 60        	ld	ra, 200(sp)
   19a00: 11 61        	addi	sp, sp, 256
   19a02: 82 80        	ret

0000000000019a04 <sprintf>:
   19a04: 2a 8e        	mv	t3, a0
   19a06: 11 71        	addi	sp, sp, -256
   19a08: 03 b5 81 93  	ld	a0, -1736(gp)
   19a0c: 93 0e 01 0d  	addi	t4, sp, 208
   19a10: be f5        	sd	a5, 232(sp)
   19a12: 37 03 00 80  	lui	t1, 524288
   19a16: c1 77        	lui	a5, 1048560
   19a18: 13 43 f3 ff  	not	t1, t1
   19a1c: b2 e9        	sd	a2, 208(sp)
   19a1e: b6 ed        	sd	a3, 216(sp)
   19a20: 93 87 87 20  	addi	a5, a5, 520
   19a24: 2e 86        	mv	a2, a1
   19a26: f6 86        	mv	a3, t4
   19a28: 0c 08        	addi	a1, sp, 16
   19a2a: 86 e5        	sd	ra, 200(sp)
   19a2c: 3e d0        	sw	a5, 32(sp)
   19a2e: ba f1        	sd	a4, 224(sp)
   19a30: c2 f9        	sd	a6, 240(sp)
   19a32: c6 fd        	sd	a7, 248(sp)
   19a34: 72 e8        	sd	t3, 16(sp)
   19a36: 72 f4        	sd	t3, 40(sp)
   19a38: 1a d8        	sw	t1, 48(sp)
   19a3a: 1a ce        	sw	t1, 28(sp)
   19a3c: 76 e4        	sd	t4, 8(sp)
   19a3e: ef 00 e0 39  	jal	0x19ddc <_svfprintf_r>
   19a42: c2 67        	ld	a5, 16(sp)
   19a44: 23 80 07 00  	sb	zero, 0(a5)
   19a48: ae 60        	ld	ra, 200(sp)
   19a4a: 11 61        	addi	sp, sp, 256
   19a4c: 82 80        	ret

0000000000019a4e <__sread>:
   19a4e: 41 11        	addi	sp, sp, -16
   19a50: 22 e0        	sd	s0, 0(sp)
   19a52: 2e 84        	mv	s0, a1
   19a54: 83 95 25 01  	lh	a1, 18(a1)
   19a58: 06 e4        	sd	ra, 8(sp)
   19a5a: ef 30 80 6b  	jal	0x1d112 <_read_r>
   19a5e: 63 49 05 00  	bltz	a0, 0x19a70 <__sread+0x22>
   19a62: 5c 68        	ld	a5, 144(s0)
   19a64: a2 60        	ld	ra, 8(sp)
   19a66: aa 97        	add	a5, a5, a0
   19a68: 5c e8        	sd	a5, 144(s0)
   19a6a: 02 64        	ld	s0, 0(sp)
   19a6c: 41 01        	addi	sp, sp, 16
   19a6e: 82 80        	ret
   19a70: 83 57 04 01  	lhu	a5, 16(s0)
   19a74: 7d 77        	lui	a4, 1048575
   19a76: 7d 17        	addi	a4, a4, -1
   19a78: f9 8f        	and	a5, a5, a4
   19a7a: a2 60        	ld	ra, 8(sp)
   19a7c: 23 18 f4 00  	sh	a5, 16(s0)
   19a80: 02 64        	ld	s0, 0(sp)
   19a82: 41 01        	addi	sp, sp, 16
   19a84: 82 80        	ret

0000000000019a86 <__seofread>:
   19a86: 01 45        	mv	a0, zero
   19a88: 82 80        	ret

0000000000019a8a <__swrite>:
   19a8a: 83 97 05 01  	lh	a5, 16(a1)
   19a8e: 79 71        	addi	sp, sp, -48
   19a90: 22 f0        	sd	s0, 32(sp)
   19a92: 26 ec        	sd	s1, 24(sp)
   19a94: 4a e8        	sd	s2, 16(sp)
   19a96: 4e e4        	sd	s3, 8(sp)
   19a98: 06 f4        	sd	ra, 40(sp)
   19a9a: 13 f7 07 10  	andi	a4, a5, 256
   19a9e: 2e 84        	mv	s0, a1
   19aa0: aa 84        	mv	s1, a0
   19aa2: 32 89        	mv	s2, a2
   19aa4: b6 89        	mv	s3, a3
   19aa6: 1d e3        	bnez	a4, 0x19acc <__swrite+0x42>
   19aa8: 7d 77        	lui	a4, 1048575
   19aaa: 7d 17        	addi	a4, a4, -1
   19aac: f9 8f        	and	a5, a5, a4
   19aae: 83 15 24 01  	lh	a1, 18(s0)
   19ab2: 23 18 f4 00  	sh	a5, 16(s0)
   19ab6: 02 74        	ld	s0, 32(sp)
   19ab8: a2 70        	ld	ra, 40(sp)
   19aba: ce 86        	mv	a3, s3
   19abc: 4a 86        	mv	a2, s2
   19abe: a2 69        	ld	s3, 8(sp)
   19ac0: 42 69        	ld	s2, 16(sp)
   19ac2: 26 85        	mv	a0, s1
   19ac4: e2 64        	ld	s1, 24(sp)
   19ac6: 45 61        	addi	sp, sp, 48
   19ac8: 6f 20 70 4f  	j	0x1c7be <_write_r>
   19acc: 83 95 25 01  	lh	a1, 18(a1)
   19ad0: 89 46        	addi	a3, zero, 2
   19ad2: 01 46        	mv	a2, zero
   19ad4: ef 30 40 42  	jal	0x1cef8 <_lseek_r>
   19ad8: 83 17 04 01  	lh	a5, 16(s0)
   19adc: f1 b7        	j	0x19aa8 <__swrite+0x1e>

0000000000019ade <__sseek>:
   19ade: 41 11        	addi	sp, sp, -16
   19ae0: 22 e0        	sd	s0, 0(sp)
   19ae2: 2e 84        	mv	s0, a1
   19ae4: 83 95 25 01  	lh	a1, 18(a1)
   19ae8: 06 e4        	sd	ra, 8(sp)
   19aea: ef 30 e0 40  	jal	0x1cef8 <_lseek_r>
   19aee: fd 57        	addi	a5, zero, -1
   19af0: 63 01 f5 02  	beq	a0, a5, 0x19b12 <__sseek+0x34>
   19af4: 83 57 04 01  	lhu	a5, 16(s0)
   19af8: 05 67        	lui	a4, 1
   19afa: a2 60        	ld	ra, 8(sp)
   19afc: d9 8f        	or	a5, a5, a4
   19afe: 9b 97 07 01  	slliw	a5, a5, 16
   19b02: 9b d7 07 41  	sraiw	a5, a5, 16
   19b06: 48 e8        	sd	a0, 144(s0)
   19b08: 23 18 f4 00  	sh	a5, 16(s0)
   19b0c: 02 64        	ld	s0, 0(sp)
   19b0e: 41 01        	addi	sp, sp, 16
   19b10: 82 80        	ret
   19b12: 83 57 04 01  	lhu	a5, 16(s0)
   19b16: 7d 77        	lui	a4, 1048575
   19b18: 7d 17        	addi	a4, a4, -1
   19b1a: f9 8f        	and	a5, a5, a4
   19b1c: 9b 97 07 01  	slliw	a5, a5, 16
   19b20: 9b d7 07 41  	sraiw	a5, a5, 16
   19b24: a2 60        	ld	ra, 8(sp)
   19b26: 23 18 f4 00  	sh	a5, 16(s0)
   19b2a: 02 64        	ld	s0, 0(sp)
   19b2c: 41 01        	addi	sp, sp, 16
   19b2e: 82 80        	ret

0000000000019b30 <__sclose>:
   19b30: 83 95 25 01  	lh	a1, 18(a1)
   19b34: 6f 20 d0 5c  	j	0x1c900 <_close_r>

0000000000019b38 <strcmp>:
   19b38: 33 67 b5 00  	or	a4, a0, a1
   19b3c: fd 53        	addi	t2, zero, -1
   19b3e: 1d 8b        	andi	a4, a4, 7
   19b40: 4d eb        	bnez	a4, 0x19bf2 <strcmp+0xba>
   19b42: 97 a7 00 00  	auipc	a5, 10
   19b46: 83 b7 67 60  	ld	a5, 1542(a5)
   19b4a: 10 61        	ld	a2, 0(a0)
   19b4c: 94 61        	ld	a3, 0(a1)
   19b4e: b3 72 f6 00  	and	t0, a2, a5
   19b52: 33 63 f6 00  	or	t1, a2, a5
   19b56: be 92        	add	t0, t0, a5
   19b58: b3 e2 62 00  	or	t0, t0, t1
   19b5c: 63 99 72 0a  	bne	t0, t2, 0x19c0e <strcmp+0xd6>
   19b60: 63 1e d6 02  	bne	a2, a3, 0x19b9c <strcmp+0x64>
   19b64: 10 65        	ld	a2, 8(a0)
   19b66: 94 65        	ld	a3, 8(a1)
   19b68: b3 72 f6 00  	and	t0, a2, a5
   19b6c: 33 63 f6 00  	or	t1, a2, a5
   19b70: be 92        	add	t0, t0, a5
   19b72: b3 e2 62 00  	or	t0, t0, t1
   19b76: 63 9a 72 08  	bne	t0, t2, 0x19c0a <strcmp+0xd2>
   19b7a: 63 11 d6 02  	bne	a2, a3, 0x19b9c <strcmp+0x64>
   19b7e: 10 69        	ld	a2, 16(a0)
   19b80: 94 69        	ld	a3, 16(a1)
   19b82: b3 72 f6 00  	and	t0, a2, a5
   19b86: 33 63 f6 00  	or	t1, a2, a5
   19b8a: be 92        	add	t0, t0, a5
   19b8c: b3 e2 62 00  	or	t0, t0, t1
   19b90: 63 93 72 08  	bne	t0, t2, 0x19c16 <strcmp+0xde>
   19b94: 61 05        	addi	a0, a0, 24
   19b96: e1 05        	addi	a1, a1, 24
   19b98: e3 09 d6 fa  	beq	a2, a3, 0x19b4a <strcmp+0x12>
   19b9c: 13 17 06 03  	slli	a4, a2, 48
   19ba0: 93 97 06 03  	slli	a5, a3, 48
   19ba4: 63 18 f7 02  	bne	a4, a5, 0x19bd4 <strcmp+0x9c>
   19ba8: 13 17 06 02  	slli	a4, a2, 32
   19bac: 93 97 06 02  	slli	a5, a3, 32
   19bb0: 63 12 f7 02  	bne	a4, a5, 0x19bd4 <strcmp+0x9c>
   19bb4: 13 17 06 01  	slli	a4, a2, 16
   19bb8: 93 97 06 01  	slli	a5, a3, 16
   19bbc: 63 1c f7 00  	bne	a4, a5, 0x19bd4 <strcmp+0x9c>
   19bc0: 13 57 06 03  	srli	a4, a2, 48
   19bc4: 93 d7 06 03  	srli	a5, a3, 48
   19bc8: 33 05 f7 40  	sub	a0, a4, a5
   19bcc: 93 75 f5 0f  	andi	a1, a0, 255
   19bd0: 91 e9        	bnez	a1, 0x19be4 <strcmp+0xac>
   19bd2: 82 80        	ret
   19bd4: 41 93        	srli	a4, a4, 48
   19bd6: c1 93        	srli	a5, a5, 48
   19bd8: 33 05 f7 40  	sub	a0, a4, a5
   19bdc: 93 75 f5 0f  	andi	a1, a0, 255
   19be0: 91 e1        	bnez	a1, 0x19be4 <strcmp+0xac>
   19be2: 82 80        	ret
   19be4: 13 77 f7 0f  	andi	a4, a4, 255
   19be8: 93 f7 f7 0f  	andi	a5, a5, 255
   19bec: 33 05 f7 40  	sub	a0, a4, a5
   19bf0: 82 80        	ret
   19bf2: 03 46 05 00  	lbu	a2, 0(a0)
   19bf6: 83 c6 05 00  	lbu	a3, 0(a1)
   19bfa: 05 05        	addi	a0, a0, 1
   19bfc: 85 05        	addi	a1, a1, 1
   19bfe: 63 13 d6 00  	bne	a2, a3, 0x19c04 <strcmp+0xcc>
   19c02: 65 fa        	bnez	a2, 0x19bf2 <strcmp+0xba>
   19c04: 33 05 d6 40  	sub	a0, a2, a3
   19c08: 82 80        	ret
   19c0a: 21 05        	addi	a0, a0, 8
   19c0c: a1 05        	addi	a1, a1, 8
   19c0e: e3 12 d6 fe  	bne	a2, a3, 0x19bf2 <strcmp+0xba>
   19c12: 01 45        	mv	a0, zero
   19c14: 82 80        	ret
   19c16: 41 05        	addi	a0, a0, 16
   19c18: c1 05        	addi	a1, a1, 16
   19c1a: e3 1c d6 fc  	bne	a2, a3, 0x19bf2 <strcmp+0xba>
   19c1e: 01 45        	mv	a0, zero
   19c20: 82 80        	ret

0000000000019c22 <strcpy>:
   19c22: b3 67 b5 00  	or	a5, a0, a1
   19c26: 9d 8b        	andi	a5, a5, 7
   19c28: ad ef        	bnez	a5, 0x19ca2 <strcpy+0x80>
   19c2a: 83 b6 81 82  	ld	a3, -2008(gp)
   19c2e: 98 61        	ld	a4, 0(a1)
   19c30: 7d 56        	addi	a2, zero, -1
   19c32: b3 77 d7 00  	and	a5, a4, a3
   19c36: b6 97        	add	a5, a5, a3
   19c38: d9 8f        	or	a5, a5, a4
   19c3a: d5 8f        	or	a5, a5, a3
   19c3c: 63 9c c7 06  	bne	a5, a2, 0x19cb4 <strcpy+0x92>
   19c40: 2a 86        	mv	a2, a0
   19c42: 7d 58        	addi	a6, zero, -1
   19c44: 18 e2        	sd	a4, 0(a2)
   19c46: 98 65        	ld	a4, 8(a1)
   19c48: a1 05        	addi	a1, a1, 8
   19c4a: 21 06        	addi	a2, a2, 8
   19c4c: b3 77 d7 00  	and	a5, a4, a3
   19c50: b6 97        	add	a5, a5, a3
   19c52: d9 8f        	or	a5, a5, a4
   19c54: d5 8f        	or	a5, a5, a3
   19c56: e3 87 07 ff  	beq	a5, a6, 0x19c44 <strcpy+0x22>
   19c5a: 83 c7 05 00  	lbu	a5, 0(a1)
   19c5e: 03 c7 15 00  	lbu	a4, 1(a1)
   19c62: 83 c6 25 00  	lbu	a3, 2(a1)
   19c66: 23 00 f6 00  	sb	a5, 0(a2)
   19c6a: 9d cb        	beqz	a5, 0x19ca0 <strcpy+0x7e>
   19c6c: a3 00 e6 00  	sb	a4, 1(a2)
   19c70: 05 cb        	beqz	a4, 0x19ca0 <strcpy+0x7e>
   19c72: 83 c7 35 00  	lbu	a5, 3(a1)
   19c76: 23 01 d6 00  	sb	a3, 2(a2)
   19c7a: 9d c2        	beqz	a3, 0x19ca0 <strcpy+0x7e>
   19c7c: 03 c7 45 00  	lbu	a4, 4(a1)
   19c80: a3 01 f6 00  	sb	a5, 3(a2)
   19c84: 91 cf        	beqz	a5, 0x19ca0 <strcpy+0x7e>
   19c86: 83 c7 55 00  	lbu	a5, 5(a1)
   19c8a: 23 02 e6 00  	sb	a4, 4(a2)
   19c8e: 09 cb        	beqz	a4, 0x19ca0 <strcpy+0x7e>
   19c90: 03 c7 65 00  	lbu	a4, 6(a1)
   19c94: a3 02 f6 00  	sb	a5, 5(a2)
   19c98: 81 c7        	beqz	a5, 0x19ca0 <strcpy+0x7e>
   19c9a: 23 03 e6 00  	sb	a4, 6(a2)
   19c9e: 09 ef        	bnez	a4, 0x19cb8 <strcpy+0x96>
   19ca0: 82 80        	ret
   19ca2: aa 87        	mv	a5, a0
   19ca4: 03 c7 05 00  	lbu	a4, 0(a1)
   19ca8: 85 07        	addi	a5, a5, 1
   19caa: 85 05        	addi	a1, a1, 1
   19cac: a3 8f e7 fe  	sb	a4, -1(a5)
   19cb0: 75 fb        	bnez	a4, 0x19ca4 <strcpy+0x82>
   19cb2: 82 80        	ret
   19cb4: 2a 86        	mv	a2, a0
   19cb6: 55 b7        	j	0x19c5a <strcpy+0x38>
   19cb8: a3 03 06 00  	sb	zero, 7(a2)
   19cbc: 82 80        	ret

0000000000019cbe <strlen>:
   19cbe: 93 77 75 00  	andi	a5, a0, 7
   19cc2: 2a 87        	mv	a4, a0
   19cc4: a9 eb        	bnez	a5, 0x19d16 <strlen+0x58>
   19cc6: 83 b6 81 82  	ld	a3, -2008(gp)
   19cca: fd 55        	addi	a1, zero, -1
   19ccc: 10 63        	ld	a2, 0(a4)
   19cce: 21 07        	addi	a4, a4, 8
   19cd0: b3 77 d6 00  	and	a5, a2, a3
   19cd4: b6 97        	add	a5, a5, a3
   19cd6: d1 8f        	or	a5, a5, a2
   19cd8: d5 8f        	or	a5, a5, a3
   19cda: e3 89 b7 fe  	beq	a5, a1, 0x19ccc <strlen+0xe>
   19cde: 83 46 87 ff  	lbu	a3, -8(a4)
   19ce2: b3 07 a7 40  	sub	a5, a4, a0
   19ce6: a9 c6        	beqz	a3, 0x19d30 <strlen+0x72>
   19ce8: 83 46 97 ff  	lbu	a3, -7(a4)
   19cec: 9d ce        	beqz	a3, 0x19d2a <strlen+0x6c>
   19cee: 83 46 a7 ff  	lbu	a3, -6(a4)
   19cf2: a9 c6        	beqz	a3, 0x19d3c <strlen+0x7e>
   19cf4: 83 46 b7 ff  	lbu	a3, -5(a4)
   19cf8: 9d ce        	beqz	a3, 0x19d36 <strlen+0x78>
   19cfa: 83 46 c7 ff  	lbu	a3, -4(a4)
   19cfe: b1 c2        	beqz	a3, 0x19d42 <strlen+0x84>
   19d00: 83 46 d7 ff  	lbu	a3, -3(a4)
   19d04: b1 c2        	beqz	a3, 0x19d48 <strlen+0x8a>
   19d06: 03 45 e7 ff  	lbu	a0, -2(a4)
   19d0a: 33 35 a0 00  	snez	a0, a0
   19d0e: 3e 95        	add	a0, a0, a5
   19d10: 79 15        	addi	a0, a0, -2
   19d12: 82 80        	ret
   19d14: cd da        	beqz	a3, 0x19cc6 <strlen+0x8>
   19d16: 83 47 07 00  	lbu	a5, 0(a4)
   19d1a: 05 07        	addi	a4, a4, 1
   19d1c: 93 76 77 00  	andi	a3, a4, 7
   19d20: f5 fb        	bnez	a5, 0x19d14 <strlen+0x56>
   19d22: 09 8f        	sub	a4, a4, a0
   19d24: 13 05 f7 ff  	addi	a0, a4, -1
   19d28: 82 80        	ret
   19d2a: 13 85 97 ff  	addi	a0, a5, -7
   19d2e: 82 80        	ret
   19d30: 13 85 87 ff  	addi	a0, a5, -8
   19d34: 82 80        	ret
   19d36: 13 85 b7 ff  	addi	a0, a5, -5
   19d3a: 82 80        	ret
   19d3c: 13 85 a7 ff  	addi	a0, a5, -6
   19d40: 82 80        	ret
   19d42: 13 85 c7 ff  	addi	a0, a5, -4
   19d46: 82 80        	ret
   19d48: 13 85 d7 ff  	addi	a0, a5, -3
   19d4c: 82 80        	ret

0000000000019d4e <strncpy>:
   19d4e: b3 67 b5 00  	or	a5, a0, a1
   19d52: 9d 8b        	andi	a5, a5, 7
   19d54: b9 e3        	bnez	a5, 0x19d9a <strncpy+0x4c>
   19d56: 9d 47        	addi	a5, zero, 7
   19d58: 63 f1 c7 04  	bgeu	a5, a2, 0x19d9a <strncpy+0x4c>
   19d5c: b7 47 02 00  	lui	a5, 36
   19d60: 03 b3 07 12  	ld	t1, 288(a5)
   19d64: b7 47 02 00  	lui	a5, 36
   19d68: 83 b8 87 12  	ld	a7, 296(a5)
   19d6c: 1d 4e        	addi	t3, zero, 7
   19d6e: aa 87        	mv	a5, a0
   19d70: 39 a0        	j	0x19d7e <strncpy+0x30>
   19d72: 94 e3        	sd	a3, 0(a5)
   19d74: 61 16        	addi	a2, a2, -8
   19d76: a1 07        	addi	a5, a5, 8
   19d78: a1 05        	addi	a1, a1, 8
   19d7a: 63 78 ce 02  	bgeu	t3, a2, 0x19daa <strncpy+0x5c>
   19d7e: 94 61        	ld	a3, 0(a1)
   19d80: 33 87 66 00  	add	a4, a3, t1
   19d84: 13 c8 f6 ff  	not	a6, a3
   19d88: 33 77 07 01  	and	a4, a4, a6
   19d8c: 33 77 17 01  	and	a4, a4, a7
   19d90: 6d d3        	beqz	a4, 0x19d72 <strncpy+0x24>
   19d92: 85 05        	addi	a1, a1, 1
   19d94: 85 07        	addi	a5, a5, 1
   19d96: 01 e6        	bnez	a2, 0x19d9e <strncpy+0x50>
   19d98: 82 80        	ret
   19d9a: aa 87        	mv	a5, a0
   19d9c: 39 a0        	j	0x19daa <strncpy+0x5c>
   19d9e: 03 c7 f5 ff  	lbu	a4, -1(a1)
   19da2: 7d 16        	addi	a2, a2, -1
   19da4: a3 8f e7 fe  	sb	a4, -1(a5)
   19da8: 11 cb        	beqz	a4, 0x19dbc <strncpy+0x6e>
   19daa: 85 05        	addi	a1, a1, 1
   19dac: 85 07        	addi	a5, a5, 1
   19dae: 6d d6        	beqz	a2, 0x19d98 <strncpy+0x4a>
   19db0: 03 c7 f5 ff  	lbu	a4, -1(a1)
   19db4: 7d 16        	addi	a2, a2, -1
   19db6: a3 8f e7 fe  	sb	a4, -1(a5)
   19dba: 65 fb        	bnez	a4, 0x19daa <strncpy+0x5c>
   19dbc: 57 77 00 04  	<unknown>
   19dc0: f3 26 20 c2  	csrr	a3, vlenb
   19dc4: 57 3c 00 5e  	<unknown>
   19dc8: 09 ca        	beqz	a2, 0x19dda <strncpy+0x8c>
   19dca: 57 77 06 04  	<unknown>
   19dce: 27 8c 07 02  	<unknown>
   19dd2: 19 8e        	sub	a2, a2, a4
   19dd4: b6 97        	add	a5, a5, a3
   19dd6: 75 fa        	bnez	a2, 0x19dca <strncpy+0x7c>
   19dd8: 82 80        	ret
   19dda: 82 80        	ret

0000000000019ddc <_svfprintf_r>:
   19ddc: 13 01 01 d9  	addi	sp, sp, -624
   19de0: 23 34 11 26  	sd	ra, 616(sp)
   19de4: 23 30 81 26  	sd	s0, 608(sp)
   19de8: 23 38 21 25  	sd	s2, 592(sp)
   19dec: 2e 84        	mv	s0, a1
   19dee: 2e e0        	sd	a1, 0(sp)
   19df0: 32 89        	mv	s2, a2
   19df2: 36 f0        	sd	a3, 32(sp)
   19df4: 23 3c 91 24  	sd	s1, 600(sp)
   19df8: 23 34 31 25  	sd	s3, 584(sp)
   19dfc: 23 30 41 25  	sd	s4, 576(sp)
   19e00: 23 3c 51 23  	sd	s5, 568(sp)
   19e04: 23 38 61 23  	sd	s6, 560(sp)
   19e08: 23 34 71 23  	sd	s7, 552(sp)
   19e0c: 23 30 81 23  	sd	s8, 544(sp)
   19e10: 23 3c 91 21  	sd	s9, 536(sp)
   19e14: 23 38 a1 21  	sd	s10, 528(sp)
   19e18: 23 34 b1 21  	sd	s11, 520(sp)
   19e1c: aa e8        	sd	a0, 80(sp)
   19e1e: ef e0 0f cb  	jal	0x182ce <_localeconv_r>
   19e22: 1c 61        	ld	a5, 0(a0)
   19e24: 3e 85        	mv	a0, a5
   19e26: be f0        	sd	a5, 96(sp)
   19e28: ef f0 7f e9  	jal	0x19cbe <strlen>
   19e2c: 03 57 04 01  	lhu	a4, 16(s0)
   19e30: 82 f9        	sd	zero, 240(sp)
   19e32: 82 fd        	sd	zero, 248(sp)
   19e34: 13 77 07 08  	andi	a4, a4, 128
   19e38: aa ec        	sd	a0, 88(sp)
   19e3a: 01 c7        	beqz	a4, 0x19e42 <_svfprintf_r+0x66>
   19e3c: 18 6c        	ld	a4, 24(s0)
   19e3e: e3 05 07 6a  	beqz	a4, 0x1ace8 <_svfprintf_r+0xf0c>
   19e42: 1c 03        	addi	a5, sp, 384
   19e44: 37 27 02 00  	lui	a4, 34
   19e48: 3e 83        	mv	t1, a5
   19e4a: 3e e2        	sd	a5, 256(sp)
   19e4c: 4a 8b        	mv	s6, s2
   19e4e: 93 07 87 33  	addi	a5, a4, 824
   19e52: 37 27 02 00  	lui	a4, 34
   19e56: 3e f4        	sd	a5, 40(sp)
   19e58: 93 07 07 54  	addi	a5, a4, 1344
   19e5c: 03 47 0b 00  	lbu	a4, 0(s6)
   19e60: 02 ea        	sd	zero, 272(sp)
   19e62: 23 24 01 10  	sw	zero, 264(sp)
   19e66: 02 fc        	sd	zero, 56(sp)
   19e68: 82 f4        	sd	zero, 104(sp)
   19e6a: 82 f8        	sd	zero, 112(sp)
   19e6c: 82 fc        	sd	zero, 120(sp)
   19e6e: 02 e1        	sd	zero, 128(sp)
   19e70: 02 e5        	sd	zero, 136(sp)
   19e72: 02 e8        	sd	zero, 16(sp)
   19e74: 3e e4        	sd	a5, 8(sp)
   19e76: c6 69        	ld	s3, 80(sp)
   19e78: 1a 84        	mv	s0, t1
   19e7a: 63 01 07 1c  	beqz	a4, 0x1a03c <_svfprintf_r+0x260>
   19e7e: da 84        	mv	s1, s6
   19e80: 93 06 50 02  	addi	a3, zero, 37
   19e84: 63 0c d7 20  	beq	a4, a3, 0x1a09c <_svfprintf_r+0x2c0>
   19e88: 03 c7 14 00  	lbu	a4, 1(s1)
   19e8c: 85 04        	addi	s1, s1, 1
   19e8e: 7d fb        	bnez	a4, 0x19e84 <_svfprintf_r+0xa8>
   19e90: 3b 89 64 41  	subw	s2, s1, s6
   19e94: 63 04 09 1a  	beqz	s2, 0x1a03c <_svfprintf_r+0x260>
   19e98: d2 6b        	ld	s7, 272(sp)
   19e9a: 83 26 81 10  	lw	a3, 264(sp)
   19e9e: 23 30 64 01  	sd	s6, 0(s0)
   19ea2: 33 87 2b 01  	add	a4, s7, s2
   19ea6: 1b 86 16 00  	addiw	a2, a3, 1
   19eaa: 23 34 24 01  	sd	s2, 8(s0)
   19eae: 3a ea        	sd	a4, 272(sp)
   19eb0: 23 24 c1 10  	sw	a2, 264(sp)
   19eb4: 1d 47        	addi	a4, zero, 7
   19eb6: 41 04        	addi	s0, s0, 16
   19eb8: 63 49 c7 1e  	blt	a4, a2, 0x1a0aa <_svfprintf_r+0x2ce>
   19ebc: c2 67        	ld	a5, 16(sp)
   19ebe: 03 c7 04 00  	lbu	a4, 0(s1)
   19ec2: bb 87 27 01  	addw	a5, a5, s2
   19ec6: 3e e8        	sd	a5, 16(sp)
   19ec8: 63 0a 07 16  	beqz	a4, 0x1a03c <_svfprintf_r+0x260>
   19ecc: 7d 53        	addi	t1, zero, -1
   19ece: 03 ce 14 00  	lbu	t3, 1(s1)
   19ed2: a3 07 01 0c  	sb	zero, 207(sp)
   19ed6: 85 04        	addi	s1, s1, 1
   19ed8: 01 4a        	mv	s4, zero
   19eda: 01 49        	mv	s2, zero
   19edc: 93 0b a0 05  	addi	s7, zero, 90
   19ee0: 9a 8a        	mv	s5, t1
   19ee2: 85 04        	addi	s1, s1, 1
   19ee4: 1b 0d 0e 00  	sext.w	s10, t3
   19ee8: 9b 07 0d fe  	addiw	a5, s10, -32
   19eec: 1b 87 07 00  	sext.w	a4, a5
   19ef0: 63 e5 eb 04  	bltu	s7, a4, 0x19f3a <_svfprintf_r+0x15e>
   19ef4: 13 97 07 02  	slli	a4, a5, 32
   19ef8: 93 57 e7 01  	srli	a5, a4, 30
   19efc: 22 77        	ld	a4, 40(sp)
   19efe: ba 97        	add	a5, a5, a4
   19f00: 9c 43        	lw	a5, 0(a5)
   19f02: 82 87        	jr	a5
   19f04: 01 4a        	mv	s4, zero
   19f06: a5 46        	addi	a3, zero, 9
   19f08: 03 c7 04 00  	lbu	a4, 0(s1)
   19f0c: 9b 17 2a 00  	slliw	a5, s4, 2
   19f10: 3b 8a 47 01  	addw	s4, a5, s4
   19f14: 1b 0d 0d fd  	addiw	s10, s10, -48
   19f18: 1b 1a 1a 00  	slliw	s4, s4, 1
   19f1c: 9b 07 07 fd  	addiw	a5, a4, -48
   19f20: 3b 0a 4d 01  	addw	s4, s10, s4
   19f24: 85 04        	addi	s1, s1, 1
   19f26: 1b 0d 07 00  	sext.w	s10, a4
   19f2a: e3 ff f6 fc  	bgeu	a3, a5, 0x19f08 <_svfprintf_r+0x12c>
   19f2e: 9b 07 0d fe  	addiw	a5, s10, -32
   19f32: 1b 87 07 00  	sext.w	a4, a5
   19f36: e3 ff eb fa  	bgeu	s7, a4, 0x19ef4 <_svfprintf_r+0x118>
   19f3a: 63 01 0d 10  	beqz	s10, 0x1a03c <_svfprintf_r+0x260>
   19f3e: 23 0c a1 11  	sb	s10, 280(sp)
   19f42: a3 07 01 0c  	sb	zero, 207(sp)
   19f46: 05 4c        	addi	s8, zero, 1
   19f48: 85 4d        	addi	s11, zero, 1
   19f4a: 13 0b 81 11  	addi	s6, sp, 280
   19f4e: 02 ec        	sd	zero, 24(sp)
   19f50: 01 43        	mv	t1, zero
   19f52: 82 e4        	sd	zero, 72(sp)
   19f54: 82 e0        	sd	zero, 64(sp)
   19f56: 02 f8        	sd	zero, 48(sp)
   19f58: 93 7c 29 00  	andi	s9, s2, 2
   19f5c: 63 83 0c 00  	beqz	s9, 0x19f62 <_svfprintf_r+0x186>
   19f60: 09 2c        	addiw	s8, s8, 2
   19f62: 93 7a 49 08  	andi	s5, s2, 132
   19f66: 63 96 0a 00  	bnez	s5, 0x19f72 <_svfprintf_r+0x196>
   19f6a: 3b 07 8a 41  	subw	a4, s4, s8
   19f6e: e3 4c e0 00  	bgtz	a4, 0x1a786 <_svfprintf_r+0x9aa>
   19f72: 03 47 f1 0c  	lbu	a4, 207(sp)
   19f76: d2 6b        	ld	s7, 272(sp)
   19f78: 1d c3        	beqz	a4, 0x19f9e <_svfprintf_r+0x1c2>
   19f7a: 03 26 81 10  	lw	a2, 264(sp)
   19f7e: 93 05 f1 0c  	addi	a1, sp, 207
   19f82: 0c e0        	sd	a1, 0(s0)
   19f84: 05 26        	addiw	a2, a2, 1
   19f86: 85 45        	addi	a1, zero, 1
   19f88: 85 0b        	addi	s7, s7, 1
   19f8a: 0c e4        	sd	a1, 8(s0)
   19f8c: 23 24 c1 10  	sw	a2, 264(sp)
   19f90: 9b 05 06 00  	sext.w	a1, a2
   19f94: 5e ea        	sd	s7, 272(sp)
   19f96: 1d 46        	addi	a2, zero, 7
   19f98: 41 04        	addi	s0, s0, 16
   19f9a: 63 4d b6 1c  	blt	a2, a1, 0x1a174 <_svfprintf_r+0x398>
   19f9e: 63 83 0c 02  	beqz	s9, 0x19fc4 <_svfprintf_r+0x1e8>
   19fa2: 03 26 81 10  	lw	a2, 264(sp)
   19fa6: 89 45        	addi	a1, zero, 2
   19fa8: 89 0b        	addi	s7, s7, 2
   19faa: 05 26        	addiw	a2, a2, 1
   19fac: 9c 09        	addi	a5, sp, 208
   19fae: 0c e4        	sd	a1, 8(s0)
   19fb0: 1c e0        	sd	a5, 0(s0)
   19fb2: 9b 05 06 00  	sext.w	a1, a2
   19fb6: 23 24 c1 10  	sw	a2, 264(sp)
   19fba: 5e ea        	sd	s7, 272(sp)
   19fbc: 1d 46        	addi	a2, zero, 7
   19fbe: 41 04        	addi	s0, s0, 16
   19fc0: e3 40 b6 08  	blt	a2, a1, 0x1a840 <_svfprintf_r+0xa64>
   19fc4: 13 06 00 08  	addi	a2, zero, 128
   19fc8: 63 89 ca 66  	beq	s5, a2, 0x1a63a <_svfprintf_r+0x85e>
   19fcc: bb 0c b3 41  	subw	s9, t1, s11
   19fd0: 63 44 90 71  	bgtz	s9, 0x1a6d8 <_svfprintf_r+0x8fc>
   19fd4: 13 76 09 10  	andi	a2, s2, 256
   19fd8: 63 1c 06 50  	bnez	a2, 0x1a4f0 <_svfprintf_r+0x714>
   19fdc: 03 26 81 10  	lw	a2, 264(sp)
   19fe0: 33 87 7d 01  	add	a4, s11, s7
   19fe4: 23 30 64 01  	sd	s6, 0(s0)
   19fe8: 9b 06 16 00  	addiw	a3, a2, 1
   19fec: 23 34 b4 01  	sd	s11, 8(s0)
   19ff0: 3a ea        	sd	a4, 272(sp)
   19ff2: 23 24 d1 10  	sw	a3, 264(sp)
   19ff6: 9d 47        	addi	a5, zero, 7
   19ff8: 63 cf d7 22  	blt	a5, a3, 0x1a236 <_svfprintf_r+0x45a>
   19ffc: 41 04        	addi	s0, s0, 16
   19ffe: 13 79 49 00  	andi	s2, s2, 4
   1a002: 63 06 09 00  	beqz	s2, 0x1a00e <_svfprintf_r+0x232>
   1a006: 3b 09 8a 41  	subw	s2, s4, s8
   1a00a: e3 47 20 05  	bgtz	s2, 0x1a858 <_svfprintf_r+0xa7c>
   1a00e: d2 87        	mv	a5, s4
   1a010: 63 53 8a 01  	bge	s4, s8, 0x1a016 <_svfprintf_r+0x23a>
   1a014: e2 87        	mv	a5, s8
   1a016: c2 66        	ld	a3, 16(sp)
   1a018: b5 9f        	addw	a5, a5, a3
   1a01a: 3e e8        	sd	a5, 16(sp)
   1a01c: 63 17 07 74  	bnez	a4, 0x1a76a <_svfprintf_r+0x98e>
   1a020: e2 67        	ld	a5, 24(sp)
   1a022: 23 24 01 10  	sw	zero, 264(sp)
   1a026: 89 c7        	beqz	a5, 0x1a030 <_svfprintf_r+0x254>
   1a028: e2 65        	ld	a1, 24(sp)
   1a02a: 4e 85        	mv	a0, s3
   1a02c: ef b0 bf 83  	jal	0x15866 <_free_r>
   1a030: 00 03        	addi	s0, sp, 384
   1a032: 26 8b        	mv	s6, s1
   1a034: 03 47 0b 00  	lbu	a4, 0(s6)
   1a038: e3 13 07 e4  	bnez	a4, 0x19e7e <_svfprintf_r+0xa2>
   1a03c: d2 67        	ld	a5, 272(sp)
   1a03e: 99 c3        	beqz	a5, 0x1a044 <_svfprintf_r+0x268>
   1a040: 6f 10 20 33  	j	0x1b372 <_svfprintf_r+0x1596>
   1a044: 82 67        	ld	a5, 0(sp)
   1a046: 83 d7 07 01  	lhu	a5, 16(a5)
   1a04a: 93 f7 07 04  	andi	a5, a5, 64
   1a04e: 99 c3        	beqz	a5, 0x1a054 <_svfprintf_r+0x278>
   1a050: 6f 10 e0 7e  	j	0x1b83e <_svfprintf_r+0x1a62>
   1a054: 83 30 81 26  	ld	ra, 616(sp)
   1a058: 03 34 01 26  	ld	s0, 608(sp)
   1a05c: 42 65        	ld	a0, 16(sp)
   1a05e: 83 34 81 25  	ld	s1, 600(sp)
   1a062: 03 39 01 25  	ld	s2, 592(sp)
   1a066: 83 39 81 24  	ld	s3, 584(sp)
   1a06a: 03 3a 01 24  	ld	s4, 576(sp)
   1a06e: 83 3a 81 23  	ld	s5, 568(sp)
   1a072: 03 3b 01 23  	ld	s6, 560(sp)
   1a076: 83 3b 81 22  	ld	s7, 552(sp)
   1a07a: 03 3c 01 22  	ld	s8, 544(sp)
   1a07e: 83 3c 81 21  	ld	s9, 536(sp)
   1a082: 03 3d 01 21  	ld	s10, 528(sp)
   1a086: 83 3d 81 20  	ld	s11, 520(sp)
   1a08a: 13 01 01 27  	addi	sp, sp, 624
   1a08e: 82 80        	ret
   1a090: 13 69 09 01  	ori	s2, s2, 16
   1a094: 03 ce 04 00  	lbu	t3, 0(s1)
   1a098: 01 29        	sext.w	s2, s2
   1a09a: a1 b5        	j	0x19ee2 <_svfprintf_r+0x106>
   1a09c: 3b 89 64 41  	subw	s2, s1, s6
   1a0a0: e3 1c 09 de  	bnez	s2, 0x19e98 <_svfprintf_r+0xbc>
   1a0a4: 03 c7 04 00  	lbu	a4, 0(s1)
   1a0a8: 05 b5        	j	0x19ec8 <_svfprintf_r+0xec>
   1a0aa: 82 65        	ld	a1, 0(sp)
   1a0ac: 10 02        	addi	a2, sp, 256
   1a0ae: 4e 85        	mv	a0, s3
   1a0b0: ef 30 e0 59  	jal	0x1d64e <__ssprint_r>
   1a0b4: 41 f9        	bnez	a0, 0x1a044 <_svfprintf_r+0x268>
   1a0b6: 00 03        	addi	s0, sp, 384
   1a0b8: 11 b5        	j	0x19ebc <_svfprintf_r+0xe0>
   1a0ba: 13 77 89 00  	andi	a4, s2, 8
   1a0be: 56 83        	mv	t1, s5
   1a0c0: e3 07 07 30  	beqz	a4, 0x1abce <_svfprintf_r+0xdf2>
   1a0c4: 82 77        	ld	a5, 32(sp)
   1a0c6: 93 8a f7 00  	addi	s5, a5, 15
   1a0ca: 93 fa 0a ff  	andi	s5, s5, -16
   1a0ce: 03 b7 0a 00  	ld	a4, 0(s5)
   1a0d2: 83 b5 8a 00  	ld	a1, 8(s5)
   1a0d6: 93 87 0a 01  	addi	a5, s5, 16
   1a0da: 3e f0        	sd	a5, 32(sp)
   1a0dc: 88 19        	addi	a0, sp, 240
   1a0de: 1a ec        	sd	t1, 24(sp)
   1a0e0: ba f9        	sd	a4, 240(sp)
   1a0e2: ae fd        	sd	a1, 248(sp)
   1a0e4: ef e0 2f 9a  	jal	0x18286 <_ldcheck>
   1a0e8: aa cd        	sw	a0, 216(sp)
   1a0ea: 09 47        	addi	a4, zero, 2
   1a0ec: 62 63        	ld	t1, 24(sp)
   1a0ee: e3 0c e5 6e  	beq	a0, a4, 0x1afe6 <_svfprintf_r+0x120a>
   1a0f2: 05 47        	addi	a4, zero, 1
   1a0f4: 63 14 e5 00  	bne	a0, a4, 0x1a0fc <_svfprintf_r+0x320>
   1a0f8: 6f 10 20 02  	j	0x1b11a <_svfprintf_r+0x133e>
   1a0fc: 13 07 10 06  	addi	a4, zero, 97
   1a100: 63 14 ed 00  	bne	s10, a4, 0x1a108 <_svfprintf_r+0x32c>
   1a104: 6f 10 80 4f  	j	0x1b5fc <_svfprintf_r+0x1820>
   1a108: 13 07 10 04  	addi	a4, zero, 65
   1a10c: 63 14 ed 00  	bne	s10, a4, 0x1a114 <_svfprintf_r+0x338>
   1a110: 6f 10 20 62  	j	0x1b732 <_svfprintf_r+0x1956>
   1a114: 7d 57        	addi	a4, zero, -1
   1a116: 63 14 e3 00  	bne	t1, a4, 0x1a11e <_svfprintf_r+0x342>
   1a11a: 6f 10 40 28  	j	0x1b39e <_svfprintf_r+0x15c2>
   1a11e: 13 77 fd fd  	andi	a4, s10, -33
   1a122: 93 06 70 04  	addi	a3, zero, 71
   1a126: 63 14 d7 00  	bne	a4, a3, 0x1a12e <_svfprintf_r+0x352>
   1a12a: 6f 10 00 70  	j	0x1b82a <_svfprintf_r+0x1a4e>
   1a12e: 13 67 09 10  	ori	a4, s2, 256
   1a132: ee 7c        	ld	s9, 248(sp)
   1a134: 9b 07 07 00  	sext.w	a5, a4
   1a138: 3e e9        	sd	a5, 144(sp)
   1a13a: ca e0        	sd	s2, 64(sp)
   1a13c: 4e 7c        	ld	s8, 240(sp)
   1a13e: 63 d4 0c 00  	bgez	s9, 0x1a146 <_svfprintf_r+0x36a>
   1a142: 6f 10 20 54  	j	0x1b684 <_svfprintf_r+0x18a8>
   1a146: 02 ed        	sd	zero, 152(sp)
   1a148: 02 ec        	sd	zero, 24(sp)
   1a14a: 1b 07 fd fb  	addiw	a4, s10, -65
   1a14e: 1b 06 07 00  	sext.w	a2, a4
   1a152: 93 06 50 02  	addi	a3, zero, 37
   1a156: 63 f4 c6 00  	bgeu	a3, a2, 0x1a15e <_svfprintf_r+0x382>
   1a15a: 6f 10 40 07  	j	0x1b1ce <_svfprintf_r+0x13f2>
   1a15e: 93 16 07 02  	slli	a3, a4, 32
   1a162: 13 d7 e6 01  	srli	a4, a3, 30
   1a166: b7 26 02 00  	lui	a3, 34
   1a16a: 93 86 46 4a  	addi	a3, a3, 1188
   1a16e: 36 97        	add	a4, a4, a3
   1a170: 18 43        	lw	a4, 0(a4)
   1a172: 02 87        	jr	a4
   1a174: 82 65        	ld	a1, 0(sp)
   1a176: 10 02        	addi	a2, sp, 256
   1a178: 4e 85        	mv	a0, s3
   1a17a: 1a e9        	sd	t1, 144(sp)
   1a17c: ef 30 20 4d  	jal	0x1d64e <__ssprint_r>
   1a180: 63 1c 05 5e  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a184: d2 6b        	ld	s7, 272(sp)
   1a186: 4a 63        	ld	t1, 144(sp)
   1a188: 00 03        	addi	s0, sp, 384
   1a18a: 11 bd        	j	0x19f9e <_svfprintf_r+0x1c2>
   1a18c: e2 77        	ld	a5, 56(sp)
   1a18e: 85 46        	addi	a3, zero, 1
   1a190: 93 0c 04 01  	addi	s9, s0, 16
   1a194: e3 d9 f6 06  	bge	a3, a5, 0x1aa06 <_svfprintf_r+0xc2a>
   1a198: 03 26 81 10  	lw	a2, 264(sp)
   1a19c: 85 46        	addi	a3, zero, 1
   1a19e: 13 87 1b 00  	addi	a4, s7, 1
   1a1a2: 05 26        	addiw	a2, a2, 1
   1a1a4: 14 e4        	sd	a3, 8(s0)
   1a1a6: 23 24 c1 10  	sw	a2, 264(sp)
   1a1aa: 9b 06 06 00  	sext.w	a3, a2
   1a1ae: 23 30 64 01  	sd	s6, 0(s0)
   1a1b2: 3a ea        	sd	a4, 272(sp)
   1a1b4: 1d 46        	addi	a2, zero, 7
   1a1b6: e3 4d d6 16  	blt	a2, a3, 0x1ab30 <_svfprintf_r+0xd54>
   1a1ba: e6 67        	ld	a5, 88(sp)
   1a1bc: 1b 84 16 00  	addiw	s0, a3, 1
   1a1c0: 23 24 81 10  	sw	s0, 264(sp)
   1a1c4: 33 8d e7 00  	add	s10, a5, a4
   1a1c8: 06 77        	ld	a4, 96(sp)
   1a1ca: 23 b4 fc 00  	sd	a5, 8(s9)
   1a1ce: 6a ea        	sd	s10, 272(sp)
   1a1d0: 23 b0 ec 00  	sd	a4, 0(s9)
   1a1d4: 1d 47        	addi	a4, zero, 7
   1a1d6: c1 0c        	addi	s9, s9, 16
   1a1d8: e3 4a 87 16  	blt	a4, s0, 0x1ab4c <_svfprintf_r+0xd70>
   1a1dc: 4e 75        	ld	a0, 240(sp)
   1a1de: ee 75        	ld	a1, 248(sp)
   1a1e0: 01 46        	mv	a2, zero
   1a1e2: 81 46        	mv	a3, zero
   1a1e4: ef 40 d0 13  	jal	0x1eb20 <__netf2>
   1a1e8: e2 77        	ld	a5, 56(sp)
   1a1ea: 9b 8d f7 ff  	addiw	s11, a5, -1
   1a1ee: e3 0f 05 04  	beqz	a0, 0x1aa4c <_svfprintf_r+0xc70>
   1a1f2: 93 07 1b 00  	addi	a5, s6, 1
   1a1f6: 05 24        	addiw	s0, s0, 1
   1a1f8: 6e 9d        	add	s10, s10, s11
   1a1fa: 23 b0 fc 00  	sd	a5, 0(s9)
   1a1fe: 23 b4 bc 01  	sd	s11, 8(s9)
   1a202: 6a ea        	sd	s10, 272(sp)
   1a204: 23 24 81 10  	sw	s0, 264(sp)
   1a208: 9d 47        	addi	a5, zero, 7
   1a20a: e3 c3 87 02  	blt	a5, s0, 0x1aa30 <_svfprintf_r+0xc54>
   1a20e: c1 0c        	addi	s9, s9, 16
   1a210: c6 76        	ld	a3, 112(sp)
   1a212: 05 24        	addiw	s0, s0, 1
   1a214: 9c 11        	addi	a5, sp, 224
   1a216: 33 87 a6 01  	add	a4, a3, s10
   1a21a: 23 b0 fc 00  	sd	a5, 0(s9)
   1a21e: 23 b4 dc 00  	sd	a3, 8(s9)
   1a222: 23 24 81 10  	sw	s0, 264(sp)
   1a226: 9b 06 04 00  	sext.w	a3, s0
   1a22a: 3a ea        	sd	a4, 272(sp)
   1a22c: 9d 47        	addi	a5, zero, 7
   1a22e: 13 84 0c 01  	addi	s0, s9, 16
   1a232: e3 d6 d7 dc  	bge	a5, a3, 0x19ffe <_svfprintf_r+0x222>
   1a236: 82 65        	ld	a1, 0(sp)
   1a238: 10 02        	addi	a2, sp, 256
   1a23a: 4e 85        	mv	a0, s3
   1a23c: ef 30 20 41  	jal	0x1d64e <__ssprint_r>
   1a240: 63 1c 05 52  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a244: 52 67        	ld	a4, 272(sp)
   1a246: 00 03        	addi	s0, sp, 384
   1a248: 5d bb        	j	0x19ffe <_svfprintf_r+0x222>
   1a24a: 02 77        	ld	a4, 32(sp)
   1a24c: a3 07 01 0c  	sb	zero, 207(sp)
   1a250: 05 4c        	addi	s8, zero, 1
   1a252: 1c 43        	lw	a5, 0(a4)
   1a254: 21 07        	addi	a4, a4, 8
   1a256: 3a f0        	sd	a4, 32(sp)
   1a258: 23 0c f1 10  	sb	a5, 280(sp)
   1a25c: 85 4d        	addi	s11, zero, 1
   1a25e: 13 0b 81 11  	addi	s6, sp, 280
   1a262: f5 b1        	j	0x19f4e <_svfprintf_r+0x172>
   1a264: 82 77        	ld	a5, 32(sp)
   1a266: a3 07 01 0c  	sb	zero, 207(sp)
   1a26a: 56 83        	mv	t1, s5
   1a26c: 03 bb 07 00  	ld	s6, 0(a5)
   1a270: 93 8a 87 00  	addi	s5, a5, 8
   1a274: e3 09 0b 5a  	beqz	s6, 0x1b026 <_svfprintf_r+0x124a>
   1a278: 7d 57        	addi	a4, zero, -1
   1a27a: e3 01 e3 2e  	beq	t1, a4, 0x1ad5c <_svfprintf_r+0xf80>
   1a27e: 1a 86        	mv	a2, t1
   1a280: 81 45        	mv	a1, zero
   1a282: 5a 85        	mv	a0, s6
   1a284: 1a f0        	sd	t1, 32(sp)
   1a286: ef e0 7f 8e  	jal	0x18b6c <memchr>
   1a28a: 2a ec        	sd	a0, 24(sp)
   1a28c: 02 73        	ld	t1, 32(sp)
   1a28e: 19 e1        	bnez	a0, 0x1a294 <_svfprintf_r+0x4b8>
   1a290: 6f 10 c0 42  	j	0x1b6bc <_svfprintf_r+0x18e0>
   1a294: e2 67        	ld	a5, 24(sp)
   1a296: bb 8d 67 41  	subw	s11, a5, s6
   1a29a: 03 47 f1 0c  	lbu	a4, 207(sp)
   1a29e: 13 c8 fd ff  	not	a6, s11
   1a2a2: 13 58 f8 43  	srai	a6, a6, 63
   1a2a6: 56 f0        	sd	s5, 32(sp)
   1a2a8: 02 ec        	sd	zero, 24(sp)
   1a2aa: 82 e4        	sd	zero, 72(sp)
   1a2ac: 82 e0        	sd	zero, 64(sp)
   1a2ae: 02 f8        	sd	zero, 48(sp)
   1a2b0: 33 7c b8 01  	and	s8, a6, s11
   1a2b4: 01 43        	mv	t1, zero
   1a2b6: e3 01 07 ca  	beqz	a4, 0x19f58 <_svfprintf_r+0x17c>
   1a2ba: 05 2c        	addiw	s8, s8, 1
   1a2bc: 71 b9        	j	0x19f58 <_svfprintf_r+0x17c>
   1a2be: 82 77        	ld	a5, 32(sp)
   1a2c0: 03 aa 07 00  	lw	s4, 0(a5)
   1a2c4: a1 07        	addi	a5, a5, 8
   1a2c6: e3 59 0a 0a  	bgez	s4, 0x1ab78 <_svfprintf_r+0xd9c>
   1a2ca: 3b 0a 40 41  	negw	s4, s4
   1a2ce: 3e f0        	sd	a5, 32(sp)
   1a2d0: 13 69 49 00  	ori	s2, s2, 4
   1a2d4: 03 ce 04 00  	lbu	t3, 0(s1)
   1a2d8: 01 29        	sext.w	s2, s2
   1a2da: 21 b1        	j	0x19ee2 <_svfprintf_r+0x106>
   1a2dc: 93 07 b0 02  	addi	a5, zero, 43
   1a2e0: 03 ce 04 00  	lbu	t3, 0(s1)
   1a2e4: a3 07 f1 0c  	sb	a5, 207(sp)
   1a2e8: ed be        	j	0x19ee2 <_svfprintf_r+0x106>
   1a2ea: 13 69 09 08  	ori	s2, s2, 128
   1a2ee: 03 ce 04 00  	lbu	t3, 0(s1)
   1a2f2: 01 29        	sext.w	s2, s2
   1a2f4: fd b6        	j	0x19ee2 <_svfprintf_r+0x106>
   1a2f6: 03 cd 04 00  	lbu	s10, 0(s1)
   1a2fa: 93 06 a0 02  	addi	a3, zero, 42
   1a2fe: 93 87 14 00  	addi	a5, s1, 1
   1a302: 63 14 dd 00  	bne	s10, a3, 0x1a30a <_svfprintf_r+0x52e>
   1a306: 6f 10 40 5e  	j	0x1b8ea <_svfprintf_r+0x1b0e>
   1a30a: 1b 07 0d fd  	addiw	a4, s10, -48
   1a30e: 25 46        	addi	a2, zero, 9
   1a310: be 84        	mv	s1, a5
   1a312: 81 4a        	mv	s5, zero
   1a314: a5 46        	addi	a3, zero, 9
   1a316: e3 69 e6 bc  	bltu	a2, a4, 0x19ee8 <_svfprintf_r+0x10c>
   1a31a: 03 cd 04 00  	lbu	s10, 0(s1)
   1a31e: 9b 97 2a 00  	slliw	a5, s5, 2
   1a322: 3b 83 57 01  	addw	t1, a5, s5
   1a326: 1b 13 13 00  	slliw	t1, t1, 1
   1a32a: bb 0a e3 00  	addw	s5, t1, a4
   1a32e: 1b 07 0d fd  	addiw	a4, s10, -48
   1a332: 85 04        	addi	s1, s1, 1
   1a334: e3 f3 e6 fe  	bgeu	a3, a4, 0x1a31a <_svfprintf_r+0x53e>
   1a338: 45 be        	j	0x19ee8 <_svfprintf_r+0x10c>
   1a33a: 4e 85        	mv	a0, s3
   1a33c: ef d0 3f f9  	jal	0x182ce <_localeconv_r>
   1a340: 1c 65        	ld	a5, 8(a0)
   1a342: 3e 85        	mv	a0, a5
   1a344: 3e e5        	sd	a5, 136(sp)
   1a346: ef f0 9f 97  	jal	0x19cbe <strlen>
   1a34a: aa 87        	mv	a5, a0
   1a34c: 4e 85        	mv	a0, s3
   1a34e: 3e 8c        	mv	s8, a5
   1a350: 3e e1        	sd	a5, 128(sp)
   1a352: ef d0 df f7  	jal	0x182ce <_localeconv_r>
   1a356: 1c 69        	ld	a5, 16(a0)
   1a358: 03 ce 04 00  	lbu	t3, 0(s1)
   1a35c: be fc        	sd	a5, 120(sp)
   1a35e: e3 02 0c b8  	beqz	s8, 0x19ee2 <_svfprintf_r+0x106>
   1a362: e3 80 07 b8  	beqz	a5, 0x19ee2 <_svfprintf_r+0x106>
   1a366: 83 c7 07 00  	lbu	a5, 0(a5)
   1a36a: e3 8c 07 b6  	beqz	a5, 0x19ee2 <_svfprintf_r+0x106>
   1a36e: 13 69 09 40  	ori	s2, s2, 1024
   1a372: 01 29        	sext.w	s2, s2
   1a374: bd b6        	j	0x19ee2 <_svfprintf_r+0x106>
   1a376: 13 69 19 00  	ori	s2, s2, 1
   1a37a: 03 ce 04 00  	lbu	t3, 0(s1)
   1a37e: 01 29        	sext.w	s2, s2
   1a380: 8d b6        	j	0x19ee2 <_svfprintf_r+0x106>
   1a382: 83 47 f1 0c  	lbu	a5, 207(sp)
   1a386: 03 ce 04 00  	lbu	t3, 0(s1)
   1a38a: e3 9c 07 b4  	bnez	a5, 0x19ee2 <_svfprintf_r+0x106>
   1a38e: 93 07 00 02  	addi	a5, zero, 32
   1a392: a3 07 f1 0c  	sb	a5, 207(sp)
   1a396: b1 b6        	j	0x19ee2 <_svfprintf_r+0x106>
   1a398: 93 6c 09 01  	ori	s9, s2, 16
   1a39c: 56 83        	mv	t1, s5
   1a39e: 81 2c        	sext.w	s9, s9
   1a3a0: 82 76        	ld	a3, 32(sp)
   1a3a2: 93 f7 0c 02  	andi	a5, s9, 32
   1a3a6: 13 87 86 00  	addi	a4, a3, 8
   1a3aa: 63 86 07 72  	beqz	a5, 0x1aad6 <_svfprintf_r+0xcfa>
   1a3ae: 83 bb 06 00  	ld	s7, 0(a3)
   1a3b2: 85 47        	addi	a5, zero, 1
   1a3b4: 3a f0        	sd	a4, 32(sp)
   1a3b6: 01 47        	mv	a4, zero
   1a3b8: a3 07 e1 0c  	sb	a4, 207(sp)
   1a3bc: 7d 57        	addi	a4, zero, -1
   1a3be: 63 0d e3 1e  	beq	t1, a4, 0x1a5b8 <_svfprintf_r+0x7dc>
   1a3c2: 13 f9 fc f7  	andi	s2, s9, -129
   1a3c6: 01 29        	sext.w	s2, s2
   1a3c8: 63 97 0b 1e  	bnez	s7, 0x1a5b6 <_svfprintf_r+0x7da>
   1a3cc: 63 1a 03 22  	bnez	t1, 0x1a600 <_svfprintf_r+0x824>
   1a3d0: 63 9c 07 74  	bnez	a5, 0x1ab28 <_svfprintf_r+0xd4c>
   1a3d4: 93 fd 1c 00  	andi	s11, s9, 1
   1a3d8: 13 0b c1 17  	addi	s6, sp, 380
   1a3dc: 63 93 0d 7a  	bnez	s11, 0x1ab82 <_svfprintf_r+0xda6>
   1a3e0: 03 47 f1 0c  	lbu	a4, 207(sp)
   1a3e4: 1a 88        	mv	a6, t1
   1a3e6: 63 53 b3 01  	bge	t1, s11, 0x1a3ec <_svfprintf_r+0x610>
   1a3ea: 6e 88        	mv	a6, s11
   1a3ec: 02 ec        	sd	zero, 24(sp)
   1a3ee: 82 e4        	sd	zero, 72(sp)
   1a3f0: 82 e0        	sd	zero, 64(sp)
   1a3f2: 02 f8        	sd	zero, 48(sp)
   1a3f4: 1b 0c 08 00  	sext.w	s8, a6
   1a3f8: e3 11 07 ec  	bnez	a4, 0x1a2ba <_svfprintf_r+0x4de>
   1a3fc: b1 be        	j	0x19f58 <_svfprintf_r+0x17c>
   1a3fe: 13 69 89 00  	ori	s2, s2, 8
   1a402: 03 ce 04 00  	lbu	t3, 0(s1)
   1a406: 01 29        	sext.w	s2, s2
   1a408: e9 bc        	j	0x19ee2 <_svfprintf_r+0x106>
   1a40a: 13 69 09 01  	ori	s2, s2, 16
   1a40e: 56 83        	mv	t1, s5
   1a410: 01 29        	sext.w	s2, s2
   1a412: 82 76        	ld	a3, 32(sp)
   1a414: 93 77 09 02  	andi	a5, s2, 32
   1a418: 13 87 86 00  	addi	a4, a3, 8
   1a41c: 63 8d 07 6c  	beqz	a5, 0x1aaf6 <_svfprintf_r+0xd1a>
   1a420: 9c 62        	ld	a5, 0(a3)
   1a422: be 8b        	mv	s7, a5
   1a424: 63 ca 07 6e  	bltz	a5, 0x1ab18 <_svfprintf_r+0xd3c>
   1a428: fd 57        	addi	a5, zero, -1
   1a42a: 63 0d f3 3e  	beq	t1, a5, 0x1a824 <_svfprintf_r+0xa48>
   1a42e: 93 7c f9 f7  	andi	s9, s2, -129
   1a432: 81 2c        	sext.w	s9, s9
   1a434: 63 9f 0b 7a  	bnez	s7, 0x1abf2 <_svfprintf_r+0xe16>
   1a438: 63 04 03 00  	beqz	t1, 0x1a440 <_svfprintf_r+0x664>
   1a43c: 6f 10 a0 40  	j	0x1b846 <_svfprintf_r+0x1a6a>
   1a440: 81 4d        	mv	s11, zero
   1a442: 3a f0        	sd	a4, 32(sp)
   1a444: 66 89        	mv	s2, s9
   1a446: 13 0b c1 17  	addi	s6, sp, 380
   1a44a: 59 bf        	j	0x1a3e0 <_svfprintf_r+0x604>
   1a44c: 13 69 09 01  	ori	s2, s2, 16
   1a450: 56 83        	mv	t1, s5
   1a452: 01 29        	sext.w	s2, s2
   1a454: 82 76        	ld	a3, 32(sp)
   1a456: 93 77 09 02  	andi	a5, s2, 32
   1a45a: 13 87 86 00  	addi	a4, a3, 8
   1a45e: 63 8e 07 64  	beqz	a5, 0x1aaba <_svfprintf_r+0xcde>
   1a462: 83 bb 06 00  	ld	s7, 0(a3)
   1a466: 93 7c f9 bf  	andi	s9, s2, -1025
   1a46a: 81 2c        	sext.w	s9, s9
   1a46c: 3a f0        	sd	a4, 32(sp)
   1a46e: 81 47        	mv	a5, zero
   1a470: 99 b7        	j	0x1a3b6 <_svfprintf_r+0x5da>
   1a472: 13 69 09 02  	ori	s2, s2, 32
   1a476: 03 ce 04 00  	lbu	t3, 0(s1)
   1a47a: 01 29        	sext.w	s2, s2
   1a47c: 9d b4        	j	0x19ee2 <_svfprintf_r+0x106>
   1a47e: 02 77        	ld	a4, 32(sp)
   1a480: 93 77 09 02  	andi	a5, s2, 32
   1a484: 18 63        	ld	a4, 0(a4)
   1a486: 63 84 07 72  	beqz	a5, 0x1abae <_svfprintf_r+0xdd2>
   1a48a: c2 67        	ld	a5, 16(sp)
   1a48c: 1c e3        	sd	a5, 0(a4)
   1a48e: 82 77        	ld	a5, 32(sp)
   1a490: 26 8b        	mv	s6, s1
   1a492: a1 07        	addi	a5, a5, 8
   1a494: 3e f0        	sd	a5, 32(sp)
   1a496: 79 be        	j	0x1a034 <_svfprintf_r+0x258>
   1a498: 03 ce 04 00  	lbu	t3, 0(s1)
   1a49c: 93 07 c0 06  	addi	a5, zero, 108
   1a4a0: e3 0c fe 02  	beq	t3, a5, 0x1acd8 <_svfprintf_r+0xefc>
   1a4a4: 13 69 09 01  	ori	s2, s2, 16
   1a4a8: 01 29        	sext.w	s2, s2
   1a4aa: 25 bc        	j	0x19ee2 <_svfprintf_r+0x106>
   1a4ac: 03 ce 04 00  	lbu	t3, 0(s1)
   1a4b0: 93 07 80 06  	addi	a5, zero, 104
   1a4b4: e3 0a fe 00  	beq	t3, a5, 0x1acc8 <_svfprintf_r+0xeec>
   1a4b8: 13 69 09 04  	ori	s2, s2, 64
   1a4bc: 01 29        	sext.w	s2, s2
   1a4be: 15 b4        	j	0x19ee2 <_svfprintf_r+0x106>
   1a4c0: 02 77        	ld	a4, 32(sp)
   1a4c2: e1 77        	lui	a5, 1048568
   1a4c4: 93 c7 07 83  	xori	a5, a5, -2000
   1a4c8: 23 18 f1 0c  	sh	a5, 208(sp)
   1a4cc: 93 07 87 00  	addi	a5, a4, 8
   1a4d0: 3e f0        	sd	a5, 32(sp)
   1a4d2: b7 27 02 00  	lui	a5, 34
   1a4d6: 93 87 87 c0  	addi	a5, a5, -1016
   1a4da: 93 6c 29 00  	ori	s9, s2, 2
   1a4de: be f4        	sd	a5, 104(sp)
   1a4e0: 83 3b 07 00  	ld	s7, 0(a4)
   1a4e4: 56 83        	mv	t1, s5
   1a4e6: 81 2c        	sext.w	s9, s9
   1a4e8: 89 47        	addi	a5, zero, 2
   1a4ea: 13 0d 80 07  	addi	s10, zero, 120
   1a4ee: e1 b5        	j	0x1a3b6 <_svfprintf_r+0x5da>
   1a4f0: 93 06 50 06  	addi	a3, zero, 101
   1a4f4: e3 dc a6 c9  	bge	a3, s10, 0x1a18c <_svfprintf_r+0x3b0>
   1a4f8: 4e 75        	ld	a0, 240(sp)
   1a4fa: ee 75        	ld	a1, 248(sp)
   1a4fc: 01 46        	mv	a2, zero
   1a4fe: 81 46        	mv	a3, zero
   1a500: ef 40 00 62  	jal	0x1eb20 <__netf2>
   1a504: 63 1b 05 3c  	bnez	a0, 0x1a8da <_svfprintf_r+0xafe>
   1a508: 83 27 81 10  	lw	a5, 264(sp)
   1a50c: b7 26 02 00  	lui	a3, 34
   1a510: 93 86 86 c2  	addi	a3, a3, -984
   1a514: 85 27        	addiw	a5, a5, 1
   1a516: 14 e0        	sd	a3, 0(s0)
   1a518: 85 0b        	addi	s7, s7, 1
   1a51a: 85 46        	addi	a3, zero, 1
   1a51c: 14 e4        	sd	a3, 8(s0)
   1a51e: 23 24 f1 10  	sw	a5, 264(sp)
   1a522: 9b 86 07 00  	sext.w	a3, a5
   1a526: 5e ea        	sd	s7, 272(sp)
   1a528: 9d 47        	addi	a5, zero, 7
   1a52a: 41 04        	addi	s0, s0, 16
   1a52c: e3 c4 d7 24  	blt	a5, a3, 0x1af74 <_svfprintf_r+0x1198>
   1a530: ee 47        	lw	a5, 216(sp)
   1a532: 62 77        	ld	a4, 56(sp)
   1a534: 63 da e7 62  	bge	a5, a4, 0x1ab68 <_svfprintf_r+0xd8c>
   1a538: e6 66        	ld	a3, 88(sp)
   1a53a: 83 27 81 10  	lw	a5, 264(sp)
   1a53e: 06 76        	ld	a2, 96(sp)
   1a540: 33 87 76 01  	add	a4, a3, s7
   1a544: 85 27        	addiw	a5, a5, 1
   1a546: 14 e4        	sd	a3, 8(s0)
   1a548: 10 e0        	sd	a2, 0(s0)
   1a54a: 9b 86 07 00  	sext.w	a3, a5
   1a54e: 23 24 f1 10  	sw	a5, 264(sp)
   1a552: 3a ea        	sd	a4, 272(sp)
   1a554: 9d 47        	addi	a5, zero, 7
   1a556: 41 04        	addi	s0, s0, 16
   1a558: 63 c4 d7 74  	blt	a5, a3, 0x1aca0 <_svfprintf_r+0xec4>
   1a55c: e2 77        	ld	a5, 56(sp)
   1a55e: 9b 8c f7 ff  	addiw	s9, a5, -1
   1a562: e3 5e 90 a9  	blez	s9, 0x19ffe <_svfprintf_r+0x222>
   1a566: 37 2f 02 00  	lui	t5, 34
   1a56a: c1 46        	addi	a3, zero, 16
   1a56c: 83 27 81 10  	lw	a5, 264(sp)
   1a570: 93 0b 0f 54  	addi	s7, t5, 1344
   1a574: e3 db 96 21  	bge	a3, s9, 0x1af8a <_svfprintf_r+0x11ae>
   1a578: 82 6a        	ld	s5, 0(sp)
   1a57a: 41 4d        	addi	s10, zero, 16
   1a57c: 9d 4d        	addi	s11, zero, 7
   1a57e: 21 a0        	j	0x1a586 <_svfprintf_r+0x7aa>
   1a580: c1 3c        	addiw	s9, s9, -16
   1a582: e3 54 9d 21  	bge	s10, s9, 0x1af8a <_svfprintf_r+0x11ae>
   1a586: 85 27        	addiw	a5, a5, 1
   1a588: 41 07        	addi	a4, a4, 16
   1a58a: 23 30 74 01  	sd	s7, 0(s0)
   1a58e: 23 34 a4 01  	sd	s10, 8(s0)
   1a592: 3a ea        	sd	a4, 272(sp)
   1a594: 23 24 f1 10  	sw	a5, 264(sp)
   1a598: 41 04        	addi	s0, s0, 16
   1a59a: e3 d3 fd fe  	bge	s11, a5, 0x1a580 <_svfprintf_r+0x7a4>
   1a59e: 10 02        	addi	a2, sp, 256
   1a5a0: d6 85        	mv	a1, s5
   1a5a2: 4e 85        	mv	a0, s3
   1a5a4: ef 30 a0 0a  	jal	0x1d64e <__ssprint_r>
   1a5a8: 63 18 05 1c  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a5ac: 52 67        	ld	a4, 272(sp)
   1a5ae: 83 27 81 10  	lw	a5, 264(sp)
   1a5b2: 00 03        	addi	s0, sp, 384
   1a5b4: f1 b7        	j	0x1a580 <_svfprintf_r+0x7a4>
   1a5b6: ca 8c        	mv	s9, s2
   1a5b8: 05 47        	addi	a4, zero, 1
   1a5ba: 63 87 e7 26  	beq	a5, a4, 0x1a828 <_svfprintf_r+0xa4c>
   1a5be: 09 47        	addi	a4, zero, 2
   1a5c0: 63 89 e7 04  	beq	a5, a4, 0x1a612 <_svfprintf_r+0x836>
   1a5c4: 93 0a c1 17  	addi	s5, sp, 380
   1a5c8: 56 8b        	mv	s6, s5
   1a5ca: 13 f7 7b 00  	andi	a4, s7, 7
   1a5ce: 13 07 07 03  	addi	a4, a4, 48
   1a5d2: a3 0f eb fe  	sb	a4, -1(s6)
   1a5d6: 93 db 3b 00  	srli	s7, s7, 3
   1a5da: 5a 86        	mv	a2, s6
   1a5dc: 7d 1b        	addi	s6, s6, -1
   1a5de: e3 96 0b fe  	bnez	s7, 0x1a5ca <_svfprintf_r+0x7ee>
   1a5e2: 93 f5 1c 00  	andi	a1, s9, 1
   1a5e6: b1 c5        	beqz	a1, 0x1a632 <_svfprintf_r+0x856>
   1a5e8: 93 05 00 03  	addi	a1, zero, 48
   1a5ec: 63 03 b7 04  	beq	a4, a1, 0x1a632 <_svfprintf_r+0x856>
   1a5f0: 79 16        	addi	a2, a2, -2
   1a5f2: a3 0f bb fe  	sb	a1, -1(s6)
   1a5f6: bb 8d ca 40  	subw	s11, s5, a2
   1a5fa: 66 89        	mv	s2, s9
   1a5fc: 32 8b        	mv	s6, a2
   1a5fe: cd b3        	j	0x1a3e0 <_svfprintf_r+0x604>
   1a600: 05 47        	addi	a4, zero, 1
   1a602: 63 94 e7 00  	bne	a5, a4, 0x1a60a <_svfprintf_r+0x82e>
   1a606: 6f 10 20 13  	j	0x1b738 <_svfprintf_r+0x195c>
   1a60a: 09 47        	addi	a4, zero, 2
   1a60c: ca 8c        	mv	s9, s2
   1a60e: e3 9b e7 fa  	bne	a5, a4, 0x1a5c4 <_svfprintf_r+0x7e8>
   1a612: 26 77        	ld	a4, 104(sp)
   1a614: 93 0a c1 17  	addi	s5, sp, 380
   1a618: 56 8b        	mv	s6, s5
   1a61a: 93 f7 fb 00  	andi	a5, s7, 15
   1a61e: ba 97        	add	a5, a5, a4
   1a620: 83 c7 07 00  	lbu	a5, 0(a5)
   1a624: 93 db 4b 00  	srli	s7, s7, 4
   1a628: 7d 1b        	addi	s6, s6, -1
   1a62a: 23 00 fb 00  	sb	a5, 0(s6)
   1a62e: e3 96 0b fe  	bnez	s7, 0x1a61a <_svfprintf_r+0x83e>
   1a632: bb 8d 6a 41  	subw	s11, s5, s6
   1a636: 66 89        	mv	s2, s9
   1a638: 65 b3        	j	0x1a3e0 <_svfprintf_r+0x604>
   1a63a: bb 0a 8a 41  	subw	s5, s4, s8
   1a63e: e3 57 50 99  	blez	s5, 0x19fcc <_svfprintf_r+0x1f0>
   1a642: 37 2f 02 00  	lui	t5, 34
   1a646: c1 45        	addi	a1, zero, 16
   1a648: 5e 87        	mv	a4, s7
   1a64a: 03 26 81 10  	lw	a2, 264(sp)
   1a64e: 93 0b 0f 54  	addi	s7, t5, 1344
   1a652: 63 d0 55 07  	bge	a1, s5, 0x1a6b2 <_svfprintf_r+0x8d6>
   1a656: 26 e9        	sd	s1, 144(sp)
   1a658: c1 4f        	addi	t6, zero, 16
   1a65a: d6 84        	mv	s1, s5
   1a65c: 9d 4c        	addi	s9, zero, 7
   1a65e: d2 8a        	mv	s5, s4
   1a660: 4a 8a        	mv	s4, s2
   1a662: 5e 89        	mv	s2, s7
   1a664: da 8b        	mv	s7, s6
   1a666: 1a 8b        	mv	s6, t1
   1a668: 21 a0        	j	0x1a670 <_svfprintf_r+0x894>
   1a66a: c1 34        	addiw	s1, s1, -16
   1a66c: 63 dc 9f 02  	bge	t6, s1, 0x1a6a4 <_svfprintf_r+0x8c8>
   1a670: 05 26        	addiw	a2, a2, 1
   1a672: 41 07        	addi	a4, a4, 16
   1a674: 23 30 24 01  	sd	s2, 0(s0)
   1a678: 23 34 f4 01  	sd	t6, 8(s0)
   1a67c: 3a ea        	sd	a4, 272(sp)
   1a67e: 23 24 c1 10  	sw	a2, 264(sp)
   1a682: 41 04        	addi	s0, s0, 16
   1a684: e3 d3 cc fe  	bge	s9, a2, 0x1a66a <_svfprintf_r+0x88e>
   1a688: 82 65        	ld	a1, 0(sp)
   1a68a: 10 02        	addi	a2, sp, 256
   1a68c: 4e 85        	mv	a0, s3
   1a68e: ef 20 10 7c  	jal	0x1d64e <__ssprint_r>
   1a692: 7d e1        	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a694: c1 4f        	addi	t6, zero, 16
   1a696: c1 34        	addiw	s1, s1, -16
   1a698: 52 67        	ld	a4, 272(sp)
   1a69a: 03 26 81 10  	lw	a2, 264(sp)
   1a69e: 00 03        	addi	s0, sp, 384
   1a6a0: e3 c8 9f fc  	blt	t6, s1, 0x1a670 <_svfprintf_r+0x894>
   1a6a4: 5a 83        	mv	t1, s6
   1a6a6: 5e 8b        	mv	s6, s7
   1a6a8: ca 8b        	mv	s7, s2
   1a6aa: 52 89        	mv	s2, s4
   1a6ac: 56 8a        	mv	s4, s5
   1a6ae: a6 8a        	mv	s5, s1
   1a6b0: ca 64        	ld	s1, 144(sp)
   1a6b2: 9b 05 16 00  	addiw	a1, a2, 1
   1a6b6: 56 97        	add	a4, a4, s5
   1a6b8: 23 30 74 01  	sd	s7, 0(s0)
   1a6bc: 23 34 54 01  	sd	s5, 8(s0)
   1a6c0: 3a ea        	sd	a4, 272(sp)
   1a6c2: 23 24 b1 10  	sw	a1, 264(sp)
   1a6c6: 1d 46        	addi	a2, zero, 7
   1a6c8: e3 42 b6 10  	blt	a2, a1, 0x1afcc <_svfprintf_r+0x11f0>
   1a6cc: bb 0c b3 41  	subw	s9, t1, s11
   1a6d0: 41 04        	addi	s0, s0, 16
   1a6d2: ba 8b        	mv	s7, a4
   1a6d4: e3 50 90 91  	blez	s9, 0x19fd4 <_svfprintf_r+0x1f8>
   1a6d8: 37 2f 02 00  	lui	t5, 34
   1a6dc: c1 45        	addi	a1, zero, 16
   1a6de: 5e 87        	mv	a4, s7
   1a6e0: 03 26 81 10  	lw	a2, 264(sp)
   1a6e4: 93 0b 0f 54  	addi	s7, t5, 1344
   1a6e8: 63 d1 95 07  	bge	a1, s9, 0x1a74a <_svfprintf_r+0x96e>
   1a6ec: a2 6b        	ld	s7, 8(sp)
   1a6ee: c1 48        	addi	a7, zero, 16
   1a6f0: 9d 4a        	addi	s5, zero, 7
   1a6f2: de 87        	mv	a5, s7
   1a6f4: da 8b        	mv	s7, s6
   1a6f6: 52 8b        	mv	s6, s4
   1a6f8: 4a 8a        	mv	s4, s2
   1a6fa: 26 89        	mv	s2, s1
   1a6fc: e6 84        	mv	s1, s9
   1a6fe: be 8c        	mv	s9, a5
   1a700: 21 a0        	j	0x1a708 <_svfprintf_r+0x92c>
   1a702: c1 34        	addiw	s1, s1, -16
   1a704: 63 dc 98 02  	bge	a7, s1, 0x1a73c <_svfprintf_r+0x960>
   1a708: a2 66        	ld	a3, 8(sp)
   1a70a: 05 26        	addiw	a2, a2, 1
   1a70c: 41 07        	addi	a4, a4, 16
   1a70e: 14 e0        	sd	a3, 0(s0)
   1a710: 23 34 14 01  	sd	a7, 8(s0)
   1a714: 3a ea        	sd	a4, 272(sp)
   1a716: 23 24 c1 10  	sw	a2, 264(sp)
   1a71a: 41 04        	addi	s0, s0, 16
   1a71c: e3 d3 ca fe  	bge	s5, a2, 0x1a702 <_svfprintf_r+0x926>
   1a720: 82 65        	ld	a1, 0(sp)
   1a722: 10 02        	addi	a2, sp, 256
   1a724: 4e 85        	mv	a0, s3
   1a726: ef 20 90 72  	jal	0x1d64e <__ssprint_r>
   1a72a: 39 e5        	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a72c: c1 48        	addi	a7, zero, 16
   1a72e: c1 34        	addiw	s1, s1, -16
   1a730: 52 67        	ld	a4, 272(sp)
   1a732: 03 26 81 10  	lw	a2, 264(sp)
   1a736: 00 03        	addi	s0, sp, 384
   1a738: e3 c8 98 fc  	blt	a7, s1, 0x1a708 <_svfprintf_r+0x92c>
   1a73c: e6 87        	mv	a5, s9
   1a73e: a6 8c        	mv	s9, s1
   1a740: ca 84        	mv	s1, s2
   1a742: 52 89        	mv	s2, s4
   1a744: 5a 8a        	mv	s4, s6
   1a746: 5e 8b        	mv	s6, s7
   1a748: be 8b        	mv	s7, a5
   1a74a: 9b 05 16 00  	addiw	a1, a2, 1
   1a74e: 66 97        	add	a4, a4, s9
   1a750: 23 30 74 01  	sd	s7, 0(s0)
   1a754: 23 34 94 01  	sd	s9, 8(s0)
   1a758: 3a ea        	sd	a4, 272(sp)
   1a75a: 23 24 b1 10  	sw	a1, 264(sp)
   1a75e: 1d 46        	addi	a2, zero, 7
   1a760: 63 42 b6 34  	blt	a2, a1, 0x1aaa4 <_svfprintf_r+0xcc8>
   1a764: 41 04        	addi	s0, s0, 16
   1a766: ba 8b        	mv	s7, a4
   1a768: b5 b0        	j	0x19fd4 <_svfprintf_r+0x1f8>
   1a76a: 82 65        	ld	a1, 0(sp)
   1a76c: 10 02        	addi	a2, sp, 256
   1a76e: 4e 85        	mv	a0, s3
   1a770: ef 20 f0 6d  	jal	0x1d64e <__ssprint_r>
   1a774: e3 06 05 8a  	beqz	a0, 0x1a020 <_svfprintf_r+0x244>
   1a778: e2 65        	ld	a1, 24(sp)
   1a77a: e3 85 05 8c  	beqz	a1, 0x1a044 <_svfprintf_r+0x268>
   1a77e: 46 65        	ld	a0, 80(sp)
   1a780: ef b0 6f 8e  	jal	0x15866 <_free_r>
   1a784: c1 b0        	j	0x1a044 <_svfprintf_r+0x268>
   1a786: b7 28 02 00  	lui	a7, 34
   1a78a: 41 45        	addi	a0, zero, 16
   1a78c: d2 65        	ld	a1, 272(sp)
   1a78e: 03 26 81 10  	lw	a2, 264(sp)
   1a792: 93 8b 08 55  	addi	s7, a7, 1360
   1a796: 63 5e e5 04  	bge	a0, a4, 0x1a7f2 <_svfprintf_r+0xa16>
   1a79a: 26 e9        	sd	s1, 144(sp)
   1a79c: a2 87        	mv	a5, s0
   1a79e: de 84        	mv	s1, s7
   1a7a0: 41 4f        	addi	t5, zero, 16
   1a7a2: da 8b        	mv	s7, s6
   1a7a4: 9d 43        	addi	t2, zero, 7
   1a7a6: 3a 84        	mv	s0, a4
   1a7a8: 1a 8b        	mv	s6, t1
   1a7aa: 21 a0        	j	0x1a7b2 <_svfprintf_r+0x9d6>
   1a7ac: 41 34        	addiw	s0, s0, -16
   1a7ae: 63 5c 8f 02  	bge	t5, s0, 0x1a7e6 <_svfprintf_r+0xa0a>
   1a7b2: 05 26        	addiw	a2, a2, 1
   1a7b4: c1 05        	addi	a1, a1, 16
   1a7b6: 84 e3        	sd	s1, 0(a5)
   1a7b8: 23 b4 e7 01  	sd	t5, 8(a5)
   1a7bc: 2e ea        	sd	a1, 272(sp)
   1a7be: 23 24 c1 10  	sw	a2, 264(sp)
   1a7c2: c1 07        	addi	a5, a5, 16
   1a7c4: e3 d4 c3 fe  	bge	t2, a2, 0x1a7ac <_svfprintf_r+0x9d0>
   1a7c8: 82 65        	ld	a1, 0(sp)
   1a7ca: 10 02        	addi	a2, sp, 256
   1a7cc: 4e 85        	mv	a0, s3
   1a7ce: ef 20 10 68  	jal	0x1d64e <__ssprint_r>
   1a7d2: 5d f1        	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a7d4: 41 4f        	addi	t5, zero, 16
   1a7d6: 41 34        	addiw	s0, s0, -16
   1a7d8: d2 65        	ld	a1, 272(sp)
   1a7da: 03 26 81 10  	lw	a2, 264(sp)
   1a7de: 1c 03        	addi	a5, sp, 384
   1a7e0: 9d 43        	addi	t2, zero, 7
   1a7e2: e3 48 8f fc  	blt	t5, s0, 0x1a7b2 <_svfprintf_r+0x9d6>
   1a7e6: 5a 83        	mv	t1, s6
   1a7e8: 5e 8b        	mv	s6, s7
   1a7ea: a6 8b        	mv	s7, s1
   1a7ec: ca 64        	ld	s1, 144(sp)
   1a7ee: 22 87        	mv	a4, s0
   1a7f0: 3e 84        	mv	s0, a5
   1a7f2: ba 95        	add	a1, a1, a4
   1a7f4: 05 26        	addiw	a2, a2, 1
   1a7f6: 18 e4        	sd	a4, 8(s0)
   1a7f8: 23 30 74 01  	sd	s7, 0(s0)
   1a7fc: 2e ea        	sd	a1, 272(sp)
   1a7fe: 23 24 c1 10  	sw	a2, 264(sp)
   1a802: 9b 05 06 00  	sext.w	a1, a2
   1a806: 1d 47        	addi	a4, zero, 7
   1a808: 41 04        	addi	s0, s0, 16
   1a80a: 63 54 b7 f6  	bge	a4, a1, 0x19f72 <_svfprintf_r+0x196>
   1a80e: 82 65        	ld	a1, 0(sp)
   1a810: 10 02        	addi	a2, sp, 256
   1a812: 4e 85        	mv	a0, s3
   1a814: 1a e9        	sd	t1, 144(sp)
   1a816: ef 20 90 63  	jal	0x1d64e <__ssprint_r>
   1a81a: 39 fd        	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a81c: 4a 63        	ld	t1, 144(sp)
   1a81e: 00 03        	addi	s0, sp, 384
   1a820: 6f f0 2f f5  	j	0x19f72 <_svfprintf_r+0x196>
   1a824: ca 8c        	mv	s9, s2
   1a826: 3a f0        	sd	a4, 32(sp)
   1a828: a5 47        	addi	a5, zero, 9
   1a82a: 63 e8 77 3d  	bltu	a5, s7, 0x1abfa <_svfprintf_r+0xe1e>
   1a82e: 9b 8b 0b 03  	addiw	s7, s7, 48
   1a832: a3 0d 71 17  	sb	s7, 379(sp)
   1a836: 66 89        	mv	s2, s9
   1a838: 85 4d        	addi	s11, zero, 1
   1a83a: 13 0b b1 17  	addi	s6, sp, 379
   1a83e: 4d b6        	j	0x1a3e0 <_svfprintf_r+0x604>
   1a840: 82 65        	ld	a1, 0(sp)
   1a842: 10 02        	addi	a2, sp, 256
   1a844: 4e 85        	mv	a0, s3
   1a846: 1a e9        	sd	t1, 144(sp)
   1a848: ef 20 70 60  	jal	0x1d64e <__ssprint_r>
   1a84c: 15 f5        	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a84e: d2 6b        	ld	s7, 272(sp)
   1a850: 4a 63        	ld	t1, 144(sp)
   1a852: 00 03        	addi	s0, sp, 384
   1a854: 6f f0 0f f7  	j	0x19fc4 <_svfprintf_r+0x1e8>
   1a858: b7 28 02 00  	lui	a7, 34
   1a85c: c1 46        	addi	a3, zero, 16
   1a85e: 83 27 81 10  	lw	a5, 264(sp)
   1a862: 93 8b 08 55  	addi	s7, a7, 1360
   1a866: 63 d3 26 05  	bge	a3, s2, 0x1a8ac <_svfprintf_r+0xad0>
   1a86a: 82 6a        	ld	s5, 0(sp)
   1a86c: c1 4c        	addi	s9, zero, 16
   1a86e: 1d 4d        	addi	s10, zero, 7
   1a870: 21 a0        	j	0x1a878 <_svfprintf_r+0xa9c>
   1a872: 41 39        	addiw	s2, s2, -16
   1a874: 63 dc 2c 03  	bge	s9, s2, 0x1a8ac <_svfprintf_r+0xad0>
   1a878: 85 27        	addiw	a5, a5, 1
   1a87a: 41 07        	addi	a4, a4, 16
   1a87c: 23 30 74 01  	sd	s7, 0(s0)
   1a880: 23 34 94 01  	sd	s9, 8(s0)
   1a884: 3a ea        	sd	a4, 272(sp)
   1a886: 23 24 f1 10  	sw	a5, 264(sp)
   1a88a: 41 04        	addi	s0, s0, 16
   1a88c: e3 53 fd fe  	bge	s10, a5, 0x1a872 <_svfprintf_r+0xa96>
   1a890: 10 02        	addi	a2, sp, 256
   1a892: d6 85        	mv	a1, s5
   1a894: 4e 85        	mv	a0, s3
   1a896: ef 20 90 5b  	jal	0x1d64e <__ssprint_r>
   1a89a: e3 1f 05 ec  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a89e: 41 39        	addiw	s2, s2, -16
   1a8a0: 52 67        	ld	a4, 272(sp)
   1a8a2: 83 27 81 10  	lw	a5, 264(sp)
   1a8a6: 00 03        	addi	s0, sp, 384
   1a8a8: e3 c8 2c fd  	blt	s9, s2, 0x1a878 <_svfprintf_r+0xa9c>
   1a8ac: 9b 86 17 00  	addiw	a3, a5, 1
   1a8b0: 4a 97        	add	a4, a4, s2
   1a8b2: 23 30 74 01  	sd	s7, 0(s0)
   1a8b6: 23 34 24 01  	sd	s2, 8(s0)
   1a8ba: 3a ea        	sd	a4, 272(sp)
   1a8bc: 23 24 d1 10  	sw	a3, 264(sp)
   1a8c0: 9d 47        	addi	a5, zero, 7
   1a8c2: 63 d6 d7 f4  	bge	a5, a3, 0x1a00e <_svfprintf_r+0x232>
   1a8c6: 82 65        	ld	a1, 0(sp)
   1a8c8: 10 02        	addi	a2, sp, 256
   1a8ca: 4e 85        	mv	a0, s3
   1a8cc: ef 20 30 58  	jal	0x1d64e <__ssprint_r>
   1a8d0: e3 14 05 ea  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a8d4: 52 67        	ld	a4, 272(sp)
   1a8d6: 6f f0 8f f3  	j	0x1a00e <_svfprintf_r+0x232>
   1a8da: 6e 46        	lw	a2, 216(sp)
   1a8dc: 63 5a c0 5e  	blez	a2, 0x1aed0 <_svfprintf_r+0x10f4>
   1a8e0: e2 77        	ld	a5, 56(sp)
   1a8e2: 42 77        	ld	a4, 48(sp)
   1a8e4: b3 0a fb 00  	add	s5, s6, a5
   1a8e8: be 86        	mv	a3, a5
   1a8ea: 63 4e f7 2e  	blt	a4, a5, 0x1abe6 <_svfprintf_r+0xe0a>
   1a8ee: 1b 8d 06 00  	sext.w	s10, a3
   1a8f2: 63 52 a0 03  	blez	s10, 0x1a916 <_svfprintf_r+0xb3a>
   1a8f6: 03 26 81 10  	lw	a2, 264(sp)
   1a8fa: ea 9b        	add	s7, s7, s10
   1a8fc: 23 30 64 01  	sd	s6, 0(s0)
   1a900: 9b 05 16 00  	addiw	a1, a2, 1
   1a904: 23 34 a4 01  	sd	s10, 8(s0)
   1a908: 5e ea        	sd	s7, 272(sp)
   1a90a: 23 24 b1 10  	sw	a1, 264(sp)
   1a90e: 1d 46        	addi	a2, zero, 7
   1a910: 41 04        	addi	s0, s0, 16
   1a912: e3 44 b6 08  	blt	a2, a1, 0x1b19a <_svfprintf_r+0x13be>
   1a916: 13 46 fd ff  	not	a2, s10
   1a91a: c2 77        	ld	a5, 48(sp)
   1a91c: 7d 96        	srai	a2, a2, 63
   1a91e: b3 76 cd 00  	and	a3, s10, a2
   1a922: 3b 8d d7 40  	subw	s10, a5, a3
   1a926: 63 41 a0 3f  	bgtz	s10, 0x1ad08 <_svfprintf_r+0xf2c>
   1a92a: c2 77        	ld	a5, 48(sp)
   1a92c: 93 76 09 40  	andi	a3, s2, 1024
   1a930: 3e 9b        	add	s6, s6, a5
   1a932: 63 9c 06 42  	bnez	a3, 0x1ad6a <_svfprintf_r+0xf8e>
   1a936: ee 46        	lw	a3, 216(sp)
   1a938: e2 77        	ld	a5, 56(sp)
   1a93a: 63 c6 f6 00  	blt	a3, a5, 0x1a946 <_svfprintf_r+0xb6a>
   1a93e: 13 76 19 00  	andi	a2, s2, 1
   1a942: e3 09 06 06  	beqz	a2, 0x1b1b4 <_svfprintf_r+0x13d8>
   1a946: 03 26 81 10  	lw	a2, 264(sp)
   1a94a: e6 67        	ld	a5, 88(sp)
   1a94c: 86 75        	ld	a1, 96(sp)
   1a94e: 41 04        	addi	s0, s0, 16
   1a950: 33 87 77 01  	add	a4, a5, s7
   1a954: 23 38 b4 fe  	sd	a1, -16(s0)
   1a958: 9b 05 16 00  	addiw	a1, a2, 1
   1a95c: 23 3c f4 fe  	sd	a5, -8(s0)
   1a960: 3a ea        	sd	a4, 272(sp)
   1a962: 23 24 b1 10  	sw	a1, 264(sp)
   1a966: 1d 46        	addi	a2, zero, 7
   1a968: e3 47 b6 52  	blt	a2, a1, 0x1b696 <_svfprintf_r+0x18ba>
   1a96c: e2 77        	ld	a5, 56(sp)
   1a96e: b3 8c 6a 41  	sub	s9, s5, s6
   1a972: 9b 85 0c 00  	sext.w	a1, s9
   1a976: 3b 86 d7 40  	subw	a2, a5, a3
   1a97a: 63 53 b6 00  	bge	a2, a1, 0x1a980 <_svfprintf_r+0xba4>
   1a97e: b2 8c        	mv	s9, a2
   1a980: 81 2c        	sext.w	s9, s9
   1a982: 63 51 90 03  	blez	s9, 0x1a9a4 <_svfprintf_r+0xbc8>
   1a986: 83 26 81 10  	lw	a3, 264(sp)
   1a98a: 66 97        	add	a4, a4, s9
   1a98c: 23 30 64 01  	sd	s6, 0(s0)
   1a990: 85 26        	addiw	a3, a3, 1
   1a992: 23 34 94 01  	sd	s9, 8(s0)
   1a996: 3a ea        	sd	a4, 272(sp)
   1a998: 23 24 d1 10  	sw	a3, 264(sp)
   1a99c: 9d 47        	addi	a5, zero, 7
   1a99e: 41 04        	addi	s0, s0, 16
   1a9a0: e3 c4 d7 56  	blt	a5, a3, 0x1b708 <_svfprintf_r+0x192c>
   1a9a4: 93 c7 fc ff  	not	a5, s9
   1a9a8: fd 97        	srai	a5, a5, 63
   1a9aa: b3 fc fc 00  	and	s9, s9, a5
   1a9ae: bb 0c 96 41  	subw	s9, a2, s9
   1a9b2: 63 56 90 e5  	blez	s9, 0x19ffe <_svfprintf_r+0x222>
   1a9b6: 37 2f 02 00  	lui	t5, 34
   1a9ba: c1 46        	addi	a3, zero, 16
   1a9bc: 83 27 81 10  	lw	a5, 264(sp)
   1a9c0: 93 0b 0f 54  	addi	s7, t5, 1344
   1a9c4: 63 d3 96 5d  	bge	a3, s9, 0x1af8a <_svfprintf_r+0x11ae>
   1a9c8: 82 6a        	ld	s5, 0(sp)
   1a9ca: 41 4d        	addi	s10, zero, 16
   1a9cc: 9d 4d        	addi	s11, zero, 7
   1a9ce: 21 a0        	j	0x1a9d6 <_svfprintf_r+0xbfa>
   1a9d0: c1 3c        	addiw	s9, s9, -16
   1a9d2: 63 5c 9d 5b  	bge	s10, s9, 0x1af8a <_svfprintf_r+0x11ae>
   1a9d6: 85 27        	addiw	a5, a5, 1
   1a9d8: 41 07        	addi	a4, a4, 16
   1a9da: 23 30 74 01  	sd	s7, 0(s0)
   1a9de: 23 34 a4 01  	sd	s10, 8(s0)
   1a9e2: 3a ea        	sd	a4, 272(sp)
   1a9e4: 23 24 f1 10  	sw	a5, 264(sp)
   1a9e8: 41 04        	addi	s0, s0, 16
   1a9ea: e3 d3 fd fe  	bge	s11, a5, 0x1a9d0 <_svfprintf_r+0xbf4>
   1a9ee: 10 02        	addi	a2, sp, 256
   1a9f0: d6 85        	mv	a1, s5
   1a9f2: 4e 85        	mv	a0, s3
   1a9f4: ef 20 b0 45  	jal	0x1d64e <__ssprint_r>
   1a9f8: e3 10 05 d8  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1a9fc: 52 67        	ld	a4, 272(sp)
   1a9fe: 83 27 81 10  	lw	a5, 264(sp)
   1aa02: 00 03        	addi	s0, sp, 384
   1aa04: f1 b7        	j	0x1a9d0 <_svfprintf_r+0xbf4>
   1aa06: 13 76 19 00  	andi	a2, s2, 1
   1aa0a: 63 17 06 f8  	bnez	a2, 0x1a198 <_svfprintf_r+0x3bc>
   1aa0e: 03 26 81 10  	lw	a2, 264(sp)
   1aa12: 13 8d 1b 00  	addi	s10, s7, 1
   1aa16: 23 30 64 01  	sd	s6, 0(s0)
   1aa1a: 9b 07 16 00  	addiw	a5, a2, 1
   1aa1e: 14 e4        	sd	a3, 8(s0)
   1aa20: 23 24 f1 10  	sw	a5, 264(sp)
   1aa24: 1b 84 07 00  	sext.w	s0, a5
   1aa28: 6a ea        	sd	s10, 272(sp)
   1aa2a: 9d 47        	addi	a5, zero, 7
   1aa2c: 63 d2 87 fe  	bge	a5, s0, 0x1a210 <_svfprintf_r+0x434>
   1aa30: 82 65        	ld	a1, 0(sp)
   1aa32: 10 02        	addi	a2, sp, 256
   1aa34: 4e 85        	mv	a0, s3
   1aa36: ef 20 90 41  	jal	0x1d64e <__ssprint_r>
   1aa3a: e3 1f 05 d2  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1aa3e: 52 6d        	ld	s10, 272(sp)
   1aa40: 03 24 81 10  	lw	s0, 264(sp)
   1aa44: 93 0c 01 18  	addi	s9, sp, 384
   1aa48: 6f f0 8f fc  	j	0x1a210 <_svfprintf_r+0x434>
   1aa4c: 62 77        	ld	a4, 56(sp)
   1aa4e: 85 47        	addi	a5, zero, 1
   1aa50: 63 d0 e7 fc  	bge	a5, a4, 0x1a210 <_svfprintf_r+0x434>
   1aa54: 37 2f 02 00  	lui	t5, 34
   1aa58: c5 47        	addi	a5, zero, 17
   1aa5a: 93 0b 0f 54  	addi	s7, t5, 1344
   1aa5e: 63 d2 e7 5e  	bge	a5, a4, 0x1b042 <_svfprintf_r+0x1266>
   1aa62: 82 6a        	ld	s5, 0(sp)
   1aa64: 41 47        	addi	a4, zero, 16
   1aa66: 1d 4b        	addi	s6, zero, 7
   1aa68: 21 a0        	j	0x1aa70 <_svfprintf_r+0xc94>
   1aa6a: c1 3d        	addiw	s11, s11, -16
   1aa6c: 63 5b b7 5d  	bge	a4, s11, 0x1b042 <_svfprintf_r+0x1266>
   1aa70: 05 24        	addiw	s0, s0, 1
   1aa72: 41 0d        	addi	s10, s10, 16
   1aa74: 23 b0 7c 01  	sd	s7, 0(s9)
   1aa78: 23 b4 ec 00  	sd	a4, 8(s9)
   1aa7c: 6a ea        	sd	s10, 272(sp)
   1aa7e: 23 24 81 10  	sw	s0, 264(sp)
   1aa82: c1 0c        	addi	s9, s9, 16
   1aa84: e3 53 8b fe  	bge	s6, s0, 0x1aa6a <_svfprintf_r+0xc8e>
   1aa88: 10 02        	addi	a2, sp, 256
   1aa8a: d6 85        	mv	a1, s5
   1aa8c: 4e 85        	mv	a0, s3
   1aa8e: ef 20 10 3c  	jal	0x1d64e <__ssprint_r>
   1aa92: e3 13 05 ce  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1aa96: 52 6d        	ld	s10, 272(sp)
   1aa98: 03 24 81 10  	lw	s0, 264(sp)
   1aa9c: 93 0c 01 18  	addi	s9, sp, 384
   1aaa0: 41 47        	addi	a4, zero, 16
   1aaa2: e1 b7        	j	0x1aa6a <_svfprintf_r+0xc8e>
   1aaa4: 82 65        	ld	a1, 0(sp)
   1aaa6: 10 02        	addi	a2, sp, 256
   1aaa8: 4e 85        	mv	a0, s3
   1aaaa: ef 20 50 3a  	jal	0x1d64e <__ssprint_r>
   1aaae: e3 15 05 cc  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1aab2: d2 6b        	ld	s7, 272(sp)
   1aab4: 00 03        	addi	s0, sp, 384
   1aab6: 6f f0 ef d1  	j	0x19fd4 <_svfprintf_r+0x1f8>
   1aaba: 93 77 09 01  	andi	a5, s2, 16
   1aabe: e5 e7        	bnez	a5, 0x1aba6 <_svfprintf_r+0xdca>
   1aac0: 82 76        	ld	a3, 32(sp)
   1aac2: 93 77 09 04  	andi	a5, s2, 64
   1aac6: 83 ab 06 00  	lw	s7, 0(a3)
   1aaca: 63 84 07 60  	beqz	a5, 0x1b0d2 <_svfprintf_r+0x12f6>
   1aace: c2 1b        	slli	s7, s7, 48
   1aad0: 93 db 0b 03  	srli	s7, s7, 48
   1aad4: 49 ba        	j	0x1a466 <_svfprintf_r+0x68a>
   1aad6: 93 f7 0c 01  	andi	a5, s9, 16
   1aada: dd ef        	bnez	a5, 0x1ab98 <_svfprintf_r+0xdbc>
   1aadc: 82 76        	ld	a3, 32(sp)
   1aade: 93 f7 0c 04  	andi	a5, s9, 64
   1aae2: 83 ab 06 00  	lw	s7, 0(a3)
   1aae6: 63 8e 07 5e  	beqz	a5, 0x1b0e2 <_svfprintf_r+0x1306>
   1aaea: c2 1b        	slli	s7, s7, 48
   1aaec: 93 db 0b 03  	srli	s7, s7, 48
   1aaf0: 3a f0        	sd	a4, 32(sp)
   1aaf2: 85 47        	addi	a5, zero, 1
   1aaf4: c9 b0        	j	0x1a3b6 <_svfprintf_r+0x5da>
   1aaf6: 93 77 09 01  	andi	a5, s2, 16
   1aafa: d9 eb        	bnez	a5, 0x1ab90 <_svfprintf_r+0xdb4>
   1aafc: 82 76        	ld	a3, 32(sp)
   1aafe: 93 77 09 04  	andi	a5, s2, 64
   1ab02: 83 ab 06 00  	lw	s7, 0(a3)
   1ab06: 63 88 07 5e  	beqz	a5, 0x1b0f6 <_svfprintf_r+0x131a>
   1ab0a: 9b 9b 0b 01  	slliw	s7, s7, 16
   1ab0e: 9b db 0b 41  	sraiw	s7, s7, 16
   1ab12: de 87        	mv	a5, s7
   1ab14: e3 da 07 90  	bgez	a5, 0x1a428 <_svfprintf_r+0x64c>
   1ab18: 3a f0        	sd	a4, 32(sp)
   1ab1a: b3 0b 70 41  	neg	s7, s7
   1ab1e: ca 8c        	mv	s9, s2
   1ab20: 13 07 d0 02  	addi	a4, zero, 45
   1ab24: 85 47        	addi	a5, zero, 1
   1ab26: 49 b8        	j	0x1a3b8 <_svfprintf_r+0x5dc>
   1ab28: 81 4d        	mv	s11, zero
   1ab2a: 13 0b c1 17  	addi	s6, sp, 380
   1ab2e: 4d b8        	j	0x1a3e0 <_svfprintf_r+0x604>
   1ab30: 82 65        	ld	a1, 0(sp)
   1ab32: 10 02        	addi	a2, sp, 256
   1ab34: 4e 85        	mv	a0, s3
   1ab36: ef 20 90 31  	jal	0x1d64e <__ssprint_r>
   1ab3a: e3 1f 05 c2  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1ab3e: 52 67        	ld	a4, 272(sp)
   1ab40: 83 26 81 10  	lw	a3, 264(sp)
   1ab44: 93 0c 01 18  	addi	s9, sp, 384
   1ab48: 6f f0 2f e7  	j	0x1a1ba <_svfprintf_r+0x3de>
   1ab4c: 82 65        	ld	a1, 0(sp)
   1ab4e: 10 02        	addi	a2, sp, 256
   1ab50: 4e 85        	mv	a0, s3
   1ab52: ef 20 d0 2f  	jal	0x1d64e <__ssprint_r>
   1ab56: e3 11 05 c2  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1ab5a: 52 6d        	ld	s10, 272(sp)
   1ab5c: 03 24 81 10  	lw	s0, 264(sp)
   1ab60: 93 0c 01 18  	addi	s9, sp, 384
   1ab64: 6f f0 8f e7  	j	0x1a1dc <_svfprintf_r+0x400>
   1ab68: 93 77 19 00  	andi	a5, s2, 1
   1ab6c: 5e 87        	mv	a4, s7
   1ab6e: 63 88 07 c8  	beqz	a5, 0x19ffe <_svfprintf_r+0x222>
   1ab72: d9 b2        	j	0x1a538 <_svfprintf_r+0x75c>
   1ab74: 56 83        	mv	t1, s5
   1ab76: 71 b8        	j	0x1a412 <_svfprintf_r+0x636>
   1ab78: 03 ce 04 00  	lbu	t3, 0(s1)
   1ab7c: 3e f0        	sd	a5, 32(sp)
   1ab7e: 6f f0 4f b6  	j	0x19ee2 <_svfprintf_r+0x106>
   1ab82: 93 07 00 03  	addi	a5, zero, 48
   1ab86: a3 0d f1 16  	sb	a5, 379(sp)
   1ab8a: 13 0b b1 17  	addi	s6, sp, 379
   1ab8e: 89 b8        	j	0x1a3e0 <_svfprintf_r+0x604>
   1ab90: 82 77        	ld	a5, 32(sp)
   1ab92: 9c 63        	ld	a5, 0(a5)
   1ab94: be 8b        	mv	s7, a5
   1ab96: 79 b0        	j	0x1a424 <_svfprintf_r+0x648>
   1ab98: 82 77        	ld	a5, 32(sp)
   1ab9a: 3a f0        	sd	a4, 32(sp)
   1ab9c: 83 bb 07 00  	ld	s7, 0(a5)
   1aba0: 85 47        	addi	a5, zero, 1
   1aba2: 6f f0 5f 81  	j	0x1a3b6 <_svfprintf_r+0x5da>
   1aba6: 82 77        	ld	a5, 32(sp)
   1aba8: 83 bb 07 00  	ld	s7, 0(a5)
   1abac: 6d b8        	j	0x1a466 <_svfprintf_r+0x68a>
   1abae: 93 77 09 01  	andi	a5, s2, 16
   1abb2: e3 9c 07 8c  	bnez	a5, 0x1a48a <_svfprintf_r+0x6ae>
   1abb6: 93 77 09 04  	andi	a5, s2, 64
   1abba: 63 9d 07 78  	bnez	a5, 0x1b354 <_svfprintf_r+0x1578>
   1abbe: 13 79 09 20  	andi	s2, s2, 512
   1abc2: c2 67        	ld	a5, 16(sp)
   1abc4: e3 04 09 48  	beqz	s2, 0x1b84c <_svfprintf_r+0x1a70>
   1abc8: 23 00 f7 00  	sb	a5, 0(a4)
   1abcc: c9 b0        	j	0x1a48e <_svfprintf_r+0x6b2>
   1abce: 82 77        	ld	a5, 32(sp)
   1abd0: 56 ec        	sd	s5, 24(sp)
   1abd2: 88 23        	fld	fa0, 0(a5)
   1abd4: 93 8a 87 00  	addi	s5, a5, 8
   1abd8: 56 f0        	sd	s5, 32(sp)
   1abda: ef 50 00 1a  	jal	0x1fd7a <__extenddftf2>
   1abde: 62 63        	ld	t1, 24(sp)
   1abe0: 2a 87        	mv	a4, a0
   1abe2: 6f f0 af cf  	j	0x1a0dc <_svfprintf_r+0x300>
   1abe6: ba 86        	mv	a3, a4
   1abe8: 1b 8d 06 00  	sext.w	s10, a3
   1abec: e3 45 a0 d1  	bgtz	s10, 0x1a8f6 <_svfprintf_r+0xb1a>
   1abf0: 1d b3        	j	0x1a916 <_svfprintf_r+0xb3a>
   1abf2: 3a f0        	sd	a4, 32(sp)
   1abf4: a5 47        	addi	a5, zero, 9
   1abf6: e3 fc 77 c3  	bgeu	a5, s7, 0x1a82e <_svfprintf_r+0xa52>
   1abfa: 93 0a c1 17  	addi	s5, sp, 380
   1abfe: 66 78        	ld	a6, 120(sp)
   1ac00: 93 f7 0c 40  	andi	a5, s9, 1024
   1ac04: 81 46        	mv	a3, zero
   1ac06: d6 85        	mv	a1, s5
   1ac08: 29 49        	addi	s2, zero, 10
   1ac0a: a5 4d        	addi	s11, zero, 9
   1ac0c: 13 0c f0 0f  	addi	s8, zero, 255
   1ac10: 39 a0        	j	0x1ac1e <_svfprintf_r+0xe42>
   1ac12: 33 d7 2b 03  	divu	a4, s7, s2
   1ac16: 63 f6 7d 03  	bgeu	s11, s7, 0x1ac42 <_svfprintf_r+0xe66>
   1ac1a: ba 8b        	mv	s7, a4
   1ac1c: da 85        	mv	a1, s6
   1ac1e: 13 8b f5 ff  	addi	s6, a1, -1
   1ac22: 85 26        	addiw	a3, a3, 1
   1ac24: 33 f7 2b 03  	remu	a4, s7, s2
   1ac28: 1b 07 07 03  	addiw	a4, a4, 48
   1ac2c: a3 8f e5 fe  	sb	a4, -1(a1)
   1ac30: ed d3        	beqz	a5, 0x1ac12 <_svfprintf_r+0xe36>
   1ac32: 03 47 08 00  	lbu	a4, 0(a6)
   1ac36: e3 9e e6 fc  	bne	a3, a4, 0x1ac12 <_svfprintf_r+0xe36>
   1ac3a: e3 0c 87 fd  	beq	a4, s8, 0x1ac12 <_svfprintf_r+0xe36>
   1ac3e: 63 e1 7d 53  	bltu	s11, s7, 0x1b160 <_svfprintf_r+0x1384>
   1ac42: c2 fc        	sd	a6, 120(sp)
   1ac44: 36 fc        	sd	a3, 56(sp)
   1ac46: bb 8d 6a 41  	subw	s11, s5, s6
   1ac4a: 66 89        	mv	s2, s9
   1ac4c: 6f f0 4f f9  	j	0x1a3e0 <_svfprintf_r+0x604>
   1ac50: b7 27 02 00  	lui	a5, 34
   1ac54: 93 87 07 bf  	addi	a5, a5, -1040
   1ac58: 56 83        	mv	t1, s5
   1ac5a: be f4        	sd	a5, 104(sp)
   1ac5c: 82 76        	ld	a3, 32(sp)
   1ac5e: 93 77 09 02  	andi	a5, s2, 32
   1ac62: 13 87 86 00  	addi	a4, a3, 8
   1ac66: 63 82 07 24  	beqz	a5, 0x1aeaa <_svfprintf_r+0x10ce>
   1ac6a: 83 bb 06 00  	ld	s7, 0(a3)
   1ac6e: 93 77 19 00  	andi	a5, s2, 1
   1ac72: 81 cf        	beqz	a5, 0x1ac8a <_svfprintf_r+0xeae>
   1ac74: 63 8b 0b 00  	beqz	s7, 0x1ac8a <_svfprintf_r+0xeae>
   1ac78: 13 69 29 00  	ori	s2, s2, 2
   1ac7c: 93 07 00 03  	addi	a5, zero, 48
   1ac80: 23 08 f1 0c  	sb	a5, 208(sp)
   1ac84: a3 08 a1 0d  	sb	s10, 209(sp)
   1ac88: 01 29        	sext.w	s2, s2
   1ac8a: 93 7c f9 bf  	andi	s9, s2, -1025
   1ac8e: 81 2c        	sext.w	s9, s9
   1ac90: 3a f0        	sd	a4, 32(sp)
   1ac92: 89 47        	addi	a5, zero, 2
   1ac94: 6f f0 2f f2  	j	0x1a3b6 <_svfprintf_r+0x5da>
   1ac98: 56 83        	mv	t1, s5
   1ac9a: ca 8c        	mv	s9, s2
   1ac9c: 6f f0 4f f0  	j	0x1a3a0 <_svfprintf_r+0x5c4>
   1aca0: 82 65        	ld	a1, 0(sp)
   1aca2: 10 02        	addi	a2, sp, 256
   1aca4: 4e 85        	mv	a0, s3
   1aca6: ef 20 90 1a  	jal	0x1d64e <__ssprint_r>
   1acaa: e3 17 05 ac  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1acae: 52 67        	ld	a4, 272(sp)
   1acb0: 00 03        	addi	s0, sp, 384
   1acb2: 6d b0        	j	0x1a55c <_svfprintf_r+0x780>
   1acb4: b7 27 02 00  	lui	a5, 34
   1acb8: 93 87 87 c0  	addi	a5, a5, -1016
   1acbc: 56 83        	mv	t1, s5
   1acbe: be f4        	sd	a5, 104(sp)
   1acc0: 71 bf        	j	0x1ac5c <_svfprintf_r+0xe80>
   1acc2: 56 83        	mv	t1, s5
   1acc4: 6f f0 0f f9  	j	0x1a454 <_svfprintf_r+0x678>
   1acc8: 13 69 09 20  	ori	s2, s2, 512
   1accc: 03 ce 14 00  	lbu	t3, 1(s1)
   1acd0: 01 29        	sext.w	s2, s2
   1acd2: 85 04        	addi	s1, s1, 1
   1acd4: 6f f0 ef a0  	j	0x19ee2 <_svfprintf_r+0x106>
   1acd8: 13 69 09 02  	ori	s2, s2, 32
   1acdc: 03 ce 14 00  	lbu	t3, 1(s1)
   1ace0: 01 29        	sext.w	s2, s2
   1ace2: 85 04        	addi	s1, s1, 1
   1ace4: 6f f0 ef 9f  	j	0x19ee2 <_svfprintf_r+0x106>
   1ace8: 46 65        	ld	a0, 80(sp)
   1acea: 93 05 00 04  	addi	a1, zero, 64
   1acee: ef d0 3f 84  	jal	0x18530 <_malloc_r>
   1acf2: 82 67        	ld	a5, 0(sp)
   1acf4: 88 e3        	sd	a0, 0(a5)
   1acf6: 88 ef        	sd	a0, 24(a5)
   1acf8: e3 00 05 34  	beqz	a0, 0x1b838 <_svfprintf_r+0x1a5c>
   1acfc: 82 67        	ld	a5, 0(sp)
   1acfe: 13 07 00 04  	addi	a4, zero, 64
   1ad02: 98 d3        	sw	a4, 32(a5)
   1ad04: 6f f0 ef 93  	j	0x19e42 <_svfprintf_r+0x66>
   1ad08: 37 2f 02 00  	lui	t5, 34
   1ad0c: c1 45        	addi	a1, zero, 16
   1ad0e: 5e 87        	mv	a4, s7
   1ad10: 03 26 81 10  	lw	a2, 264(sp)
   1ad14: 93 0b 0f 54  	addi	s7, t5, 1344
   1ad18: 63 d3 a5 43  	bge	a1, s10, 0x1b13e <_svfprintf_r+0x1362>
   1ad1c: 82 6c        	ld	s9, 0(sp)
   1ad1e: c1 48        	addi	a7, zero, 16
   1ad20: 9d 4d        	addi	s11, zero, 7
   1ad22: 21 a0        	j	0x1ad2a <_svfprintf_r+0xf4e>
   1ad24: 41 3d        	addiw	s10, s10, -16
   1ad26: 63 dc a8 41  	bge	a7, s10, 0x1b13e <_svfprintf_r+0x1362>
   1ad2a: 05 26        	addiw	a2, a2, 1
   1ad2c: 41 07        	addi	a4, a4, 16
   1ad2e: 23 30 74 01  	sd	s7, 0(s0)
   1ad32: 23 34 14 01  	sd	a7, 8(s0)
   1ad36: 3a ea        	sd	a4, 272(sp)
   1ad38: 23 24 c1 10  	sw	a2, 264(sp)
   1ad3c: 41 04        	addi	s0, s0, 16
   1ad3e: e3 d3 cd fe  	bge	s11, a2, 0x1ad24 <_svfprintf_r+0xf48>
   1ad42: 10 02        	addi	a2, sp, 256
   1ad44: e6 85        	mv	a1, s9
   1ad46: 4e 85        	mv	a0, s3
   1ad48: ef 20 70 10  	jal	0x1d64e <__ssprint_r>
   1ad4c: e3 16 05 a2  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1ad50: 52 67        	ld	a4, 272(sp)
   1ad52: 03 26 81 10  	lw	a2, 264(sp)
   1ad56: 00 03        	addi	s0, sp, 384
   1ad58: c1 48        	addi	a7, zero, 16
   1ad5a: e9 b7        	j	0x1ad24 <_svfprintf_r+0xf48>
   1ad5c: 5a 85        	mv	a0, s6
   1ad5e: ef e0 1f f6  	jal	0x19cbe <strlen>
   1ad62: 9b 0d 05 00  	sext.w	s11, a0
   1ad66: 6f f0 4f d3  	j	0x1a29a <_svfprintf_r+0x4be>
   1ad6a: 06 6d        	ld	s10, 64(sp)
   1ad6c: 37 2e 02 00  	lui	t3, 34
   1ad70: 26 e9        	sd	s1, 144(sp)
   1ad72: 4a ed        	sd	s2, 152(sp)
   1ad74: 52 f1        	sd	s4, 160(sp)
   1ad76: e2 e0        	sd	s8, 64(sp)
   1ad78: 5a 8a        	mv	s4, s6
   1ad7a: 66 79        	ld	s2, 120(sp)
   1ad7c: aa 64        	ld	s1, 136(sp)
   1ad7e: 0a 6b        	ld	s6, 128(sp)
   1ad80: 26 6c        	ld	s8, 72(sp)
   1ad82: 82 6c        	ld	s9, 0(sp)
   1ad84: 1d 43        	addi	t1, zero, 7
   1ad86: c1 48        	addi	a7, zero, 16
   1ad88: 93 0d 0e 54  	addi	s11, t3, 1344
   1ad8c: 5e 87        	mv	a4, s7
   1ad8e: 63 00 0d 08  	beqz	s10, 0x1ae0e <_svfprintf_r+0x1032>
   1ad92: 63 09 0c 10  	beqz	s8, 0x1aea4 <_svfprintf_r+0x10c8>
   1ad96: 7d 3c        	addiw	s8, s8, -1
   1ad98: 83 27 81 10  	lw	a5, 264(sp)
   1ad9c: 5a 97        	add	a4, a4, s6
   1ad9e: 04 e0        	sd	s1, 0(s0)
   1ada0: 9b 86 17 00  	addiw	a3, a5, 1
   1ada4: 23 34 64 01  	sd	s6, 8(s0)
   1ada8: 3a ea        	sd	a4, 272(sp)
   1adaa: 23 24 d1 10  	sw	a3, 264(sp)
   1adae: 41 04        	addi	s0, s0, 16
   1adb0: 63 44 d3 10  	blt	t1, a3, 0x1aeb8 <_svfprintf_r+0x10dc>
   1adb4: 83 46 09 00  	lbu	a3, 0(s2)
   1adb8: b3 8b 4a 41  	sub	s7, s5, s4
   1adbc: 9b 87 0b 00  	sext.w	a5, s7
   1adc0: 1b 86 06 00  	sext.w	a2, a3
   1adc4: 63 d3 f6 00  	bge	a3, a5, 0x1adca <_svfprintf_r+0xfee>
   1adc8: b2 8b        	mv	s7, a2
   1adca: 81 2b        	sext.w	s7, s7
   1adcc: 63 55 70 03  	blez	s7, 0x1adf6 <_svfprintf_r+0x101a>
   1add0: 83 27 81 10  	lw	a5, 264(sp)
   1add4: 5e 97        	add	a4, a4, s7
   1add6: 23 30 44 01  	sd	s4, 0(s0)
   1adda: 9b 86 17 00  	addiw	a3, a5, 1
   1adde: 23 34 74 01  	sd	s7, 8(s0)
   1ade2: 3a ea        	sd	a4, 272(sp)
   1ade4: 23 24 d1 10  	sw	a3, 264(sp)
   1ade8: 63 42 d3 1c  	blt	t1, a3, 0x1afac <_svfprintf_r+0x11d0>
   1adec: 83 46 09 00  	lbu	a3, 0(s2)
   1adf0: 41 04        	addi	s0, s0, 16
   1adf2: 1b 86 06 00  	sext.w	a2, a3
   1adf6: 93 c7 fb ff  	not	a5, s7
   1adfa: fd 97        	srai	a5, a5, 63
   1adfc: b3 f7 fb 00  	and	a5, s7, a5
   1ae00: bb 0b f6 40  	subw	s7, a2, a5
   1ae04: 63 42 70 03  	bgtz	s7, 0x1ae28 <_svfprintf_r+0x104c>
   1ae08: 36 9a        	add	s4, s4, a3
   1ae0a: e3 14 0d f8  	bnez	s10, 0x1ad92 <_svfprintf_r+0xfb6>
   1ae0e: e3 14 0c f8  	bnez	s8, 0x1ad96 <_svfprintf_r+0xfba>
   1ae12: ca fc        	sd	s2, 120(sp)
   1ae14: 52 8b        	mv	s6, s4
   1ae16: ca 64        	ld	s1, 144(sp)
   1ae18: 6a 69        	ld	s2, 152(sp)
   1ae1a: 0a 7a        	ld	s4, 160(sp)
   1ae1c: 06 6c        	ld	s8, 64(sp)
   1ae1e: ba 8b        	mv	s7, a4
   1ae20: e3 fb 6a b1  	bgeu	s5, s6, 0x1a936 <_svfprintf_r+0xb5a>
   1ae24: 56 8b        	mv	s6, s5
   1ae26: 01 be        	j	0x1a936 <_svfprintf_r+0xb5a>
   1ae28: 37 26 02 00  	lui	a2, 34
   1ae2c: 83 27 81 10  	lw	a5, 264(sp)
   1ae30: 13 06 06 54  	addi	a2, a2, 1344
   1ae34: 63 d8 78 05  	bge	a7, s7, 0x1ae84 <_svfprintf_r+0x10a8>
   1ae38: 26 f8        	sd	s1, 48(sp)
   1ae3a: de 84        	mv	s1, s7
   1ae3c: ee 8b        	mv	s7, s11
   1ae3e: 21 a0        	j	0x1ae46 <_svfprintf_r+0x106a>
   1ae40: c1 34        	addiw	s1, s1, -16
   1ae42: 63 de 98 02  	bge	a7, s1, 0x1ae7e <_svfprintf_r+0x10a2>
   1ae46: 85 27        	addiw	a5, a5, 1
   1ae48: 41 07        	addi	a4, a4, 16
   1ae4a: 23 30 b4 01  	sd	s11, 0(s0)
   1ae4e: 23 34 14 01  	sd	a7, 8(s0)
   1ae52: 3a ea        	sd	a4, 272(sp)
   1ae54: 23 24 f1 10  	sw	a5, 264(sp)
   1ae58: 41 04        	addi	s0, s0, 16
   1ae5a: e3 53 f3 fe  	bge	t1, a5, 0x1ae40 <_svfprintf_r+0x1064>
   1ae5e: 10 02        	addi	a2, sp, 256
   1ae60: e6 85        	mv	a1, s9
   1ae62: 4e 85        	mv	a0, s3
   1ae64: ef 20 a0 7e  	jal	0x1d64e <__ssprint_r>
   1ae68: e3 18 05 90  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1ae6c: c1 48        	addi	a7, zero, 16
   1ae6e: c1 34        	addiw	s1, s1, -16
   1ae70: 52 67        	ld	a4, 272(sp)
   1ae72: 83 27 81 10  	lw	a5, 264(sp)
   1ae76: 00 03        	addi	s0, sp, 384
   1ae78: 1d 43        	addi	t1, zero, 7
   1ae7a: e3 c6 98 fc  	blt	a7, s1, 0x1ae46 <_svfprintf_r+0x106a>
   1ae7e: 5e 86        	mv	a2, s7
   1ae80: a6 8b        	mv	s7, s1
   1ae82: c2 74        	ld	s1, 48(sp)
   1ae84: 9b 86 17 00  	addiw	a3, a5, 1
   1ae88: 5e 97        	add	a4, a4, s7
   1ae8a: 10 e0        	sd	a2, 0(s0)
   1ae8c: 23 34 74 01  	sd	s7, 8(s0)
   1ae90: 3a ea        	sd	a4, 272(sp)
   1ae92: 23 24 d1 10  	sw	a3, 264(sp)
   1ae96: 63 43 d3 74  	blt	t1, a3, 0x1b5dc <_svfprintf_r+0x1800>
   1ae9a: 83 46 09 00  	lbu	a3, 0(s2)
   1ae9e: 41 04        	addi	s0, s0, 16
   1aea0: 36 9a        	add	s4, s4, a3
   1aea2: a5 b7        	j	0x1ae0a <_svfprintf_r+0x102e>
   1aea4: 7d 19        	addi	s2, s2, -1
   1aea6: 7d 3d        	addiw	s10, s10, -1
   1aea8: c5 bd        	j	0x1ad98 <_svfprintf_r+0xfbc>
   1aeaa: 93 77 09 01  	andi	a5, s2, 16
   1aeae: c5 cb        	beqz	a5, 0x1af5e <_svfprintf_r+0x1182>
   1aeb0: 82 77        	ld	a5, 32(sp)
   1aeb2: 83 bb 07 00  	ld	s7, 0(a5)
   1aeb6: 65 bb        	j	0x1ac6e <_svfprintf_r+0xe92>
   1aeb8: 10 02        	addi	a2, sp, 256
   1aeba: e6 85        	mv	a1, s9
   1aebc: 4e 85        	mv	a0, s3
   1aebe: ef 20 00 79  	jal	0x1d64e <__ssprint_r>
   1aec2: e3 1b 05 8a  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1aec6: 52 67        	ld	a4, 272(sp)
   1aec8: 00 03        	addi	s0, sp, 384
   1aeca: c1 48        	addi	a7, zero, 16
   1aecc: 1d 43        	addi	t1, zero, 7
   1aece: dd b5        	j	0x1adb4 <_svfprintf_r+0xfd8>
   1aed0: 83 26 81 10  	lw	a3, 264(sp)
   1aed4: b7 25 02 00  	lui	a1, 34
   1aed8: 93 85 85 c2  	addi	a1, a1, -984
   1aedc: 85 26        	addiw	a3, a3, 1
   1aede: 0c e0        	sd	a1, 0(s0)
   1aee0: 13 87 1b 00  	addi	a4, s7, 1
   1aee4: 85 45        	addi	a1, zero, 1
   1aee6: 0c e4        	sd	a1, 8(s0)
   1aee8: 23 24 d1 10  	sw	a3, 264(sp)
   1aeec: 9b 85 06 00  	sext.w	a1, a3
   1aef0: 3a ea        	sd	a4, 272(sp)
   1aef2: 9d 46        	addi	a3, zero, 7
   1aef4: 41 04        	addi	s0, s0, 16
   1aef6: 63 c9 b6 04  	blt	a3, a1, 0x1af48 <_svfprintf_r+0x116c>
   1aefa: 63 11 06 16  	bnez	a2, 0x1b05c <_svfprintf_r+0x1280>
   1aefe: e2 77        	ld	a5, 56(sp)
   1af00: 93 76 19 00  	andi	a3, s2, 1
   1af04: dd 8e        	or	a3, a3, a5
   1af06: 63 8c 06 8e  	beqz	a3, 0x19ffe <_svfprintf_r+0x222>
   1af0a: 03 26 81 10  	lw	a2, 264(sp)
   1af0e: e6 67        	ld	a5, 88(sp)
   1af10: 86 76        	ld	a3, 96(sp)
   1af12: 3e 97        	add	a4, a4, a5
   1af14: 14 e0        	sd	a3, 0(s0)
   1af16: 9b 06 16 00  	addiw	a3, a2, 1
   1af1a: 1c e4        	sd	a5, 8(s0)
   1af1c: 3a ea        	sd	a4, 272(sp)
   1af1e: 23 24 d1 10  	sw	a3, 264(sp)
   1af22: 1d 46        	addi	a2, zero, 7
   1af24: 63 4e d6 44  	blt	a2, a3, 0x1b380 <_svfprintf_r+0x15a4>
   1af28: 41 04        	addi	s0, s0, 16
   1af2a: e2 77        	ld	a5, 56(sp)
   1af2c: 1b 86 16 00  	addiw	a2, a3, 1
   1af30: 23 30 64 01  	sd	s6, 0(s0)
   1af34: 3e 97        	add	a4, a4, a5
   1af36: 1c e4        	sd	a5, 8(s0)
   1af38: 3a ea        	sd	a4, 272(sp)
   1af3a: 23 24 c1 10  	sw	a2, 264(sp)
   1af3e: 9d 47        	addi	a5, zero, 7
   1af40: 63 de c7 8a  	bge	a5, a2, 0x19ffc <_svfprintf_r+0x220>
   1af44: 6f f0 2f af  	j	0x1a236 <_svfprintf_r+0x45a>
   1af48: 82 65        	ld	a1, 0(sp)
   1af4a: 10 02        	addi	a2, sp, 256
   1af4c: 4e 85        	mv	a0, s3
   1af4e: ef 20 00 70  	jal	0x1d64e <__ssprint_r>
   1af52: e3 13 05 82  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1af56: 6e 46        	lw	a2, 216(sp)
   1af58: 52 67        	ld	a4, 272(sp)
   1af5a: 00 03        	addi	s0, sp, 384
   1af5c: 79 bf        	j	0x1aefa <_svfprintf_r+0x111e>
   1af5e: 82 76        	ld	a3, 32(sp)
   1af60: 93 77 09 04  	andi	a5, s2, 64
   1af64: 83 ab 06 00  	lw	s7, 0(a3)
   1af68: 63 82 07 1a  	beqz	a5, 0x1b10c <_svfprintf_r+0x1330>
   1af6c: c2 1b        	slli	s7, s7, 48
   1af6e: 93 db 0b 03  	srli	s7, s7, 48
   1af72: f5 b9        	j	0x1ac6e <_svfprintf_r+0xe92>
   1af74: 82 65        	ld	a1, 0(sp)
   1af76: 10 02        	addi	a2, sp, 256
   1af78: 4e 85        	mv	a0, s3
   1af7a: ef 20 40 6d  	jal	0x1d64e <__ssprint_r>
   1af7e: 63 1d 05 fe  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1af82: d2 6b        	ld	s7, 272(sp)
   1af84: 00 03        	addi	s0, sp, 384
   1af86: 6f f0 af da  	j	0x1a530 <_svfprintf_r+0x754>
   1af8a: 9b 86 17 00  	addiw	a3, a5, 1
   1af8e: 23 30 74 01  	sd	s7, 0(s0)
   1af92: 23 34 94 01  	sd	s9, 8(s0)
   1af96: 66 97        	add	a4, a4, s9
   1af98: 3a ea        	sd	a4, 272(sp)
   1af9a: 23 24 d1 10  	sw	a3, 264(sp)
   1af9e: 9d 47        	addi	a5, zero, 7
   1afa0: 63 c4 d7 00  	blt	a5, a3, 0x1afa8 <_svfprintf_r+0x11cc>
   1afa4: 6f f0 8f 85  	j	0x19ffc <_svfprintf_r+0x220>
   1afa8: 6f f0 ef a8  	j	0x1a236 <_svfprintf_r+0x45a>
   1afac: 10 02        	addi	a2, sp, 256
   1afae: e6 85        	mv	a1, s9
   1afb0: 4e 85        	mv	a0, s3
   1afb2: ef 20 c0 69  	jal	0x1d64e <__ssprint_r>
   1afb6: 63 11 05 fc  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1afba: 83 46 09 00  	lbu	a3, 0(s2)
   1afbe: 52 67        	ld	a4, 272(sp)
   1afc0: 00 03        	addi	s0, sp, 384
   1afc2: 1b 86 06 00  	sext.w	a2, a3
   1afc6: c1 48        	addi	a7, zero, 16
   1afc8: 1d 43        	addi	t1, zero, 7
   1afca: 35 b5        	j	0x1adf6 <_svfprintf_r+0x101a>
   1afcc: 82 65        	ld	a1, 0(sp)
   1afce: 10 02        	addi	a2, sp, 256
   1afd0: 4e 85        	mv	a0, s3
   1afd2: 1a e9        	sd	t1, 144(sp)
   1afd4: ef 20 a0 67  	jal	0x1d64e <__ssprint_r>
   1afd8: 63 10 05 fa  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1afdc: d2 6b        	ld	s7, 272(sp)
   1afde: 4a 63        	ld	t1, 144(sp)
   1afe0: 00 03        	addi	s0, sp, 384
   1afe2: 6f e0 bf fe  	j	0x19fcc <_svfprintf_r+0x1f0>
   1afe6: 4e 75        	ld	a0, 240(sp)
   1afe8: ee 75        	ld	a1, 248(sp)
   1afea: 01 46        	mv	a2, zero
   1afec: 81 46        	mv	a3, zero
   1afee: ef 30 d0 48  	jal	0x1ec7a <__lttf2>
   1aff2: 63 4e 05 6a  	bltz	a0, 0x1b6ae <_svfprintf_r+0x18d2>
   1aff6: 03 47 f1 0c  	lbu	a4, 207(sp)
   1affa: 93 07 70 04  	addi	a5, zero, 71
   1affe: 63 d9 a7 19  	bge	a5, s10, 0x1b190 <_svfprintf_r+0x13b4>
   1b002: b7 27 02 00  	lui	a5, 34
   1b006: 13 8b 87 bd  	addi	s6, a5, -1064
   1b00a: 13 79 f9 f7  	andi	s2, s2, -129
   1b00e: 02 ec        	sd	zero, 24(sp)
   1b010: 82 e4        	sd	zero, 72(sp)
   1b012: 82 e0        	sd	zero, 64(sp)
   1b014: 02 f8        	sd	zero, 48(sp)
   1b016: 01 29        	sext.w	s2, s2
   1b018: 0d 4c        	addi	s8, zero, 3
   1b01a: 8d 4d        	addi	s11, zero, 3
   1b01c: 01 43        	mv	t1, zero
   1b01e: 63 1e 07 a8  	bnez	a4, 0x1a2ba <_svfprintf_r+0x4de>
   1b022: 6f e0 7f f3  	j	0x19f58 <_svfprintf_r+0x17c>
   1b026: 99 47        	addi	a5, zero, 6
   1b028: 9a 86        	mv	a3, t1
   1b02a: 63 e3 67 18  	bltu	a5, t1, 0x1b1b0 <_svfprintf_r+0x13d4>
   1b02e: 9b 8d 06 00  	sext.w	s11, a3
   1b032: b7 27 02 00  	lui	a5, 34
   1b036: 6e 8c        	mv	s8, s11
   1b038: 56 f0        	sd	s5, 32(sp)
   1b03a: 13 8b 07 c2  	addi	s6, a5, -992
   1b03e: 6f e0 1f f1  	j	0x19f4e <_svfprintf_r+0x172>
   1b042: 05 24        	addiw	s0, s0, 1
   1b044: 6e 9d        	add	s10, s10, s11
   1b046: 23 b0 7c 01  	sd	s7, 0(s9)
   1b04a: 23 b4 bc 01  	sd	s11, 8(s9)
   1b04e: 6a ea        	sd	s10, 272(sp)
   1b050: 23 24 81 10  	sw	s0, 264(sp)
   1b054: 9d 47        	addi	a5, zero, 7
   1b056: 63 dc 87 9a  	bge	a5, s0, 0x1a20e <_svfprintf_r+0x432>
   1b05a: d9 ba        	j	0x1aa30 <_svfprintf_r+0xc54>
   1b05c: 83 25 81 10  	lw	a1, 264(sp)
   1b060: e6 67        	ld	a5, 88(sp)
   1b062: 86 76        	ld	a3, 96(sp)
   1b064: 41 04        	addi	s0, s0, 16
   1b066: 3e 97        	add	a4, a4, a5
   1b068: 23 38 d4 fe  	sd	a3, -16(s0)
   1b06c: 9b 86 15 00  	addiw	a3, a1, 1
   1b070: 23 3c f4 fe  	sd	a5, -8(s0)
   1b074: 3a ea        	sd	a4, 272(sp)
   1b076: 23 24 d1 10  	sw	a3, 264(sp)
   1b07a: 9d 45        	addi	a1, zero, 7
   1b07c: 63 c2 d5 30  	blt	a1, a3, 0x1b380 <_svfprintf_r+0x15a4>
   1b080: e3 55 06 ea  	bgez	a2, 0x1af2a <_svfprintf_r+0x114e>
   1b084: 37 2f 02 00  	lui	t5, 34
   1b088: c1 55        	addi	a1, zero, -16
   1b08a: bb 0c c0 40  	negw	s9, a2
   1b08e: 93 0b 0f 54  	addi	s7, t5, 1344
   1b092: 41 4d        	addi	s10, zero, 16
   1b094: 9d 4d        	addi	s11, zero, 7
   1b096: 63 46 b6 00  	blt	a2, a1, 0x1b0a2 <_svfprintf_r+0x12c6>
   1b09a: c5 ad        	j	0x1b78a <_svfprintf_r+0x19ae>
   1b09c: c1 3c        	addiw	s9, s9, -16
   1b09e: 63 56 9d 6f  	bge	s10, s9, 0x1b78a <_svfprintf_r+0x19ae>
   1b0a2: 85 26        	addiw	a3, a3, 1
   1b0a4: 41 07        	addi	a4, a4, 16
   1b0a6: 23 30 74 01  	sd	s7, 0(s0)
   1b0aa: 23 34 a4 01  	sd	s10, 8(s0)
   1b0ae: 3a ea        	sd	a4, 272(sp)
   1b0b0: 23 24 d1 10  	sw	a3, 264(sp)
   1b0b4: 41 04        	addi	s0, s0, 16
   1b0b6: e3 d3 dd fe  	bge	s11, a3, 0x1b09c <_svfprintf_r+0x12c0>
   1b0ba: 82 65        	ld	a1, 0(sp)
   1b0bc: 10 02        	addi	a2, sp, 256
   1b0be: 4e 85        	mv	a0, s3
   1b0c0: ef 20 e0 58  	jal	0x1d64e <__ssprint_r>
   1b0c4: 63 1a 05 ea  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b0c8: 52 67        	ld	a4, 272(sp)
   1b0ca: 83 26 81 10  	lw	a3, 264(sp)
   1b0ce: 00 03        	addi	s0, sp, 384
   1b0d0: f1 b7        	j	0x1b09c <_svfprintf_r+0x12c0>
   1b0d2: 93 77 09 20  	andi	a5, s2, 512
   1b0d6: 63 8a 07 26  	beqz	a5, 0x1b34a <_svfprintf_r+0x156e>
   1b0da: 93 fb fb 0f  	andi	s7, s7, 255
   1b0de: 6f f0 8f b8  	j	0x1a466 <_svfprintf_r+0x68a>
   1b0e2: 93 f7 0c 20  	andi	a5, s9, 512
   1b0e6: 63 8b 07 24  	beqz	a5, 0x1b33c <_svfprintf_r+0x1560>
   1b0ea: 93 fb fb 0f  	andi	s7, s7, 255
   1b0ee: 3a f0        	sd	a4, 32(sp)
   1b0f0: 85 47        	addi	a5, zero, 1
   1b0f2: 6f f0 4f ac  	j	0x1a3b6 <_svfprintf_r+0x5da>
   1b0f6: 93 77 09 20  	andi	a5, s2, 512
   1b0fa: 63 86 07 26  	beqz	a5, 0x1b366 <_svfprintf_r+0x158a>
   1b0fe: 9b 9b 8b 01  	slliw	s7, s7, 24
   1b102: 9b db 8b 41  	sraiw	s7, s7, 24
   1b106: de 87        	mv	a5, s7
   1b108: 6f f0 cf b1  	j	0x1a424 <_svfprintf_r+0x648>
   1b10c: 93 77 09 20  	andi	a5, s2, 512
   1b110: 63 87 07 24  	beqz	a5, 0x1b35e <_svfprintf_r+0x1582>
   1b114: 93 fb fb 0f  	andi	s7, s7, 255
   1b118: 99 be        	j	0x1ac6e <_svfprintf_r+0xe92>
   1b11a: ee 77        	ld	a5, 248(sp)
   1b11c: 63 d8 07 24  	bgez	a5, 0x1b36c <_svfprintf_r+0x1590>
   1b120: 93 07 d0 02  	addi	a5, zero, 45
   1b124: a3 07 f1 0c  	sb	a5, 207(sp)
   1b128: 13 07 d0 02  	addi	a4, zero, 45
   1b12c: 93 07 70 04  	addi	a5, zero, 71
   1b130: 63 dc a7 5f  	bge	a5, s10, 0x1b728 <_svfprintf_r+0x194c>
   1b134: b7 27 02 00  	lui	a5, 34
   1b138: 13 8b 87 be  	addi	s6, a5, -1048
   1b13c: f9 b5        	j	0x1b00a <_svfprintf_r+0x122e>
   1b13e: 9b 05 16 00  	addiw	a1, a2, 1
   1b142: 6a 97        	add	a4, a4, s10
   1b144: 23 30 74 01  	sd	s7, 0(s0)
   1b148: 23 34 a4 01  	sd	s10, 8(s0)
   1b14c: 3a ea        	sd	a4, 272(sp)
   1b14e: 23 24 b1 10  	sw	a1, 264(sp)
   1b152: 9d 46        	addi	a3, zero, 7
   1b154: 63 c8 b6 24  	blt	a3, a1, 0x1b3a4 <_svfprintf_r+0x15c8>
   1b158: 41 04        	addi	s0, s0, 16
   1b15a: ba 8b        	mv	s7, a4
   1b15c: 6f f0 ef fc  	j	0x1a92a <_svfprintf_r+0xb4e>
   1b160: 3e fc        	sd	a5, 56(sp)
   1b162: 8a 67        	ld	a5, 128(sp)
   1b164: aa 65        	ld	a1, 136(sp)
   1b166: 1a f8        	sd	t1, 48(sp)
   1b168: 33 0b fb 40  	sub	s6, s6, a5
   1b16c: 3e 86        	mv	a2, a5
   1b16e: 5a 85        	mv	a0, s6
   1b170: 42 ec        	sd	a6, 24(sp)
   1b172: ef e0 df bd  	jal	0x19d4e <strncpy>
   1b176: 62 68        	ld	a6, 24(sp)
   1b178: 29 47        	addi	a4, zero, 10
   1b17a: 33 d7 eb 02  	divu	a4, s7, a4
   1b17e: 83 46 18 00  	lbu	a3, 1(a6)
   1b182: 42 73        	ld	t1, 48(sp)
   1b184: e2 77        	ld	a5, 56(sp)
   1b186: b3 36 d0 00  	snez	a3, a3
   1b18a: 36 98        	add	a6, a6, a3
   1b18c: 81 46        	mv	a3, zero
   1b18e: 71 b4        	j	0x1ac1a <_svfprintf_r+0xe3e>
   1b190: b7 27 02 00  	lui	a5, 34
   1b194: 13 8b 07 bd  	addi	s6, a5, -1072
   1b198: 8d bd        	j	0x1b00a <_svfprintf_r+0x122e>
   1b19a: 82 65        	ld	a1, 0(sp)
   1b19c: 10 02        	addi	a2, sp, 256
   1b19e: 4e 85        	mv	a0, s3
   1b1a0: ef 20 e0 4a  	jal	0x1d64e <__ssprint_r>
   1b1a4: 63 1a 05 dc  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b1a8: d2 6b        	ld	s7, 272(sp)
   1b1aa: 00 03        	addi	s0, sp, 384
   1b1ac: 6f f0 af f6  	j	0x1a916 <_svfprintf_r+0xb3a>
   1b1b0: 99 46        	addi	a3, zero, 6
   1b1b2: b5 bd        	j	0x1b02e <_svfprintf_r+0x1252>
   1b1b4: b3 8c 6a 41  	sub	s9, s5, s6
   1b1b8: 3b 86 d7 40  	subw	a2, a5, a3
   1b1bc: 9b 87 0c 00  	sext.w	a5, s9
   1b1c0: 5e 87        	mv	a4, s7
   1b1c2: 63 53 f6 00  	bge	a2, a5, 0x1b1c8 <_svfprintf_r+0x13ec>
   1b1c6: b2 8c        	mv	s9, a2
   1b1c8: 81 2c        	sext.w	s9, s9
   1b1ca: 6f f0 af fd  	j	0x1a9a4 <_svfprintf_r+0xbc8>
   1b1ce: 1a 89        	mv	s2, t1
   1b1d0: 89 46        	addi	a3, zero, 2
   1b1d2: 4a 87        	mv	a4, s2
   1b1d4: 93 08 81 0e  	addi	a7, sp, 232
   1b1d8: 13 08 c1 0d  	addi	a6, sp, 220
   1b1dc: bc 09        	addi	a5, sp, 216
   1b1de: e2 85        	mv	a1, s8
   1b1e0: 66 86        	mv	a2, s9
   1b1e2: 4e 85        	mv	a0, s3
   1b1e4: 1a f8        	sd	t1, 48(sp)
   1b1e6: ef b0 1f f5  	jal	0x17136 <_ldtoa_r>
   1b1ea: 13 07 70 06  	addi	a4, zero, 103
   1b1ee: 42 73        	ld	t1, 48(sp)
   1b1f0: 2a 8b        	mv	s6, a0
   1b1f2: 63 0c ed 3c  	beq	s10, a4, 0x1b5ca <_svfprintf_r+0x17ee>
   1b1f6: 13 07 70 04  	addi	a4, zero, 71
   1b1fa: 63 08 ed 3c  	beq	s10, a4, 0x1b5ca <_svfprintf_r+0x17ee>
   1b1fe: 93 7d fd fd  	andi	s11, s10, -33
   1b202: 13 07 60 04  	addi	a4, zero, 70
   1b206: b3 0a 25 01  	add	s5, a0, s2
   1b20a: 63 87 ed 4c  	beq	s11, a4, 0x1b6d8 <_svfprintf_r+0x18fc>
   1b20e: 62 85        	mv	a0, s8
   1b210: 01 46        	mv	a2, zero
   1b212: 81 46        	mv	a3, zero
   1b214: e6 85        	mv	a1, s9
   1b216: 1a f8        	sd	t1, 48(sp)
   1b218: ef 30 90 10  	jal	0x1eb20 <__netf2>
   1b21c: 42 73        	ld	t1, 48(sp)
   1b21e: 56 8c        	mv	s8, s5
   1b220: 11 cd        	beqz	a0, 0x1b23c <_svfprintf_r+0x1460>
   1b222: 2e 7c        	ld	s8, 232(sp)
   1b224: 63 7c 5c 01  	bgeu	s8, s5, 0x1b23c <_svfprintf_r+0x1460>
   1b228: 93 06 00 03  	addi	a3, zero, 48
   1b22c: 93 07 1c 00  	addi	a5, s8, 1
   1b230: be f5        	sd	a5, 232(sp)
   1b232: 23 00 dc 00  	sb	a3, 0(s8)
   1b236: 2e 7c        	ld	s8, 232(sp)
   1b238: e3 6a 5c ff  	bltu	s8, s5, 0x1b22c <_svfprintf_r+0x1450>
   1b23c: bb 07 6c 41  	subw	a5, s8, s6
   1b240: 3e fc        	sd	a5, 56(sp)
   1b242: 6e 46        	lw	a2, 216(sp)
   1b244: 13 07 70 04  	addi	a4, zero, 71
   1b248: 32 f8        	sd	a2, 48(sp)
   1b24a: 63 83 ed 2e  	beq	s11, a4, 0x1b530 <_svfprintf_r+0x1754>
   1b24e: 13 07 60 04  	addi	a4, zero, 70
   1b252: 63 8e ed 50  	beq	s11, a4, 0x1b76e <_svfprintf_r+0x1992>
   1b256: 1b 07 f6 ff  	addiw	a4, a2, -1
   1b25a: ba cd        	sw	a4, 216(sp)
   1b25c: 93 06 10 06  	addi	a3, zero, 97
   1b260: 63 02 dd 68  	beq	s10, a3, 0x1b8e4 <_svfprintf_r+0x1b08>
   1b264: 93 05 10 04  	addi	a1, zero, 65
   1b268: 93 06 00 05  	addi	a3, zero, 80
   1b26c: 63 1c bd 2c  	bne	s10, a1, 0x1b544 <_svfprintf_r+0x1768>
   1b270: 85 45        	addi	a1, zero, 1
   1b272: 23 00 d1 0e  	sb	a3, 224(sp)
   1b276: 93 06 b0 02  	addi	a3, zero, 43
   1b27a: 63 56 07 00  	bgez	a4, 0x1b286 <_svfprintf_r+0x14aa>
   1b27e: 05 47        	addi	a4, zero, 1
   1b280: 11 9f        	subw	a4, a4, a2
   1b282: 93 06 d0 02  	addi	a3, zero, 45
   1b286: a3 00 d1 0e  	sb	a3, 225(sp)
   1b28a: a5 46        	addi	a3, zero, 9
   1b28c: 63 d9 e6 52  	bge	a3, a4, 0x1b7be <_svfprintf_r+0x19e2>
   1b290: 13 08 f1 0e  	addi	a6, sp, 239
   1b294: c2 88        	mv	a7, a6
   1b296: 29 43        	addi	t1, zero, 10
   1b298: 13 0f 30 06  	addi	t5, zero, 99
   1b29c: bb 67 67 02  	remw	a5, a4, t1
   1b2a0: 46 86        	mv	a2, a7
   1b2a2: ba 86        	mv	a3, a4
   1b2a4: fd 18        	addi	a7, a7, -1
   1b2a6: 9b 87 07 03  	addiw	a5, a5, 48
   1b2aa: a3 0f f6 fe  	sb	a5, -1(a2)
   1b2ae: 3b 47 67 02  	divw	a4, a4, t1
   1b2b2: e3 45 df fe  	blt	t5, a3, 0x1b29c <_svfprintf_r+0x14c0>
   1b2b6: 1b 05 07 03  	addiw	a0, a4, 48
   1b2ba: a3 8f a8 fe  	sb	a0, -1(a7)
   1b2be: 13 07 e6 ff  	addi	a4, a2, -2
   1b2c2: 63 7b 07 67  	bgeu	a4, a6, 0x1b938 <_svfprintf_r+0x1b5c>
   1b2c6: 93 06 21 0e  	addi	a3, sp, 226
   1b2ca: 83 47 07 00  	lbu	a5, 0(a4)
   1b2ce: 05 07        	addi	a4, a4, 1
   1b2d0: 85 06        	addi	a3, a3, 1
   1b2d2: a3 8f f6 fe  	sb	a5, -1(a3)
   1b2d6: e3 1a 07 ff  	bne	a4, a6, 0x1b2ca <_svfprintf_r+0x14ee>
   1b2da: 13 07 11 0f  	addi	a4, sp, 241
   1b2de: 93 07 21 0e  	addi	a5, sp, 226
   1b2e2: 11 8f        	sub	a4, a4, a2
   1b2e4: 3e 97        	add	a4, a4, a5
   1b2e6: 94 11        	addi	a3, sp, 224
   1b2e8: bb 07 d7 40  	subw	a5, a4, a3
   1b2ec: be f8        	sd	a5, 112(sp)
   1b2ee: 46 77        	ld	a4, 112(sp)
   1b2f0: e2 77        	ld	a5, 56(sp)
   1b2f2: 05 46        	addi	a2, zero, 1
   1b2f4: bb 0d f7 00  	addw	s11, a4, a5
   1b2f8: 6e 87        	mv	a4, s11
   1b2fa: 63 52 f6 50  	bge	a2, a5, 0x1b7fe <_svfprintf_r+0x1a22>
   1b2fe: e6 67        	ld	a5, 88(sp)
   1b300: bb 0d f7 00  	addw	s11, a4, a5
   1b304: 86 67        	ld	a5, 64(sp)
   1b306: 13 c8 fd ff  	not	a6, s11
   1b30a: 13 58 f8 43  	srai	a6, a6, 63
   1b30e: 13 f9 f7 bf  	andi	s2, a5, -1025
   1b312: 01 29        	sext.w	s2, s2
   1b314: 33 f8 0d 01  	and	a6, s11, a6
   1b318: 13 69 09 10  	ori	s2, s2, 256
   1b31c: 1b 0c 08 00  	sext.w	s8, a6
   1b320: 82 e4        	sd	zero, 72(sp)
   1b322: 82 e0        	sd	zero, 64(sp)
   1b324: 02 f8        	sd	zero, 48(sp)
   1b326: ea 67        	ld	a5, 152(sp)
   1b328: 63 8a 07 24  	beqz	a5, 0x1b57c <_svfprintf_r+0x17a0>
   1b32c: 13 07 d0 02  	addi	a4, zero, 45
   1b330: a3 07 e1 0c  	sb	a4, 207(sp)
   1b334: 01 43        	mv	t1, zero
   1b336: 05 2c        	addiw	s8, s8, 1
   1b338: 6f e0 1f c2  	j	0x19f58 <_svfprintf_r+0x17c>
   1b33c: 82 1b        	slli	s7, s7, 32
   1b33e: 93 db 0b 02  	srli	s7, s7, 32
   1b342: 3a f0        	sd	a4, 32(sp)
   1b344: 85 47        	addi	a5, zero, 1
   1b346: 6f f0 0f 87  	j	0x1a3b6 <_svfprintf_r+0x5da>
   1b34a: 82 1b        	slli	s7, s7, 32
   1b34c: 93 db 0b 02  	srli	s7, s7, 32
   1b350: 6f f0 6f 91  	j	0x1a466 <_svfprintf_r+0x68a>
   1b354: c2 67        	ld	a5, 16(sp)
   1b356: 23 10 f7 00  	sh	a5, 0(a4)
   1b35a: 6f f0 4f 93  	j	0x1a48e <_svfprintf_r+0x6b2>
   1b35e: 82 1b        	slli	s7, s7, 32
   1b360: 93 db 0b 02  	srli	s7, s7, 32
   1b364: 29 b2        	j	0x1ac6e <_svfprintf_r+0xe92>
   1b366: de 87        	mv	a5, s7
   1b368: 6f f0 cf 8b  	j	0x1a424 <_svfprintf_r+0x648>
   1b36c: 03 47 f1 0c  	lbu	a4, 207(sp)
   1b370: 75 bb        	j	0x1b12c <_svfprintf_r+0x1350>
   1b372: 82 65        	ld	a1, 0(sp)
   1b374: 46 65        	ld	a0, 80(sp)
   1b376: 10 02        	addi	a2, sp, 256
   1b378: ef 20 60 2d  	jal	0x1d64e <__ssprint_r>
   1b37c: 6f e0 9f cc  	j	0x1a044 <_svfprintf_r+0x268>
   1b380: 82 65        	ld	a1, 0(sp)
   1b382: 10 02        	addi	a2, sp, 256
   1b384: 4e 85        	mv	a0, s3
   1b386: ef 20 80 2c  	jal	0x1d64e <__ssprint_r>
   1b38a: 63 17 05 be  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b38e: 6e 46        	lw	a2, 216(sp)
   1b390: 52 67        	ld	a4, 272(sp)
   1b392: 83 26 81 10  	lw	a3, 264(sp)
   1b396: 00 03        	addi	s0, sp, 384
   1b398: e3 59 06 b8  	bgez	a2, 0x1af2a <_svfprintf_r+0x114e>
   1b39c: e5 b1        	j	0x1b084 <_svfprintf_r+0x12a8>
   1b39e: 19 43        	addi	t1, zero, 6
   1b3a0: 6f e0 ff d8  	j	0x1a12e <_svfprintf_r+0x352>
   1b3a4: 82 65        	ld	a1, 0(sp)
   1b3a6: 10 02        	addi	a2, sp, 256
   1b3a8: 4e 85        	mv	a0, s3
   1b3aa: ef 20 40 2a  	jal	0x1d64e <__ssprint_r>
   1b3ae: 63 15 05 bc  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b3b2: d2 6b        	ld	s7, 272(sp)
   1b3b4: 00 03        	addi	s0, sp, 384
   1b3b6: 6f f0 4f d7  	j	0x1a92a <_svfprintf_r+0xb4e>
   1b3ba: 1b 09 13 00  	addiw	s2, t1, 1
   1b3be: 89 46        	addi	a3, zero, 2
   1b3c0: 09 bd        	j	0x1b1d2 <_svfprintf_r+0x13f6>
   1b3c2: 1a 89        	mv	s2, t1
   1b3c4: 8d 46        	addi	a3, zero, 3
   1b3c6: 31 b5        	j	0x1b1d2 <_svfprintf_r+0x13f6>
   1b3c8: e6 85        	mv	a1, s9
   1b3ca: 62 85        	mv	a0, s8
   1b3cc: 1a f8        	sd	t1, 48(sp)
   1b3ce: ef 40 10 27  	jal	0x1fe3e <__trunctfdf2>
   1b3d2: a8 09        	addi	a0, sp, 216
   1b3d4: ef e0 4f d7  	jal	0x19948 <frexp>
   1b3d8: ef 40 30 1a  	jal	0x1fd7a <__extenddftf2>
   1b3dc: 37 27 02 00  	lui	a4, 34
   1b3e0: 83 36 87 e5  	ld	a3, -424(a4)
   1b3e4: 01 46        	mv	a2, zero
   1b3e6: ef 30 50 15  	jal	0x1ed3a <__multf3>
   1b3ea: 01 46        	mv	a2, zero
   1b3ec: 81 46        	mv	a3, zero
   1b3ee: aa 8d        	mv	s11, a0
   1b3f0: 2e 89        	mv	s2, a1
   1b3f2: ef 30 e0 72  	jal	0x1eb20 <__netf2>
   1b3f6: 42 73        	ld	t1, 48(sp)
   1b3f8: 19 e1        	bnez	a0, 0x1b3fe <_svfprintf_r+0x1622>
   1b3fa: 05 47        	addi	a4, zero, 1
   1b3fc: ba cd        	sw	a4, 216(sp)
   1b3fe: 37 27 02 00  	lui	a4, 34
   1b402: 93 07 07 bf  	addi	a5, a4, -1040
   1b406: 3e fc        	sd	a5, 56(sp)
   1b408: b7 26 02 00  	lui	a3, 34
   1b40c: 83 b7 86 e6  	ld	a5, -408(a3)
   1b410: 13 17 03 02  	slli	a4, t1, 32
   1b414: 01 93        	srli	a4, a4, 32
   1b416: 05 07        	addi	a4, a4, 1
   1b418: 3e f8        	sd	a5, 48(sp)
   1b41a: 1b 06 03 00  	sext.w	a2, t1
   1b41e: b3 07 eb 00  	add	a5, s6, a4
   1b422: a6 e4        	sd	s1, 72(sp)
   1b424: 52 f1        	sd	s4, 160(sp)
   1b426: 22 f9        	sd	s0, 176(sp)
   1b428: e2 74        	ld	s1, 56(sp)
   1b42a: 1a f5        	sd	t1, 168(sp)
   1b42c: 5a 8a        	mv	s4, s6
   1b42e: 32 fd        	sd	a2, 184(sp)
   1b430: 3e 84        	mv	s0, a5
   1b432: ce 8b        	mv	s7, s3
   1b434: da 8c        	mv	s9, s6
   1b436: 39 a0        	j	0x1b444 <_svfprintf_r+0x1668>
   1b438: 01 46        	mv	a2, zero
   1b43a: 81 46        	mv	a3, zero
   1b43c: ef 30 40 6e  	jal	0x1eb20 <__netf2>
   1b440: 63 05 05 3c  	beqz	a0, 0x1b80a <_svfprintf_r+0x1a2e>
   1b444: c2 76        	ld	a3, 48(sp)
   1b446: 01 46        	mv	a2, zero
   1b448: 6e 85        	mv	a0, s11
   1b44a: ca 85        	mv	a1, s2
   1b44c: ef 30 f0 0e  	jal	0x1ed3a <__multf3>
   1b450: 2e 89        	mv	s2, a1
   1b452: aa 89        	mv	s3, a0
   1b454: ef 40 10 04  	jal	0x1fc94 <__fixtfsi>
   1b458: 1b 0c 05 00  	sext.w	s8, a0
   1b45c: 62 85        	mv	a0, s8
   1b45e: ef 40 70 0c  	jal	0x1fd24 <__floatsitf>
   1b462: ae 86        	mv	a3, a1
   1b464: 2a 86        	mv	a2, a0
   1b466: ca 85        	mv	a1, s2
   1b468: 4e 85        	mv	a0, s3
   1b46a: ef 30 30 6f  	jal	0x1f35c <__subtf3>
   1b46e: 33 87 84 01  	add	a4, s1, s8
   1b472: 83 46 07 00  	lbu	a3, 0(a4)
   1b476: d2 89        	mv	s3, s4
   1b478: 05 0a        	addi	s4, s4, 1
   1b47a: a3 0f da fe  	sb	a3, -1(s4)
   1b47e: aa 8a        	mv	s5, a0
   1b480: aa 8d        	mv	s11, a0
   1b482: 2e 89        	mv	s2, a1
   1b484: e3 1a 8a fa  	bne	s4, s0, 0x1b438 <_svfprintf_r+0x165c>
   1b488: d2 8a        	mv	s5, s4
   1b48a: a6 64        	ld	s1, 72(sp)
   1b48c: 2a 73        	ld	t1, 168(sp)
   1b48e: 4a 74        	ld	s0, 176(sp)
   1b490: 0a 7a        	ld	s4, 160(sp)
   1b492: 4e f8        	sd	s3, 48(sp)
   1b494: 66 8b        	mv	s6, s9
   1b496: de 89        	mv	s3, s7
   1b498: ae 8c        	mv	s9, a1
   1b49a: aa 8b        	mv	s7, a0
   1b49c: fd 5d        	addi	s11, zero, -1
   1b49e: b7 26 02 00  	lui	a3, 34
   1b4a2: 03 b9 86 e7  	ld	s2, -392(a3)
   1b4a6: 01 46        	mv	a2, zero
   1b4a8: 5e 85        	mv	a0, s7
   1b4aa: ca 86        	mv	a3, s2
   1b4ac: e6 85        	mv	a1, s9
   1b4ae: 9a e4        	sd	t1, 72(sp)
   1b4b0: ef 30 a0 70  	jal	0x1ebba <__gttf2>
   1b4b4: 26 63        	ld	t1, 72(sp)
   1b4b6: 63 4e a0 00  	bgtz	a0, 0x1b4d2 <_svfprintf_r+0x16f6>
   1b4ba: 01 46        	mv	a2, zero
   1b4bc: ca 86        	mv	a3, s2
   1b4be: 5e 85        	mv	a0, s7
   1b4c0: e6 85        	mv	a1, s9
   1b4c2: ef 30 e0 65  	jal	0x1eb20 <__netf2>
   1b4c6: 26 63        	ld	t1, 72(sp)
   1b4c8: 51 e1        	bnez	a0, 0x1b54c <_svfprintf_r+0x1770>
   1b4ca: 13 7c 1c 00  	andi	s8, s8, 1
   1b4ce: 63 0f 0c 06  	beqz	s8, 0x1b54c <_svfprintf_r+0x1770>
   1b4d2: c2 77        	ld	a5, 48(sp)
   1b4d4: d6 86        	mv	a3, s5
   1b4d6: be f5        	sd	a5, 232(sp)
   1b4d8: e2 77        	ld	a5, 56(sp)
   1b4da: 03 c6 fa ff  	lbu	a2, -1(s5)
   1b4de: 83 c5 f7 00  	lbu	a1, 15(a5)
   1b4e2: 63 9e c5 00  	bne	a1, a2, 0x1b4fe <_svfprintf_r+0x1722>
   1b4e6: 13 05 00 03  	addi	a0, zero, 48
   1b4ea: a3 8f a6 fe  	sb	a0, -1(a3)
   1b4ee: ae 76        	ld	a3, 232(sp)
   1b4f0: 93 87 f6 ff  	addi	a5, a3, -1
   1b4f4: be f5        	sd	a5, 232(sp)
   1b4f6: 03 c6 f6 ff  	lbu	a2, -1(a3)
   1b4fa: e3 88 c5 fe  	beq	a1, a2, 0x1b4ea <_svfprintf_r+0x170e>
   1b4fe: 9b 05 16 00  	addiw	a1, a2, 1
   1b502: 13 05 90 03  	addi	a0, zero, 57
   1b506: 93 f5 f5 0f  	andi	a1, a1, 255
   1b50a: 63 0a a6 00  	beq	a2, a0, 0x1b51e <_svfprintf_r+0x1742>
   1b50e: a3 8f b6 fe  	sb	a1, -1(a3)
   1b512: bb 87 6a 41  	subw	a5, s5, s6
   1b516: 3e fc        	sd	a5, 56(sp)
   1b518: 93 7d fd fd  	andi	s11, s10, -33
   1b51c: 1d b3        	j	0x1b242 <_svfprintf_r+0x1466>
   1b51e: e2 77        	ld	a5, 56(sp)
   1b520: 83 c5 a7 00  	lbu	a1, 10(a5)
   1b524: ed b7        	j	0x1b50e <_svfprintf_r+0x1732>
   1b526: 2e 7c        	ld	s8, 232(sp)
   1b528: 6e 46        	lw	a2, 216(sp)
   1b52a: bb 07 6c 41  	subw	a5, s8, s6
   1b52e: 3e fc        	sd	a5, 56(sp)
   1b530: 75 57        	addi	a4, zero, -3
   1b532: 32 f8        	sd	a2, 48(sp)
   1b534: 63 44 e6 00  	blt	a2, a4, 0x1b53c <_svfprintf_r+0x1760>
   1b538: 63 5a c3 04  	bge	t1, a2, 0x1b58c <_svfprintf_r+0x17b0>
   1b53c: 1b 07 f6 ff  	addiw	a4, a2, -1
   1b540: 79 3d        	addiw	s10, s10, -2
   1b542: ba cd        	sw	a4, 216(sp)
   1b544: 93 76 fd 0f  	andi	a3, s10, 255
   1b548: 81 45        	mv	a1, zero
   1b54a: 25 b3        	j	0x1b272 <_svfprintf_r+0x1496>
   1b54c: e3 c3 0d fc  	bltz	s11, 0x1b512 <_svfprintf_r+0x1736>
   1b550: 85 2d        	addiw	s11, s11, 1
   1b552: d7 75 00 04  	<unknown>
   1b556: 93 96 0d 02  	slli	a3, s11, 32
   1b55a: 93 05 00 03  	addi	a1, zero, 48
   1b55e: 81 92        	srli	a3, a3, 32
   1b560: 56 86        	mv	a2, s5
   1b562: 73 25 20 c2  	csrr	a0, vlenb
   1b566: 57 cc 05 5e  	<unknown>
   1b56a: d7 f7 06 04  	<unknown>
   1b56e: 27 0c 06 02  	<unknown>
   1b572: 9d 8e        	sub	a3, a3, a5
   1b574: 2a 96        	add	a2, a2, a0
   1b576: f5 fa        	bnez	a3, 0x1b56a <_svfprintf_r+0x178e>
   1b578: ee 9a        	add	s5, s5, s11
   1b57a: 61 bf        	j	0x1b512 <_svfprintf_r+0x1736>
   1b57c: 03 47 f1 0c  	lbu	a4, 207(sp)
   1b580: 01 43        	mv	t1, zero
   1b582: 19 c3        	beqz	a4, 0x1b588 <_svfprintf_r+0x17ac>
   1b584: 6f e0 7f d3  	j	0x1a2ba <_svfprintf_r+0x4de>
   1b588: 6f e0 1f 9d  	j	0x19f58 <_svfprintf_r+0x17c>
   1b58c: e2 77        	ld	a5, 56(sp)
   1b58e: 63 48 f6 1a  	blt	a2, a5, 0x1b73e <_svfprintf_r+0x1962>
   1b592: 86 67        	ld	a5, 64(sp)
   1b594: b2 8d        	mv	s11, a2
   1b596: 13 f7 17 00  	andi	a4, a5, 1
   1b59a: 01 c7        	beqz	a4, 0x1b5a2 <_svfprintf_r+0x17c6>
   1b59c: e6 67        	ld	a5, 88(sp)
   1b59e: bb 8d c7 00  	addw	s11, a5, a2
   1b5a2: 86 67        	ld	a5, 64(sp)
   1b5a4: 13 f7 07 40  	andi	a4, a5, 1024
   1b5a8: 19 c3        	beqz	a4, 0x1b5ae <_svfprintf_r+0x17d2>
   1b5aa: 63 44 c0 2a  	bgtz	a2, 0x1b852 <_svfprintf_r+0x1a76>
   1b5ae: 13 c8 fd ff  	not	a6, s11
   1b5b2: 13 58 f8 43  	srai	a6, a6, 63
   1b5b6: 4a 69        	ld	s2, 144(sp)
   1b5b8: 33 f8 0d 01  	and	a6, s11, a6
   1b5bc: 1b 0c 08 00  	sext.w	s8, a6
   1b5c0: 13 0d 70 06  	addi	s10, zero, 103
   1b5c4: 82 e4        	sd	zero, 72(sp)
   1b5c6: 82 e0        	sd	zero, 64(sp)
   1b5c8: b9 bb        	j	0x1b326 <_svfprintf_r+0x154a>
   1b5ca: 86 67        	ld	a5, 64(sp)
   1b5cc: 13 f7 17 00  	andi	a4, a5, 1
   1b5d0: 39 db        	beqz	a4, 0x1b526 <_svfprintf_r+0x174a>
   1b5d2: b3 0a 2b 01  	add	s5, s6, s2
   1b5d6: 93 0d 70 04  	addi	s11, zero, 71
   1b5da: 15 b9        	j	0x1b20e <_svfprintf_r+0x1432>
   1b5dc: 10 02        	addi	a2, sp, 256
   1b5de: e6 85        	mv	a1, s9
   1b5e0: 4e 85        	mv	a0, s3
   1b5e2: ef 20 c0 06  	jal	0x1d64e <__ssprint_r>
   1b5e6: 63 19 05 98  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b5ea: 83 46 09 00  	lbu	a3, 0(s2)
   1b5ee: 52 67        	ld	a4, 272(sp)
   1b5f0: 00 03        	addi	s0, sp, 384
   1b5f2: c1 48        	addi	a7, zero, 16
   1b5f4: 1d 43        	addi	t1, zero, 7
   1b5f6: 36 9a        	add	s4, s4, a3
   1b5f8: 6f f0 3f 81  	j	0x1ae0a <_svfprintf_r+0x102e>
   1b5fc: 13 07 80 07  	addi	a4, zero, 120
   1b600: 93 67 29 00  	ori	a5, s2, 2
   1b604: 93 06 00 03  	addi	a3, zero, 48
   1b608: 81 27        	sext.w	a5, a5
   1b60a: a3 08 e1 0c  	sb	a4, 209(sp)
   1b60e: 23 08 d1 0c  	sb	a3, 208(sp)
   1b612: 13 07 30 06  	addi	a4, zero, 99
   1b616: be e0        	sd	a5, 64(sp)
   1b618: 02 ec        	sd	zero, 24(sp)
   1b61a: 13 0b 81 11  	addi	s6, sp, 280
   1b61e: 63 44 67 1c  	blt	a4, t1, 0x1b7e6 <_svfprintf_r+0x1a0a>
   1b622: 13 69 29 10  	ori	s2, s2, 258
   1b626: ee 7c        	ld	s9, 248(sp)
   1b628: 9b 07 09 00  	sext.w	a5, s2
   1b62c: 3e e9        	sd	a5, 144(sp)
   1b62e: 02 ed        	sd	zero, 152(sp)
   1b630: 4e 7c        	ld	s8, 240(sp)
   1b632: 63 ca 0c 04  	bltz	s9, 0x1b686 <_svfprintf_r+0x18aa>
   1b636: 13 07 10 06  	addi	a4, zero, 97
   1b63a: 63 04 ed 00  	beq	s10, a4, 0x1b642 <_svfprintf_r+0x1866>
   1b63e: 6f e0 df b0  	j	0x1a14a <_svfprintf_r+0x36e>
   1b642: e6 85        	mv	a1, s9
   1b644: 62 85        	mv	a0, s8
   1b646: 1a f8        	sd	t1, 48(sp)
   1b648: ef 40 60 7f  	jal	0x1fe3e <__trunctfdf2>
   1b64c: a8 09        	addi	a0, sp, 216
   1b64e: ef e0 af af  	jal	0x19948 <frexp>
   1b652: ef 40 80 72  	jal	0x1fd7a <__extenddftf2>
   1b656: 37 27 02 00  	lui	a4, 34
   1b65a: 83 36 87 e5  	ld	a3, -424(a4)
   1b65e: 01 46        	mv	a2, zero
   1b660: ef 30 a0 6d  	jal	0x1ed3a <__multf3>
   1b664: 01 46        	mv	a2, zero
   1b666: 81 46        	mv	a3, zero
   1b668: aa 8d        	mv	s11, a0
   1b66a: 2e 89        	mv	s2, a1
   1b66c: ef 30 40 4b  	jal	0x1eb20 <__netf2>
   1b670: 42 73        	ld	t1, 48(sp)
   1b672: 19 e1        	bnez	a0, 0x1b678 <_svfprintf_r+0x189c>
   1b674: 05 47        	addi	a4, zero, 1
   1b676: ba cd        	sw	a4, 216(sp)
   1b678: 37 27 02 00  	lui	a4, 34
   1b67c: 93 07 87 c0  	addi	a5, a4, -1016
   1b680: 3e fc        	sd	a5, 56(sp)
   1b682: 59 b3        	j	0x1b408 <_svfprintf_r+0x162c>
   1b684: 02 ec        	sd	zero, 24(sp)
   1b686: 7d 57        	addi	a4, zero, -1
   1b688: 7e 17        	slli	a4, a4, 63
   1b68a: 93 07 d0 02  	addi	a5, zero, 45
   1b68e: b3 cc ec 00  	xor	s9, s9, a4
   1b692: 3e ed        	sd	a5, 152(sp)
   1b694: 4d b7        	j	0x1b636 <_svfprintf_r+0x185a>
   1b696: 82 65        	ld	a1, 0(sp)
   1b698: 10 02        	addi	a2, sp, 256
   1b69a: 4e 85        	mv	a0, s3
   1b69c: ef 10 30 7b  	jal	0x1d64e <__ssprint_r>
   1b6a0: 63 1c 05 8c  	bnez	a0, 0x1a778 <_svfprintf_r+0x99c>
   1b6a4: ee 46        	lw	a3, 216(sp)
   1b6a6: 52 67        	ld	a4, 272(sp)
   1b6a8: 00 03        	addi	s0, sp, 384
   1b6aa: 6f f0 2f ac  	j	0x1a96c <_svfprintf_r+0xb90>
   1b6ae: 93 07 d0 02  	addi	a5, zero, 45
   1b6b2: a3 07 f1 0c  	sb	a5, 207(sp)
   1b6b6: 13 07 d0 02  	addi	a4, zero, 45
   1b6ba: 81 b2        	j	0x1affa <_svfprintf_r+0x121e>
   1b6bc: 03 47 f1 0c  	lbu	a4, 207(sp)
   1b6c0: 56 f0        	sd	s5, 32(sp)
   1b6c2: 82 e4        	sd	zero, 72(sp)
   1b6c4: 82 e0        	sd	zero, 64(sp)
   1b6c6: 02 f8        	sd	zero, 48(sp)
   1b6c8: 1a 8c        	mv	s8, t1
   1b6ca: 9a 8d        	mv	s11, t1
   1b6cc: 01 43        	mv	t1, zero
   1b6ce: 19 c3        	beqz	a4, 0x1b6d4 <_svfprintf_r+0x18f8>
   1b6d0: 6f e0 bf be  	j	0x1a2ba <_svfprintf_r+0x4de>
   1b6d4: 6f e0 5f 88  	j	0x19f58 <_svfprintf_r+0x17c>
   1b6d8: 83 46 05 00  	lbu	a3, 0(a0)
   1b6dc: 13 07 00 03  	addi	a4, zero, 48
   1b6e0: 63 85 e6 00  	beq	a3, a4, 0x1b6ea <_svfprintf_r+0x190e>
   1b6e4: ee 46        	lw	a3, 216(sp)
   1b6e6: b6 9a        	add	s5, s5, a3
   1b6e8: 1d b6        	j	0x1b20e <_svfprintf_r+0x1432>
   1b6ea: 01 46        	mv	a2, zero
   1b6ec: 81 46        	mv	a3, zero
   1b6ee: 62 85        	mv	a0, s8
   1b6f0: e6 85        	mv	a1, s9
   1b6f2: 1a f8        	sd	t1, 48(sp)
   1b6f4: ef 30 c0 42  	jal	0x1eb20 <__netf2>
   1b6f8: 42 73        	ld	t1, 48(sp)
   1b6fa: 6d d5        	beqz	a0, 0x1b6e4 <_svfprintf_r+0x1908>
   1b6fc: 05 47        	addi	a4, zero, 1
   1b6fe: bb 06 27 41  	subw	a3, a4, s2
   1b702: b6 cd        	sw	a3, 216(sp)
   1b704: b6 9a        	add	s5, s5, a3
   1b706: 21 b6        	j	0x1b20e <_svfprintf_r+0x1432>
   1b708: 82 65        	ld	a1, 0(sp)
   1b70a: 10 02        	addi	a2, sp, 256
   1b70c: 4e 85        	mv	a0, s3
   1b70e: ef 10 10 74  	jal	0x1d64e <__ssprint_r>
   1b712: 19 c1        	beqz	a0, 0x1b718 <_svfprintf_r+0x193c>
   1b714: 6f f0 4f 86  	j	0x1a778 <_svfprintf_r+0x99c>
   1b718: 6e 46        	lw	a2, 216(sp)
   1b71a: e2 77        	ld	a5, 56(sp)
   1b71c: 52 67        	ld	a4, 272(sp)
   1b71e: 00 03        	addi	s0, sp, 384
   1b720: 3b 86 c7 40  	subw	a2, a5, a2
   1b724: 6f f0 0f a8  	j	0x1a9a4 <_svfprintf_r+0xbc8>
   1b728: b7 27 02 00  	lui	a5, 34
   1b72c: 13 8b 07 be  	addi	s6, a5, -1056
   1b730: e9 b8        	j	0x1b00a <_svfprintf_r+0x122e>
   1b732: 13 07 80 05  	addi	a4, zero, 88
   1b736: e9 b5        	j	0x1b600 <_svfprintf_r+0x1824>
   1b738: ca 8c        	mv	s9, s2
   1b73a: 6f f0 4f 8f  	j	0x1a82e <_svfprintf_r+0xa52>
   1b73e: e6 67        	ld	a5, 88(sp)
   1b740: 62 77        	ld	a4, 56(sp)
   1b742: 13 0d 70 06  	addi	s10, zero, 103
   1b746: bb 8d e7 00  	addw	s11, a5, a4
   1b74a: c2 77        	ld	a5, 48(sp)
   1b74c: 63 57 f0 16  	blez	a5, 0x1b8ba <_svfprintf_r+0x1ade>
   1b750: 86 67        	ld	a5, 64(sp)
   1b752: 13 f7 07 40  	andi	a4, a5, 1024
   1b756: 63 10 07 10  	bnez	a4, 0x1b856 <_svfprintf_r+0x1a7a>
   1b75a: 13 c8 fd ff  	not	a6, s11
   1b75e: 13 58 f8 43  	srai	a6, a6, 63
   1b762: 33 f8 0d 01  	and	a6, s11, a6
   1b766: 4a 69        	ld	s2, 144(sp)
   1b768: 1b 0c 08 00  	sext.w	s8, a6
   1b76c: a1 bd        	j	0x1b5c4 <_svfprintf_r+0x17e8>
   1b76e: 86 67        	ld	a5, 64(sp)
   1b770: 13 f7 17 00  	andi	a4, a5, 1
   1b774: c2 77        	ld	a5, 48(sp)
   1b776: 33 67 e3 00  	or	a4, t1, a4
   1b77a: 63 57 f0 18  	blez	a5, 0x1b908 <_svfprintf_r+0x1b2c>
   1b77e: 63 1b 07 14  	bnez	a4, 0x1b8d4 <_svfprintf_r+0x1af8>
   1b782: c2 7d        	ld	s11, 48(sp)
   1b784: 13 0d 60 06  	addi	s10, zero, 102
   1b788: e1 b7        	j	0x1b750 <_svfprintf_r+0x1974>
   1b78a: 85 26        	addiw	a3, a3, 1
   1b78c: 66 97        	add	a4, a4, s9
   1b78e: 23 30 74 01  	sd	s7, 0(s0)
   1b792: 23 34 94 01  	sd	s9, 8(s0)
   1b796: 3a ea        	sd	a4, 272(sp)
   1b798: 23 24 d1 10  	sw	a3, 264(sp)
   1b79c: 1d 46        	addi	a2, zero, 7
   1b79e: 63 55 d6 f8  	bge	a2, a3, 0x1af28 <_svfprintf_r+0x114c>
   1b7a2: 82 65        	ld	a1, 0(sp)
   1b7a4: 10 02        	addi	a2, sp, 256
   1b7a6: 4e 85        	mv	a0, s3
   1b7a8: ef 10 70 6a  	jal	0x1d64e <__ssprint_r>
   1b7ac: 19 c1        	beqz	a0, 0x1b7b2 <_svfprintf_r+0x19d6>
   1b7ae: 6f e0 bf fc  	j	0x1a778 <_svfprintf_r+0x99c>
   1b7b2: 52 67        	ld	a4, 272(sp)
   1b7b4: 83 26 81 10  	lw	a3, 264(sp)
   1b7b8: 00 03        	addi	s0, sp, 384
   1b7ba: 6f f0 0f f7  	j	0x1af2a <_svfprintf_r+0x114e>
   1b7be: 93 06 21 0e  	addi	a3, sp, 226
   1b7c2: 99 e5        	bnez	a1, 0x1b7d0 <_svfprintf_r+0x19f4>
   1b7c4: 93 06 00 03  	addi	a3, zero, 48
   1b7c8: 23 01 d1 0e  	sb	a3, 226(sp)
   1b7cc: 93 06 31 0e  	addi	a3, sp, 227
   1b7d0: 1c 04        	addi	a5, sp, 512
   1b7d2: 33 86 f6 40  	sub	a2, a3, a5
   1b7d6: 1b 07 07 03  	addiw	a4, a4, 48
   1b7da: 9b 07 16 12  	addiw	a5, a2, 289
   1b7de: 23 80 e6 00  	sb	a4, 0(a3)
   1b7e2: be f8        	sd	a5, 112(sp)
   1b7e4: 29 b6        	j	0x1b2ee <_svfprintf_r+0x1512>
   1b7e6: 9b 05 13 00  	addiw	a1, t1, 1
   1b7ea: 4e 85        	mv	a0, s3
   1b7ec: 1a ec        	sd	t1, 24(sp)
   1b7ee: ef c0 3f d4  	jal	0x18530 <_malloc_r>
   1b7f2: 62 63        	ld	t1, 24(sp)
   1b7f4: 2a 8b        	mv	s6, a0
   1b7f6: 63 07 05 14  	beqz	a0, 0x1b944 <_svfprintf_r+0x1b68>
   1b7fa: 2a ec        	sd	a0, 24(sp)
   1b7fc: 1d b5        	j	0x1b622 <_svfprintf_r+0x1846>
   1b7fe: 86 67        	ld	a5, 64(sp)
   1b800: 13 f6 17 00  	andi	a2, a5, 1
   1b804: e3 00 06 b0  	beqz	a2, 0x1b304 <_svfprintf_r+0x1528>
   1b808: dd bc        	j	0x1b2fe <_svfprintf_r+0x1522>
   1b80a: 6a 76        	ld	a2, 184(sp)
   1b80c: 66 8b        	mv	s6, s9
   1b80e: 4e f8        	sd	s3, 48(sp)
   1b810: 3b 06 cb 00  	addw	a2, s6, a2
   1b814: de 89        	mv	s3, s7
   1b816: d6 8b        	mv	s7, s5
   1b818: d2 8a        	mv	s5, s4
   1b81a: a6 64        	ld	s1, 72(sp)
   1b81c: 2a 73        	ld	t1, 168(sp)
   1b81e: 4a 74        	ld	s0, 176(sp)
   1b820: 0a 7a        	ld	s4, 160(sp)
   1b822: ca 8c        	mv	s9, s2
   1b824: bb 0d 56 41  	subw	s11, a2, s5
   1b828: 9d b9        	j	0x1b49e <_svfprintf_r+0x16c2>
   1b82a: 63 04 03 00  	beqz	t1, 0x1b832 <_svfprintf_r+0x1a56>
   1b82e: 6f e0 1f 90  	j	0x1a12e <_svfprintf_r+0x352>
   1b832: 05 43        	addi	t1, zero, 1
   1b834: 6f e0 bf 8f  	j	0x1a12e <_svfprintf_r+0x352>
   1b838: 46 67        	ld	a4, 80(sp)
   1b83a: b1 47        	addi	a5, zero, 12
   1b83c: 1c c3        	sw	a5, 0(a4)
   1b83e: fd 57        	addi	a5, zero, -1
   1b840: 3e e8        	sd	a5, 16(sp)
   1b842: 6f e0 3f 81  	j	0x1a054 <_svfprintf_r+0x278>
   1b846: 3a f0        	sd	a4, 32(sp)
   1b848: 6f e0 7f fe  	j	0x1a82e <_svfprintf_r+0xa52>
   1b84c: 1c c3        	sw	a5, 0(a4)
   1b84e: 6f e0 1f c4  	j	0x1a48e <_svfprintf_r+0x6b2>
   1b852: 13 0d 70 06  	addi	s10, zero, 103
   1b856: e6 76        	ld	a3, 120(sp)
   1b858: 13 06 f0 0f  	addi	a2, zero, 255
   1b85c: 03 c7 06 00  	lbu	a4, 0(a3)
   1b860: 63 0f c7 0c  	beq	a4, a2, 0x1b93e <_svfprintf_r+0x1b62>
   1b864: c2 77        	ld	a5, 48(sp)
   1b866: 01 48        	mv	a6, zero
   1b868: 01 45        	mv	a0, zero
   1b86a: 93 05 f0 0f  	addi	a1, zero, 255
   1b86e: 1b 06 07 00  	sext.w	a2, a4
   1b872: 63 5a f7 00  	bge	a4, a5, 0x1b886 <_svfprintf_r+0x1aaa>
   1b876: 03 c7 16 00  	lbu	a4, 1(a3)
   1b87a: 91 9f        	subw	a5, a5, a2
   1b87c: 1d cb        	beqz	a4, 0x1b8b2 <_svfprintf_r+0x1ad6>
   1b87e: 05 25        	addiw	a0, a0, 1
   1b880: 85 06        	addi	a3, a3, 1
   1b882: e3 16 b7 fe  	bne	a4, a1, 0x1b86e <_svfprintf_r+0x1a92>
   1b886: b6 fc        	sd	a3, 120(sp)
   1b888: 3e f8        	sd	a5, 48(sp)
   1b88a: aa e0        	sd	a0, 64(sp)
   1b88c: c2 e4        	sd	a6, 72(sp)
   1b88e: a6 67        	ld	a5, 72(sp)
   1b890: 06 67        	ld	a4, 64(sp)
   1b892: 4a 69        	ld	s2, 144(sp)
   1b894: 3d 9f        	addw	a4, a4, a5
   1b896: 8a 67        	ld	a5, 128(sp)
   1b898: 3b 07 f7 02  	mulw	a4, a4, a5
   1b89c: 3b 07 b7 01  	addw	a4, a4, s11
   1b8a0: 9b 0d 07 00  	sext.w	s11, a4
   1b8a4: 13 c6 fd ff  	not	a2, s11
   1b8a8: 7d 96        	srai	a2, a2, 63
   1b8aa: 71 8f        	and	a4, a4, a2
   1b8ac: 1b 0c 07 00  	sext.w	s8, a4
   1b8b0: 9d bc        	j	0x1b326 <_svfprintf_r+0x154a>
   1b8b2: 03 c7 06 00  	lbu	a4, 0(a3)
   1b8b6: 05 28        	addiw	a6, a6, 1
   1b8b8: e9 b7        	j	0x1b882 <_svfprintf_r+0x1aa6>
   1b8ba: 3b 87 fd 40  	subw	a4, s11, a5
   1b8be: 05 27        	addiw	a4, a4, 1
   1b8c0: 9b 0d 07 00  	sext.w	s11, a4
   1b8c4: 13 c6 fd ff  	not	a2, s11
   1b8c8: 7d 96        	srai	a2, a2, 63
   1b8ca: 71 8f        	and	a4, a4, a2
   1b8cc: 4a 69        	ld	s2, 144(sp)
   1b8ce: 1b 0c 07 00  	sext.w	s8, a4
   1b8d2: cd b9        	j	0x1b5c4 <_svfprintf_r+0x17e8>
   1b8d4: 66 67        	ld	a4, 88(sp)
   1b8d6: 13 0d 60 06  	addi	s10, zero, 102
   1b8da: bb 06 f7 00  	addw	a3, a4, a5
   1b8de: bb 8d 66 00  	addw	s11, a3, t1
   1b8e2: bd b5        	j	0x1b750 <_svfprintf_r+0x1974>
   1b8e4: 93 06 00 07  	addi	a3, zero, 112
   1b8e8: 61 b2        	j	0x1b270 <_svfprintf_r+0x1494>
   1b8ea: 82 76        	ld	a3, 32(sp)
   1b8ec: 98 42        	lw	a4, 0(a3)
   1b8ee: a1 06        	addi	a3, a3, 8
   1b8f0: 3a 83        	mv	t1, a4
   1b8f2: 63 53 07 00  	bgez	a4, 0x1b8f8 <_svfprintf_r+0x1b1c>
   1b8f6: 7d 53        	addi	t1, zero, -1
   1b8f8: 03 ce 14 00  	lbu	t3, 1(s1)
   1b8fc: 9b 0a 03 00  	sext.w	s5, t1
   1b900: 36 f0        	sd	a3, 32(sp)
   1b902: be 84        	mv	s1, a5
   1b904: 6f e0 ef dd  	j	0x19ee2 <_svfprintf_r+0x106>
   1b908: 19 e7        	bnez	a4, 0x1b916 <_svfprintf_r+0x1b3a>
   1b90a: 4a 69        	ld	s2, 144(sp)
   1b90c: 05 4c        	addi	s8, zero, 1
   1b90e: 13 0d 60 06  	addi	s10, zero, 102
   1b912: 85 4d        	addi	s11, zero, 1
   1b914: 45 b9        	j	0x1b5c4 <_svfprintf_r+0x17e8>
   1b916: e6 67        	ld	a5, 88(sp)
   1b918: 4a 69        	ld	s2, 144(sp)
   1b91a: 13 0d 60 06  	addi	s10, zero, 102
   1b91e: 1b 87 17 00  	addiw	a4, a5, 1
   1b922: 3b 07 67 00  	addw	a4, a4, t1
   1b926: 9b 0d 07 00  	sext.w	s11, a4
   1b92a: 13 c6 fd ff  	not	a2, s11
   1b92e: 7d 96        	srai	a2, a2, 63
   1b930: 71 8f        	and	a4, a4, a2
   1b932: 1b 0c 07 00  	sext.w	s8, a4
   1b936: 79 b1        	j	0x1b5c4 <_svfprintf_r+0x17e8>
   1b938: 89 47        	addi	a5, zero, 2
   1b93a: be f8        	sd	a5, 112(sp)
   1b93c: 4d ba        	j	0x1b2ee <_svfprintf_r+0x1512>
   1b93e: 82 e4        	sd	zero, 72(sp)
   1b940: 82 e0        	sd	zero, 64(sp)
   1b942: b1 b7        	j	0x1b88e <_svfprintf_r+0x1ab2>
   1b944: 02 67        	ld	a4, 0(sp)
   1b946: 83 57 07 01  	lhu	a5, 16(a4)
   1b94a: 93 e7 07 04  	ori	a5, a5, 64
   1b94e: 23 18 f7 00  	sh	a5, 16(a4)
   1b952: 6f e0 2f ef  	j	0x1a044 <_svfprintf_r+0x268>

000000000001b956 <__sprint_r.part.0>:
   1b956: 83 a7 c5 0a  	lw	a5, 172(a1)
   1b95a: 5d 71        	addi	sp, sp, -80
   1b95c: 56 ec        	sd	s5, 24(sp)
   1b95e: b5 83        	srli	a5, a5, 13
   1b960: 86 e4        	sd	ra, 72(sp)
   1b962: a2 e0        	sd	s0, 64(sp)
   1b964: 26 fc        	sd	s1, 56(sp)
   1b966: 4a f8        	sd	s2, 48(sp)
   1b968: 4e f4        	sd	s3, 40(sp)
   1b96a: 52 f0        	sd	s4, 32(sp)
   1b96c: 5a e8        	sd	s6, 16(sp)
   1b96e: 5e e4        	sd	s7, 8(sp)
   1b970: 85 8b        	andi	a5, a5, 1
   1b972: b2 8a        	mv	s5, a2
   1b974: c1 c3        	beqz	a5, 0x1b9f4 <__sprint_r.part.0+0x9e>
   1b976: 1c 6a        	ld	a5, 16(a2)
   1b978: 03 3b 06 00  	ld	s6, 0(a2)
   1b97c: 2e 89        	mv	s2, a1
   1b97e: aa 89        	mv	s3, a0
   1b980: 7d 5a        	addi	s4, zero, -1
   1b982: bd c7        	beqz	a5, 0x1b9f0 <__sprint_r.part.0+0x9a>
   1b984: 83 34 8b 00  	ld	s1, 8(s6)
   1b988: 03 34 0b 00  	ld	s0, 0(s6)
   1b98c: 89 80        	srli	s1, s1, 2
   1b98e: 9b 8b 04 00  	sext.w	s7, s1
   1b992: 63 58 70 05  	blez	s7, 0x1b9e2 <__sprint_r.part.0+0x8c>
   1b996: fd 34        	addiw	s1, s1, -1
   1b998: 93 97 04 02  	slli	a5, s1, 32
   1b99c: 93 d4 e7 01  	srli	s1, a5, 30
   1b9a0: 93 07 44 00  	addi	a5, s0, 4
   1b9a4: be 94        	add	s1, s1, a5
   1b9a6: 19 a0        	j	0x1b9ac <__sprint_r.part.0+0x56>
   1b9a8: 63 8b 84 02  	beq	s1, s0, 0x1b9de <__sprint_r.part.0+0x88>
   1b9ac: 0c 40        	lw	a1, 0(s0)
   1b9ae: 4a 86        	mv	a2, s2
   1b9b0: 4e 85        	mv	a0, s3
   1b9b2: ef 10 40 13  	jal	0x1cae6 <_fputwc_r>
   1b9b6: 01 25        	sext.w	a0, a0
   1b9b8: 11 04        	addi	s0, s0, 4
   1b9ba: e3 17 45 ff  	bne	a0, s4, 0x1b9a8 <__sprint_r.part.0+0x52>
   1b9be: 7d 55        	addi	a0, zero, -1
   1b9c0: a6 60        	ld	ra, 72(sp)
   1b9c2: 06 64        	ld	s0, 64(sp)
   1b9c4: 23 b8 0a 00  	sd	zero, 16(s5)
   1b9c8: 23 a4 0a 00  	sw	zero, 8(s5)
   1b9cc: e2 74        	ld	s1, 56(sp)
   1b9ce: 42 79        	ld	s2, 48(sp)
   1b9d0: a2 79        	ld	s3, 40(sp)
   1b9d2: 02 7a        	ld	s4, 32(sp)
   1b9d4: e2 6a        	ld	s5, 24(sp)
   1b9d6: 42 6b        	ld	s6, 16(sp)
   1b9d8: a2 6b        	ld	s7, 8(sp)
   1b9da: 61 61        	addi	sp, sp, 80
   1b9dc: 82 80        	ret
   1b9de: 83 b7 0a 01  	ld	a5, 16(s5)
   1b9e2: 8a 0b        	slli	s7, s7, 2
   1b9e4: b3 87 77 41  	sub	a5, a5, s7
   1b9e8: 23 b8 fa 00  	sd	a5, 16(s5)
   1b9ec: 41 0b        	addi	s6, s6, 16
   1b9ee: d9 fb        	bnez	a5, 0x1b984 <__sprint_r.part.0+0x2e>
   1b9f0: 01 45        	mv	a0, zero
   1b9f2: f9 b7        	j	0x1b9c0 <__sprint_r.part.0+0x6a>
   1b9f4: ef 10 00 1a  	jal	0x1cb94 <__sfvwrite_r>
   1b9f8: e1 b7        	j	0x1b9c0 <__sprint_r.part.0+0x6a>

000000000001b9fa <__sprint_r>:
   1b9fa: 18 6a        	ld	a4, 16(a2)
   1b9fc: 11 c3        	beqz	a4, 0x1ba00 <__sprint_r+0x6>
   1b9fe: a1 bf        	j	0x1b956 <__sprint_r.part.0>
   1ba00: 23 24 06 00  	sw	zero, 8(a2)
   1ba04: 01 45        	mv	a0, zero
   1ba06: 82 80        	ret

000000000001ba08 <_vfiprintf_r>:
   1ba08: 05 71        	addi	sp, sp, -480
   1ba0a: 52 fb        	sd	s4, 432(sp)
   1ba0c: 56 f7        	sd	s5, 424(sp)
   1ba0e: 5e ef        	sd	s7, 408(sp)
   1ba10: ee fe        	sd	s11, 376(sp)
   1ba12: 86 ef        	sd	ra, 472(sp)
   1ba14: a2 eb        	sd	s0, 464(sp)
   1ba16: a6 e7        	sd	s1, 456(sp)
   1ba18: ca e3        	sd	s2, 448(sp)
   1ba1a: 4e ff        	sd	s3, 440(sp)
   1ba1c: 5a f3        	sd	s6, 416(sp)
   1ba1e: 62 eb        	sd	s8, 400(sp)
   1ba20: 66 e7        	sd	s9, 392(sp)
   1ba22: 6a e3        	sd	s10, 384(sp)
   1ba24: aa 8a        	mv	s5, a0
   1ba26: 2e 8a        	mv	s4, a1
   1ba28: b2 8d        	mv	s11, a2
   1ba2a: b6 8b        	mv	s7, a3
   1ba2c: 01 c5        	beqz	a0, 0x1ba34 <_vfiprintf_r+0x2c>
   1ba2e: 3c 49        	lw	a5, 80(a0)
   1ba30: 63 8a 07 38  	beqz	a5, 0x1bdc4 <_vfiprintf_r+0x3bc>
   1ba34: 83 17 0a 01  	lh	a5, 16(s4)
   1ba38: 13 97 27 03  	slli	a4, a5, 50
   1ba3c: 63 41 07 02  	bltz	a4, 0x1ba5e <_vfiprintf_r+0x56>
   1ba40: 03 27 ca 0a  	lw	a4, 172(s4)
   1ba44: 89 66        	lui	a3, 2
   1ba46: d5 8f        	or	a5, a5, a3
   1ba48: f9 76        	lui	a3, 1048574
   1ba4a: 9b 97 07 01  	slliw	a5, a5, 16
   1ba4e: fd 16        	addi	a3, a3, -1
   1ba50: 9b d7 07 41  	sraiw	a5, a5, 16
   1ba54: 75 8f        	and	a4, a4, a3
   1ba56: 23 18 fa 00  	sh	a5, 16(s4)
   1ba5a: 23 26 ea 0a  	sw	a4, 172(s4)
   1ba5e: 13 f7 87 00  	andi	a4, a5, 8
   1ba62: 69 cb        	beqz	a4, 0x1bb34 <_vfiprintf_r+0x12c>
   1ba64: 03 37 8a 01  	ld	a4, 24(s4)
   1ba68: 71 c7        	beqz	a4, 0x1bb34 <_vfiprintf_r+0x12c>
   1ba6a: e9 8b        	andi	a5, a5, 26
   1ba6c: 29 47        	addi	a4, zero, 10
   1ba6e: 63 8f e7 0c  	beq	a5, a4, 0x1bb4c <_vfiprintf_r+0x144>
   1ba72: 37 2d 02 00  	lui	s10, 34
   1ba76: 93 09 01 0f  	addi	s3, sp, 240
   1ba7a: 01 4c        	mv	s8, zero
   1ba7c: 93 07 0d 56  	addi	a5, s10, 1376
   1ba80: 37 2b 02 00  	lui	s6, 34
   1ba84: e2 8c        	mv	s9, s8
   1ba86: ce f8        	sd	s3, 112(sp)
   1ba88: 02 e1        	sd	zero, 128(sp)
   1ba8a: 82 dc        	sw	zero, 120(sp)
   1ba8c: 4e 84        	mv	s0, s3
   1ba8e: 02 e8        	sd	zero, 16(sp)
   1ba90: 02 fc        	sd	zero, 56(sp)
   1ba92: 02 f8        	sd	zero, 48(sp)
   1ba94: 82 e0        	sd	zero, 64(sp)
   1ba96: 3e e4        	sd	a5, 8(sp)
   1ba98: 41 49        	addi	s2, zero, 16
   1ba9a: 13 0b 0b 6d  	addi	s6, s6, 1744
   1ba9e: 5e 8c        	mv	s8, s7
   1baa0: 83 c7 0d 00  	lbu	a5, 0(s11)
   1baa4: 63 89 07 24  	beqz	a5, 0x1bcf6 <_vfiprintf_r+0x2ee>
   1baa8: ee 84        	mv	s1, s11
   1baaa: 13 07 50 02  	addi	a4, zero, 37
   1baae: 63 80 e7 2a  	beq	a5, a4, 0x1bd4e <_vfiprintf_r+0x346>
   1bab2: 83 c7 14 00  	lbu	a5, 1(s1)
   1bab6: 85 04        	addi	s1, s1, 1
   1bab8: fd fb        	bnez	a5, 0x1baae <_vfiprintf_r+0xa6>
   1baba: bb 8b b4 41  	subw	s7, s1, s11
   1babe: 63 8c 0b 22  	beqz	s7, 0x1bcf6 <_vfiprintf_r+0x2ee>
   1bac2: e6 57        	lw	a5, 120(sp)
   1bac4: 0a 67        	ld	a4, 128(sp)
   1bac6: 23 30 b4 01  	sd	s11, 0(s0)
   1baca: 9b 86 17 00  	addiw	a3, a5, 1
   1bace: 5e 97        	add	a4, a4, s7
   1bad0: 23 34 74 01  	sd	s7, 8(s0)
   1bad4: 3a e1        	sd	a4, 128(sp)
   1bad6: b6 dc        	sw	a3, 120(sp)
   1bad8: 9d 47        	addi	a5, zero, 7
   1bada: 41 04        	addi	s0, s0, 16
   1badc: 63 db d7 00  	bge	a5, a3, 0x1baf2 <_vfiprintf_r+0xea>
   1bae0: e3 00 07 10  	beqz	a4, 0x1c3e0 <_vfiprintf_r+0x9d8>
   1bae4: 90 18        	addi	a2, sp, 112
   1bae6: d2 85        	mv	a1, s4
   1bae8: 56 85        	mv	a0, s5
   1baea: ef f0 df e6  	jal	0x1b956 <__sprint_r.part.0>
   1baee: 51 ed        	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1baf0: 4e 84        	mv	s0, s3
   1baf2: 83 c7 04 00  	lbu	a5, 0(s1)
   1baf6: bb 8c 7c 01  	addw	s9, s9, s7
   1bafa: 63 8e 07 1e  	beqz	a5, 0x1bcf6 <_vfiprintf_r+0x2ee>
   1bafe: 83 c7 14 00  	lbu	a5, 1(s1)
   1bb02: 93 8d 14 00  	addi	s11, s1, 1
   1bb06: a3 03 01 06  	sb	zero, 103(sp)
   1bb0a: 7d 5d        	addi	s10, zero, -1
   1bb0c: 81 48        	mv	a7, zero
   1bb0e: 81 4b        	mv	s7, zero
   1bb10: 93 04 a0 05  	addi	s1, zero, 90
   1bb14: 85 0d        	addi	s11, s11, 1
   1bb16: 81 27        	sext.w	a5, a5
   1bb18: 1b 87 07 fe  	addiw	a4, a5, -32
   1bb1c: 9b 06 07 00  	sext.w	a3, a4
   1bb20: 63 e7 d4 0c  	bltu	s1, a3, 0x1bbee <_vfiprintf_r+0x1e6>
   1bb24: 93 16 07 02  	slli	a3, a4, 32
   1bb28: 13 d7 e6 01  	srli	a4, a3, 30
   1bb2c: a2 66        	ld	a3, 8(sp)
   1bb2e: 36 97        	add	a4, a4, a3
   1bb30: 18 43        	lw	a4, 0(a4)
   1bb32: 02 87        	jr	a4
   1bb34: d2 85        	mv	a1, s4
   1bb36: 56 85        	mv	a0, s5
   1bb38: ef 90 8f e0  	jal	0x15140 <__swsetup_r>
   1bb3c: e3 13 05 38  	bnez	a0, 0x1c6c2 <_vfiprintf_r+0xcba>
   1bb40: 83 17 0a 01  	lh	a5, 16(s4)
   1bb44: 29 47        	addi	a4, zero, 10
   1bb46: e9 8b        	andi	a5, a5, 26
   1bb48: e3 95 e7 f2  	bne	a5, a4, 0x1ba72 <_vfiprintf_r+0x6a>
   1bb4c: 83 17 2a 01  	lh	a5, 18(s4)
   1bb50: e3 c1 07 f2  	bltz	a5, 0x1ba72 <_vfiprintf_r+0x6a>
   1bb54: 5e 64        	ld	s0, 464(sp)
   1bb56: fe 60        	ld	ra, 472(sp)
   1bb58: be 64        	ld	s1, 456(sp)
   1bb5a: 1e 69        	ld	s2, 448(sp)
   1bb5c: fa 79        	ld	s3, 440(sp)
   1bb5e: 1a 7b        	ld	s6, 416(sp)
   1bb60: 5a 6c        	ld	s8, 400(sp)
   1bb62: ba 6c        	ld	s9, 392(sp)
   1bb64: 1a 6d        	ld	s10, 384(sp)
   1bb66: de 86        	mv	a3, s7
   1bb68: 6e 86        	mv	a2, s11
   1bb6a: fa 6b        	ld	s7, 408(sp)
   1bb6c: f6 7d        	ld	s11, 376(sp)
   1bb6e: d2 85        	mv	a1, s4
   1bb70: 56 85        	mv	a0, s5
   1bb72: 5a 7a        	ld	s4, 432(sp)
   1bb74: ba 7a        	ld	s5, 424(sp)
   1bb76: 3d 61        	addi	sp, sp, 480
   1bb78: 6f 00 50 38  	j	0x1c6fc <__sbprintf>
   1bb7c: 90 18        	addi	a2, sp, 112
   1bb7e: d2 85        	mv	a1, s4
   1bb80: 56 85        	mv	a0, s5
   1bb82: ef f0 5f dd  	jal	0x1b956 <__sprint_r.part.0>
   1bb86: 63 02 05 16  	beqz	a0, 0x1bcea <_vfiprintf_r+0x2e2>
   1bb8a: 66 8c        	mv	s8, s9
   1bb8c: 83 57 0a 01  	lhu	a5, 16(s4)
   1bb90: 93 f7 07 04  	andi	a5, a5, 64
   1bb94: e3 97 07 32  	bnez	a5, 0x1c6c2 <_vfiprintf_r+0xcba>
   1bb98: fe 60        	ld	ra, 472(sp)
   1bb9a: 5e 64        	ld	s0, 464(sp)
   1bb9c: be 64        	ld	s1, 456(sp)
   1bb9e: 1e 69        	ld	s2, 448(sp)
   1bba0: fa 79        	ld	s3, 440(sp)
   1bba2: 5a 7a        	ld	s4, 432(sp)
   1bba4: ba 7a        	ld	s5, 424(sp)
   1bba6: 1a 7b        	ld	s6, 416(sp)
   1bba8: fa 6b        	ld	s7, 408(sp)
   1bbaa: ba 6c        	ld	s9, 392(sp)
   1bbac: 1a 6d        	ld	s10, 384(sp)
   1bbae: f6 7d        	ld	s11, 376(sp)
   1bbb0: 62 85        	mv	a0, s8
   1bbb2: 5a 6c        	ld	s8, 400(sp)
   1bbb4: 3d 61        	addi	sp, sp, 480
   1bbb6: 82 80        	ret
   1bbb8: 81 48        	mv	a7, zero
   1bbba: 25 46        	addi	a2, zero, 9
   1bbbc: 83 c6 0d 00  	lbu	a3, 0(s11)
   1bbc0: 1b 97 28 00  	slliw	a4, a7, 2
   1bbc4: bb 08 17 01  	addw	a7, a4, a7
   1bbc8: 9b 87 07 fd  	addiw	a5, a5, -48
   1bbcc: 9b 98 18 00  	slliw	a7, a7, 1
   1bbd0: 1b 87 06 fd  	addiw	a4, a3, -48
   1bbd4: bb 88 17 01  	addw	a7, a5, a7
   1bbd8: 85 0d        	addi	s11, s11, 1
   1bbda: 9b 87 06 00  	sext.w	a5, a3
   1bbde: e3 7f e6 fc  	bgeu	a2, a4, 0x1bbbc <_vfiprintf_r+0x1b4>
   1bbe2: 1b 87 07 fe  	addiw	a4, a5, -32
   1bbe6: 9b 06 07 00  	sext.w	a3, a4
   1bbea: e3 fd d4 f2  	bgeu	s1, a3, 0x1bb24 <_vfiprintf_r+0x11c>
   1bbee: 63 84 07 10  	beqz	a5, 0x1bcf6 <_vfiprintf_r+0x2ee>
   1bbf2: 23 04 f1 08  	sb	a5, 136(sp)
   1bbf6: a3 03 01 06  	sb	zero, 103(sp)
   1bbfa: 05 4e        	addi	t3, zero, 1
   1bbfc: 05 43        	addi	t1, zero, 1
   1bbfe: 24 01        	addi	s1, sp, 136
   1bc00: 01 4d        	mv	s10, zero
   1bc02: 93 ff 2b 00  	andi	t6, s7, 2
   1bc06: 63 83 0f 00  	beqz	t6, 0x1bc0c <_vfiprintf_r+0x204>
   1bc0a: 09 2e        	addiw	t3, t3, 2
   1bc0c: 66 57        	lw	a4, 120(sp)
   1bc0e: 93 f2 4b 08  	andi	t0, s7, 132
   1bc12: 8a 67        	ld	a5, 128(sp)
   1bc14: 9b 05 17 00  	addiw	a1, a4, 1
   1bc18: 2e 86        	mv	a2, a1
   1bc1a: 63 96 02 00  	bnez	t0, 0x1bc26 <_vfiprintf_r+0x21e>
   1bc1e: bb 86 c8 41  	subw	a3, a7, t3
   1bc22: 63 46 d0 62  	bgtz	a3, 0x1c24e <_vfiprintf_r+0x846>
   1bc26: 83 46 71 06  	lbu	a3, 103(sp)
   1bc2a: 99 ce        	beqz	a3, 0x1bc48 <_vfiprintf_r+0x240>
   1bc2c: 13 07 71 06  	addi	a4, sp, 103
   1bc30: 85 07        	addi	a5, a5, 1
   1bc32: 18 e0        	sd	a4, 0(s0)
   1bc34: 05 47        	addi	a4, zero, 1
   1bc36: 18 e4        	sd	a4, 8(s0)
   1bc38: 3e e1        	sd	a5, 128(sp)
   1bc3a: ae dc        	sw	a1, 120(sp)
   1bc3c: 1d 47        	addi	a4, zero, 7
   1bc3e: 63 49 b7 58  	blt	a4, a1, 0x1c1d0 <_vfiprintf_r+0x7c8>
   1bc42: 2e 87        	mv	a4, a1
   1bc44: 41 04        	addi	s0, s0, 16
   1bc46: 85 25        	addiw	a1, a1, 1
   1bc48: 63 81 0f 04  	beqz	t6, 0x1bc8a <_vfiprintf_r+0x282>
   1bc4c: b8 10        	addi	a4, sp, 104
   1bc4e: 89 07        	addi	a5, a5, 2
   1bc50: 18 e0        	sd	a4, 0(s0)
   1bc52: 09 47        	addi	a4, zero, 2
   1bc54: 18 e4        	sd	a4, 8(s0)
   1bc56: 3e e1        	sd	a5, 128(sp)
   1bc58: ae dc        	sw	a1, 120(sp)
   1bc5a: 1d 47        	addi	a4, zero, 7
   1bc5c: 63 5a b7 5a  	bge	a4, a1, 0x1c210 <_vfiprintf_r+0x808>
   1bc60: 63 84 07 78  	beqz	a5, 0x1c3e8 <_vfiprintf_r+0x9e0>
   1bc64: 90 18        	addi	a2, sp, 112
   1bc66: d2 85        	mv	a1, s4
   1bc68: 56 85        	mv	a0, s5
   1bc6a: 9a e4        	sd	t1, 72(sp)
   1bc6c: 72 f4        	sd	t3, 40(sp)
   1bc6e: 46 f0        	sd	a7, 32(sp)
   1bc70: 16 ec        	sd	t0, 24(sp)
   1bc72: ef f0 5f ce  	jal	0x1b956 <__sprint_r.part.0>
   1bc76: 11 f9        	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1bc78: 66 57        	lw	a4, 120(sp)
   1bc7a: 8a 67        	ld	a5, 128(sp)
   1bc7c: 26 63        	ld	t1, 72(sp)
   1bc7e: 22 7e        	ld	t3, 40(sp)
   1bc80: 82 78        	ld	a7, 32(sp)
   1bc82: e2 62        	ld	t0, 24(sp)
   1bc84: 4e 84        	mv	s0, s3
   1bc86: 9b 05 17 00  	addiw	a1, a4, 1
   1bc8a: 93 06 00 08  	addi	a3, zero, 128
   1bc8e: 63 83 d2 3e  	beq	t0, a3, 0x1c074 <_vfiprintf_r+0x66c>
   1bc92: 3b 0d 6d 40  	subw	s10, s10, t1
   1bc96: 63 41 a0 4b  	bgtz	s10, 0x1c138 <_vfiprintf_r+0x730>
   1bc9a: 9a 97        	add	a5, a5, t1
   1bc9c: 04 e0        	sd	s1, 0(s0)
   1bc9e: 23 34 64 00  	sd	t1, 8(s0)
   1bca2: 3e e1        	sd	a5, 128(sp)
   1bca4: ae dc        	sw	a1, 120(sp)
   1bca6: 1d 47        	addi	a4, zero, 7
   1bca8: 63 54 b7 3c  	bge	a4, a1, 0x1c070 <_vfiprintf_r+0x668>
   1bcac: 63 84 07 64  	beqz	a5, 0x1c2f4 <_vfiprintf_r+0x8ec>
   1bcb0: 90 18        	addi	a2, sp, 112
   1bcb2: d2 85        	mv	a1, s4
   1bcb4: 56 85        	mv	a0, s5
   1bcb6: 72 f0        	sd	t3, 32(sp)
   1bcb8: 46 ec        	sd	a7, 24(sp)
   1bcba: ef f0 df c9  	jal	0x1b956 <__sprint_r.part.0>
   1bcbe: e3 16 05 ec  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1bcc2: 8a 67        	ld	a5, 128(sp)
   1bcc4: 02 7e        	ld	t3, 32(sp)
   1bcc6: e2 68        	ld	a7, 24(sp)
   1bcc8: 4e 84        	mv	s0, s3
   1bcca: 93 fb 4b 00  	andi	s7, s7, 4
   1bcce: 63 86 0b 00  	beqz	s7, 0x1bcda <_vfiprintf_r+0x2d2>
   1bcd2: bb 84 c8 41  	subw	s1, a7, t3
   1bcd6: 63 49 90 62  	bgtz	s1, 0x1c308 <_vfiprintf_r+0x900>
   1bcda: 46 87        	mv	a4, a7
   1bcdc: 63 d3 c8 01  	bge	a7, t3, 0x1bce2 <_vfiprintf_r+0x2da>
   1bce0: 72 87        	mv	a4, t3
   1bce2: bb 0c 97 01  	addw	s9, a4, s9
   1bce6: e3 9b 07 e8  	bnez	a5, 0x1bb7c <_vfiprintf_r+0x174>
   1bcea: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bcee: 82 dc        	sw	zero, 120(sp)
   1bcf0: 4e 84        	mv	s0, s3
   1bcf2: e3 9b 07 da  	bnez	a5, 0x1baa8 <_vfiprintf_r+0xa0>
   1bcf6: 8a 67        	ld	a5, 128(sp)
   1bcf8: 66 8c        	mv	s8, s9
   1bcfa: e3 89 07 e8  	beqz	a5, 0x1bb8c <_vfiprintf_r+0x184>
   1bcfe: 90 18        	addi	a2, sp, 112
   1bd00: d2 85        	mv	a1, s4
   1bd02: 56 85        	mv	a0, s5
   1bd04: ef f0 3f c5  	jal	0x1b956 <__sprint_r.part.0>
   1bd08: 51 b5        	j	0x1bb8c <_vfiprintf_r+0x184>
   1bd0a: 93 eb 0b 01  	ori	s7, s7, 16
   1bd0e: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bd12: 81 2b        	sext.w	s7, s7
   1bd14: 01 b5        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bd16: 93 eb 0b 01  	ori	s7, s7, 16
   1bd1a: 81 2b        	sext.w	s7, s7
   1bd1c: 93 f7 0b 02  	andi	a5, s7, 32
   1bd20: 93 0e 8c 00  	addi	t4, s8, 8
   1bd24: 85 cf        	beqz	a5, 0x1bd5c <_vfiprintf_r+0x354>
   1bd26: 83 37 0c 00  	ld	a5, 0(s8)
   1bd2a: 3e 8c        	mv	s8, a5
   1bd2c: 63 c8 07 04  	bltz	a5, 0x1bd7c <_vfiprintf_r+0x374>
   1bd30: fd 57        	addi	a5, zero, -1
   1bd32: 63 01 fd 06  	beq	s10, a5, 0x1bd94 <_vfiprintf_r+0x38c>
   1bd36: 93 fb fb f7  	andi	s7, s7, -129
   1bd3a: 81 2b        	sext.w	s7, s7
   1bd3c: 63 1c 0c 04  	bnez	s8, 0x1bd94 <_vfiprintf_r+0x38c>
   1bd40: e3 1f 0d 14  	bnez	s10, 0x1c69e <_vfiprintf_r+0xc96>
   1bd44: 01 4e        	mv	t3, zero
   1bd46: 01 43        	mv	t1, zero
   1bd48: 76 8c        	mv	s8, t4
   1bd4a: e4 11        	addi	s1, sp, 236
   1bd4c: b5 a0        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1bd4e: bb 8b b4 41  	subw	s7, s1, s11
   1bd52: e3 98 0b d6  	bnez	s7, 0x1bac2 <_vfiprintf_r+0xba>
   1bd56: 83 c7 04 00  	lbu	a5, 0(s1)
   1bd5a: 45 b3        	j	0x1bafa <_vfiprintf_r+0xf2>
   1bd5c: 93 f7 0b 01  	andi	a5, s7, 16
   1bd60: f9 f3        	bnez	a5, 0x1bd26 <_vfiprintf_r+0x31e>
   1bd62: 93 f7 0b 04  	andi	a5, s7, 64
   1bd66: 03 2c 0c 00  	lw	s8, 0(s8)
   1bd6a: e3 89 07 0c  	beqz	a5, 0x1c63c <_vfiprintf_r+0xc34>
   1bd6e: 1b 1c 0c 01  	slliw	s8, s8, 16
   1bd72: 1b 5c 0c 41  	sraiw	s8, s8, 16
   1bd76: e2 87        	mv	a5, s8
   1bd78: e3 dc 07 fa  	bgez	a5, 0x1bd30 <_vfiprintf_r+0x328>
   1bd7c: 93 07 d0 02  	addi	a5, zero, 45
   1bd80: a3 03 f1 06  	sb	a5, 103(sp)
   1bd84: fd 57        	addi	a5, zero, -1
   1bd86: 33 0c 80 41  	neg	s8, s8
   1bd8a: 63 05 fd 00  	beq	s10, a5, 0x1bd94 <_vfiprintf_r+0x38c>
   1bd8e: 93 fb fb f7  	andi	s7, s7, -129
   1bd92: 81 2b        	sext.w	s7, s7
   1bd94: a5 47        	addi	a5, zero, 9
   1bd96: 63 eb 87 69  	bltu	a5, s8, 0x1c42c <_vfiprintf_r+0xa24>
   1bd9a: 9b 07 0c 03  	addiw	a5, s8, 48
   1bd9e: 93 f7 f7 0f  	andi	a5, a5, 255
   1bda2: 76 8c        	mv	s8, t4
   1bda4: a3 05 f1 0e  	sb	a5, 235(sp)
   1bda8: 1b 0e 0d 00  	sext.w	t3, s10
   1bdac: 63 43 a0 01  	bgtz	s10, 0x1bdb2 <_vfiprintf_r+0x3aa>
   1bdb0: 05 4e        	addi	t3, zero, 1
   1bdb2: 05 43        	addi	t1, zero, 1
   1bdb4: 93 04 b1 0e  	addi	s1, sp, 235
   1bdb8: 83 47 71 06  	lbu	a5, 103(sp)
   1bdbc: e3 83 07 e4  	beqz	a5, 0x1bc02 <_vfiprintf_r+0x1fa>
   1bdc0: 05 2e        	addiw	t3, t3, 1
   1bdc2: 81 b5        	j	0x1bc02 <_vfiprintf_r+0x1fa>
   1bdc4: ef 90 ff 96  	jal	0x15732 <__sinit>
   1bdc8: b5 b1        	j	0x1ba34 <_vfiprintf_r+0x2c>
   1bdca: 83 34 0c 00  	ld	s1, 0(s8)
   1bdce: a3 03 01 06  	sb	zero, 103(sp)
   1bdd2: 21 0c        	addi	s8, s8, 8
   1bdd4: e3 84 04 02  	beqz	s1, 0x1c5fc <_vfiprintf_r+0xbf4>
   1bdd8: fd 57        	addi	a5, zero, -1
   1bdda: 46 ec        	sd	a7, 24(sp)
   1bddc: 63 02 fd 78  	beq	s10, a5, 0x1c560 <_vfiprintf_r+0xb58>
   1bde0: 6a 86        	mv	a2, s10
   1bde2: 81 45        	mv	a1, zero
   1bde4: 26 85        	mv	a0, s1
   1bde6: ef c0 7f d8  	jal	0x18b6c <memchr>
   1bdea: e2 68        	ld	a7, 24(sp)
   1bdec: e3 0f 05 08  	beqz	a0, 0x1c68a <_vfiprintf_r+0xc82>
   1bdf0: 3b 03 95 40  	subw	t1, a0, s1
   1bdf4: 13 4e f3 ff  	not	t3, t1
   1bdf8: 13 5e fe 43  	srai	t3, t3, 63
   1bdfc: 33 7e 6e 00  	and	t3, t3, t1
   1be00: 01 4d        	mv	s10, zero
   1be02: 5d bf        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1be04: 83 27 0c 00  	lw	a5, 0(s8)
   1be08: a3 03 01 06  	sb	zero, 103(sp)
   1be0c: 21 0c        	addi	s8, s8, 8
   1be0e: 23 04 f1 08  	sb	a5, 136(sp)
   1be12: 05 4e        	addi	t3, zero, 1
   1be14: 05 43        	addi	t1, zero, 1
   1be16: 24 01        	addi	s1, sp, 136
   1be18: e5 b3        	j	0x1bc00 <_vfiprintf_r+0x1f8>
   1be1a: 83 28 0c 00  	lw	a7, 0(s8)
   1be1e: 21 0c        	addi	s8, s8, 8
   1be20: 63 db 08 08  	bgez	a7, 0x1beb6 <_vfiprintf_r+0x4ae>
   1be24: bb 08 10 41  	negw	a7, a7
   1be28: 93 eb 4b 00  	ori	s7, s7, 4
   1be2c: 83 c7 0d 00  	lbu	a5, 0(s11)
   1be30: 81 2b        	sext.w	s7, s7
   1be32: cd b1        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1be34: 13 07 b0 02  	addi	a4, zero, 43
   1be38: 83 c7 0d 00  	lbu	a5, 0(s11)
   1be3c: a3 03 e1 06  	sb	a4, 103(sp)
   1be40: d1 b9        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1be42: 93 eb 0b 08  	ori	s7, s7, 128
   1be46: 83 c7 0d 00  	lbu	a5, 0(s11)
   1be4a: 81 2b        	sext.w	s7, s7
   1be4c: e1 b1        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1be4e: 83 c7 0d 00  	lbu	a5, 0(s11)
   1be52: 13 06 a0 02  	addi	a2, zero, 42
   1be56: 13 87 1d 00  	addi	a4, s11, 1
   1be5a: e3 8c c7 06  	beq	a5, a2, 0x1c6d2 <_vfiprintf_r+0xcca>
   1be5e: 9b 86 07 fd  	addiw	a3, a5, -48
   1be62: a5 45        	addi	a1, zero, 9
   1be64: ba 8d        	mv	s11, a4
   1be66: 01 4d        	mv	s10, zero
   1be68: 25 46        	addi	a2, zero, 9
   1be6a: e3 e7 d5 ca  	bltu	a1, a3, 0x1bb18 <_vfiprintf_r+0x110>
   1be6e: 83 c7 0d 00  	lbu	a5, 0(s11)
   1be72: 1b 17 2d 00  	slliw	a4, s10, 2
   1be76: 3b 0d a7 01  	addw	s10, a4, s10
   1be7a: 1b 1d 1d 00  	slliw	s10, s10, 1
   1be7e: 3b 0d dd 00  	addw	s10, s10, a3
   1be82: 9b 86 07 fd  	addiw	a3, a5, -48
   1be86: 85 0d        	addi	s11, s11, 1
   1be88: e3 73 d6 fe  	bgeu	a2, a3, 0x1be6e <_vfiprintf_r+0x466>
   1be8c: 71 b1        	j	0x1bb18 <_vfiprintf_r+0x110>
   1be8e: 56 85        	mv	a0, s5
   1be90: 46 ec        	sd	a7, 24(sp)
   1be92: ef c0 cf c3  	jal	0x182ce <_localeconv_r>
   1be96: 1c 65        	ld	a5, 8(a0)
   1be98: 3e 85        	mv	a0, a5
   1be9a: be e0        	sd	a5, 64(sp)
   1be9c: ef d0 3f e2  	jal	0x19cbe <strlen>
   1bea0: aa 87        	mv	a5, a0
   1bea2: 56 85        	mv	a0, s5
   1bea4: 3e f8        	sd	a5, 48(sp)
   1bea6: ef c0 8f c2  	jal	0x182ce <_localeconv_r>
   1beaa: 18 69        	ld	a4, 16(a0)
   1beac: c2 77        	ld	a5, 48(sp)
   1beae: e2 68        	ld	a7, 24(sp)
   1beb0: 3a fc        	sd	a4, 56(sp)
   1beb2: 63 99 07 60  	bnez	a5, 0x1c4c4 <_vfiprintf_r+0xabc>
   1beb6: 83 c7 0d 00  	lbu	a5, 0(s11)
   1beba: a9 b9        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bebc: 93 eb 1b 00  	ori	s7, s7, 1
   1bec0: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bec4: 81 2b        	sext.w	s7, s7
   1bec6: b9 b1        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bec8: 03 47 71 06  	lbu	a4, 103(sp)
   1becc: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bed0: e3 12 07 c4  	bnez	a4, 0x1bb14 <_vfiprintf_r+0x10c>
   1bed4: 13 07 00 02  	addi	a4, zero, 32
   1bed8: a3 03 e1 06  	sb	a4, 103(sp)
   1bedc: 25 b9        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bede: 93 e7 0b 01  	ori	a5, s7, 16
   1bee2: 81 27        	sext.w	a5, a5
   1bee4: 13 f7 07 02  	andi	a4, a5, 32
   1bee8: 13 06 8c 00  	addi	a2, s8, 8
   1beec: 09 e7        	bnez	a4, 0x1bef6 <_vfiprintf_r+0x4ee>
   1beee: 13 f7 07 01  	andi	a4, a5, 16
   1bef2: 63 00 07 5c  	beqz	a4, 0x1c4b2 <_vfiprintf_r+0xaaa>
   1bef6: 03 37 0c 00  	ld	a4, 0(s8)
   1befa: 93 fb f7 bf  	andi	s7, a5, -1025
   1befe: a3 03 01 06  	sb	zero, 103(sp)
   1bf02: fd 56        	addi	a3, zero, -1
   1bf04: 81 2b        	sext.w	s7, s7
   1bf06: 32 8c        	mv	s8, a2
   1bf08: 63 01 dd 0a  	beq	s10, a3, 0x1bfaa <_vfiprintf_r+0x5a2>
   1bf0c: 93 f7 f7 b7  	andi	a5, a5, -1153
   1bf10: 81 27        	sext.w	a5, a5
   1bf12: 63 10 07 6c  	bnez	a4, 0x1c5d2 <_vfiprintf_r+0xbca>
   1bf16: 5e 83        	mv	t1, s7
   1bf18: be 8b        	mv	s7, a5
   1bf1a: 63 11 0d 08  	bnez	s10, 0x1bf9c <_vfiprintf_r+0x594>
   1bf1e: 63 13 07 50  	bnez	a4, 0x1c424 <_vfiprintf_r+0xa1c>
   1bf22: 13 73 13 00  	andi	t1, t1, 1
   1bf26: 63 03 03 56  	beqz	t1, 0x1c48c <_vfiprintf_r+0xa84>
   1bf2a: 93 07 00 03  	addi	a5, zero, 48
   1bf2e: a3 05 f1 0e  	sb	a5, 235(sp)
   1bf32: 1a 8e        	mv	t3, t1
   1bf34: 93 04 b1 0e  	addi	s1, sp, 235
   1bf38: 41 b5        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1bf3a: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bf3e: 13 07 c0 06  	addi	a4, zero, 108
   1bf42: 63 85 e7 6a  	beq	a5, a4, 0x1c5ec <_vfiprintf_r+0xbe4>
   1bf46: 93 eb 0b 01  	ori	s7, s7, 16
   1bf4a: 81 2b        	sext.w	s7, s7
   1bf4c: e1 b6        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bf4e: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bf52: 13 07 80 06  	addi	a4, zero, 104
   1bf56: 63 83 e7 68  	beq	a5, a4, 0x1c5dc <_vfiprintf_r+0xbd4>
   1bf5a: 93 eb 0b 04  	ori	s7, s7, 64
   1bf5e: 81 2b        	sext.w	s7, s7
   1bf60: 55 be        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bf62: 13 e3 0b 01  	ori	t1, s7, 16
   1bf66: 01 23        	sext.w	t1, t1
   1bf68: 93 77 03 02  	andi	a5, t1, 32
   1bf6c: 93 0e 8c 00  	addi	t4, s8, 8
   1bf70: 89 e7        	bnez	a5, 0x1bf7a <_vfiprintf_r+0x572>
   1bf72: 93 77 03 01  	andi	a5, t1, 16
   1bf76: 63 84 07 52  	beqz	a5, 0x1c49e <_vfiprintf_r+0xa96>
   1bf7a: 03 3c 0c 00  	ld	s8, 0(s8)
   1bf7e: a3 03 01 06  	sb	zero, 103(sp)
   1bf82: fd 57        	addi	a5, zero, -1
   1bf84: 9a 8b        	mv	s7, t1
   1bf86: e3 07 fd e0  	beq	s10, a5, 0x1bd94 <_vfiprintf_r+0x38c>
   1bf8a: 93 7b f3 f7  	andi	s7, t1, -129
   1bf8e: 81 2b        	sext.w	s7, s7
   1bf90: e3 12 0c e0  	bnez	s8, 0x1bd94 <_vfiprintf_r+0x38c>
   1bf94: 76 8c        	mv	s8, t4
   1bf96: 05 47        	addi	a4, zero, 1
   1bf98: e3 03 0d f8  	beqz	s10, 0x1bf1e <_vfiprintf_r+0x516>
   1bf9c: 85 47        	addi	a5, zero, 1
   1bf9e: 63 01 f7 70  	beq	a4, a5, 0x1c6a0 <_vfiprintf_r+0xc98>
   1bfa2: 89 47        	addi	a5, zero, 2
   1bfa4: 63 08 f7 66  	beq	a4, a5, 0x1c614 <_vfiprintf_r+0xc0c>
   1bfa8: 01 47        	mv	a4, zero
   1bfaa: 13 03 c1 0e  	addi	t1, sp, 236
   1bfae: 9a 84        	mv	s1, t1
   1bfb0: 93 77 77 00  	andi	a5, a4, 7
   1bfb4: 93 87 07 03  	addi	a5, a5, 48
   1bfb8: a3 8f f4 fe  	sb	a5, -1(s1)
   1bfbc: 0d 83        	srli	a4, a4, 3
   1bfbe: a6 86        	mv	a3, s1
   1bfc0: fd 14        	addi	s1, s1, -1
   1bfc2: 7d f7        	bnez	a4, 0x1bfb0 <_vfiprintf_r+0x5a8>
   1bfc4: 13 f7 1b 00  	andi	a4, s7, 1
   1bfc8: 63 0a 07 26  	beqz	a4, 0x1c23c <_vfiprintf_r+0x834>
   1bfcc: 13 07 00 03  	addi	a4, zero, 48
   1bfd0: 63 86 e7 26  	beq	a5, a4, 0x1c23c <_vfiprintf_r+0x834>
   1bfd4: f9 16        	addi	a3, a3, -2
   1bfd6: 3b 03 d3 40  	subw	t1, t1, a3
   1bfda: a3 8f e4 fe  	sb	a4, -1(s1)
   1bfde: 1b 0e 03 00  	sext.w	t3, t1
   1bfe2: 63 43 a3 51  	blt	t1, s10, 0x1c4e8 <_vfiprintf_r+0xae0>
   1bfe6: b6 84        	mv	s1, a3
   1bfe8: c1 bb        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1bfea: 93 eb 0b 02  	ori	s7, s7, 32
   1bfee: 83 c7 0d 00  	lbu	a5, 0(s11)
   1bff2: 81 2b        	sext.w	s7, s7
   1bff4: 05 b6        	j	0x1bb14 <_vfiprintf_r+0x10c>
   1bff6: e1 77        	lui	a5, 1048568
   1bff8: 93 c7 07 83  	xori	a5, a5, -2000
   1bffc: 23 14 f1 06  	sh	a5, 104(sp)
   1c000: b7 27 02 00  	lui	a5, 34
   1c004: 93 eb 2b 00  	ori	s7, s7, 2
   1c008: 93 87 87 c0  	addi	a5, a5, -1016
   1c00c: 03 37 0c 00  	ld	a4, 0(s8)
   1c010: 81 2b        	sext.w	s7, s7
   1c012: 21 0c        	addi	s8, s8, 8
   1c014: 3e e8        	sd	a5, 16(sp)
   1c016: a3 03 01 06  	sb	zero, 103(sp)
   1c01a: fd 57        	addi	a5, zero, -1
   1c01c: 63 02 fd 20  	beq	s10, a5, 0x1c220 <_vfiprintf_r+0x818>
   1c020: 93 f7 fb f7  	andi	a5, s7, -129
   1c024: 81 27        	sext.w	a5, a5
   1c026: 63 19 07 1e  	bnez	a4, 0x1c218 <_vfiprintf_r+0x810>
   1c02a: 5e 83        	mv	t1, s7
   1c02c: 09 47        	addi	a4, zero, 2
   1c02e: be 8b        	mv	s7, a5
   1c030: ed b5        	j	0x1bf1a <_vfiprintf_r+0x512>
   1c032: 93 f7 0b 02  	andi	a5, s7, 32
   1c036: 03 37 0c 00  	ld	a4, 0(s8)
   1c03a: 85 e3        	bnez	a5, 0x1c05a <_vfiprintf_r+0x652>
   1c03c: 93 f7 0b 01  	andi	a5, s7, 16
   1c040: 89 ef        	bnez	a5, 0x1c05a <_vfiprintf_r+0x652>
   1c042: 93 f7 0b 04  	andi	a5, s7, 64
   1c046: 63 90 07 62  	bnez	a5, 0x1c666 <_vfiprintf_r+0xc5e>
   1c04a: 93 fb 0b 20  	andi	s7, s7, 512
   1c04e: 63 8d 0b 66  	beqz	s7, 0x1c6c8 <_vfiprintf_r+0xcc0>
   1c052: 23 00 97 01  	sb	s9, 0(a4)
   1c056: 21 0c        	addi	s8, s8, 8
   1c058: a1 b4        	j	0x1baa0 <_vfiprintf_r+0x98>
   1c05a: 23 30 97 01  	sd	s9, 0(a4)
   1c05e: 21 0c        	addi	s8, s8, 8
   1c060: 81 b4        	j	0x1baa0 <_vfiprintf_r+0x98>
   1c062: 05 47        	addi	a4, zero, 1
   1c064: 9a 87        	mv	a5, t1
   1c066: a6 f9        	sd	s1, 240(sp)
   1c068: 9a fd        	sd	t1, 248(sp)
   1c06a: 1a e1        	sd	t1, 128(sp)
   1c06c: ba dc        	sw	a4, 120(sp)
   1c06e: 4e 84        	mv	s0, s3
   1c070: 41 04        	addi	s0, s0, 16
   1c072: a1 b9        	j	0x1bcca <_vfiprintf_r+0x2c2>
   1c074: bb 8e c8 41  	subw	t4, a7, t3
   1c078: e3 5d d0 c1  	blez	t4, 0x1bc92 <_vfiprintf_r+0x28a>
   1c07c: c1 46        	addi	a3, zero, 16
   1c07e: 63 dc d6 63  	bge	a3, t4, 0x1c6b6 <_vfiprintf_r+0xcae>
   1c082: 37 2f 02 00  	lui	t5, 34
   1c086: 13 0f 0f 6d  	addi	t5, t5, 1744
   1c08a: 26 ec        	sd	s1, 24(sp)
   1c08c: 22 86        	mv	a2, s0
   1c08e: c1 4f        	addi	t6, zero, 16
   1c090: 9d 42        	addi	t0, zero, 7
   1c092: 46 f0        	sd	a7, 32(sp)
   1c094: 72 f4        	sd	t3, 40(sp)
   1c096: 9a e4        	sd	t1, 72(sp)
   1c098: 76 84        	mv	s0, t4
   1c09a: fa 84        	mv	s1, t5
   1c09c: 01 a8        	j	0x1c0ac <_vfiprintf_r+0x6a4>
   1c09e: 9b 06 27 00  	addiw	a3, a4, 2
   1c0a2: 41 06        	addi	a2, a2, 16
   1c0a4: 2e 87        	mv	a4, a1
   1c0a6: 41 34        	addiw	s0, s0, -16
   1c0a8: 63 df 8f 02  	bge	t6, s0, 0x1c0e6 <_vfiprintf_r+0x6de>
   1c0ac: 9b 05 17 00  	addiw	a1, a4, 1
   1c0b0: c1 07        	addi	a5, a5, 16
   1c0b2: 04 e2        	sd	s1, 0(a2)
   1c0b4: 23 34 f6 01  	sd	t6, 8(a2)
   1c0b8: 3e e1        	sd	a5, 128(sp)
   1c0ba: ae dc        	sw	a1, 120(sp)
   1c0bc: e3 d1 b2 fe  	bge	t0, a1, 0x1c09e <_vfiprintf_r+0x696>
   1c0c0: 63 84 07 10  	beqz	a5, 0x1c1c8 <_vfiprintf_r+0x7c0>
   1c0c4: 90 18        	addi	a2, sp, 112
   1c0c6: d2 85        	mv	a1, s4
   1c0c8: 56 85        	mv	a0, s5
   1c0ca: ef f0 df 88  	jal	0x1b956 <__sprint_r.part.0>
   1c0ce: e3 1e 05 aa  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c0d2: 66 57        	lw	a4, 120(sp)
   1c0d4: c1 4f        	addi	t6, zero, 16
   1c0d6: 41 34        	addiw	s0, s0, -16
   1c0d8: 8a 67        	ld	a5, 128(sp)
   1c0da: 4e 86        	mv	a2, s3
   1c0dc: 9b 06 17 00  	addiw	a3, a4, 1
   1c0e0: 9d 42        	addi	t0, zero, 7
   1c0e2: e3 c5 8f fc  	blt	t6, s0, 0x1c0ac <_vfiprintf_r+0x6a4>
   1c0e6: 26 8f        	mv	t5, s1
   1c0e8: 82 78        	ld	a7, 32(sp)
   1c0ea: 22 7e        	ld	t3, 40(sp)
   1c0ec: 26 63        	ld	t1, 72(sp)
   1c0ee: e2 64        	ld	s1, 24(sp)
   1c0f0: a2 8e        	mv	t4, s0
   1c0f2: 32 84        	mv	s0, a2
   1c0f4: f6 97        	add	a5, a5, t4
   1c0f6: 23 30 e4 01  	sd	t5, 0(s0)
   1c0fa: 23 34 d4 01  	sd	t4, 8(s0)
   1c0fe: 3e e1        	sd	a5, 128(sp)
   1c100: b6 dc        	sw	a3, 120(sp)
   1c102: 1d 47        	addi	a4, zero, 7
   1c104: 63 57 d7 38  	bge	a4, a3, 0x1c492 <_vfiprintf_r+0xa8a>
   1c108: 63 84 07 54  	beqz	a5, 0x1c650 <_vfiprintf_r+0xc48>
   1c10c: 90 18        	addi	a2, sp, 112
   1c10e: d2 85        	mv	a1, s4
   1c110: 56 85        	mv	a0, s5
   1c112: 1a f4        	sd	t1, 40(sp)
   1c114: 72 f0        	sd	t3, 32(sp)
   1c116: 46 ec        	sd	a7, 24(sp)
   1c118: ef f0 ff 83  	jal	0x1b956 <__sprint_r.part.0>
   1c11c: e3 17 05 a6  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c120: 22 73        	ld	t1, 40(sp)
   1c122: 66 57        	lw	a4, 120(sp)
   1c124: 8a 67        	ld	a5, 128(sp)
   1c126: 3b 0d 6d 40  	subw	s10, s10, t1
   1c12a: 02 7e        	ld	t3, 32(sp)
   1c12c: e2 68        	ld	a7, 24(sp)
   1c12e: 4e 84        	mv	s0, s3
   1c130: 9b 05 17 00  	addiw	a1, a4, 1
   1c134: e3 53 a0 b7  	blez	s10, 0x1bc9a <_vfiprintf_r+0x292>
   1c138: 37 2f 02 00  	lui	t5, 34
   1c13c: 13 0f 0f 6d  	addi	t5, t5, 1744
   1c140: 63 53 a9 07  	bge	s2, s10, 0x1c1a6 <_vfiprintf_r+0x79e>
   1c144: a2 86        	mv	a3, s0
   1c146: 9d 4e        	addi	t4, zero, 7
   1c148: 6a 84        	mv	s0, s10
   1c14a: 46 ec        	sd	a7, 24(sp)
   1c14c: 72 f0        	sd	t3, 32(sp)
   1c14e: 1a f4        	sd	t1, 40(sp)
   1c150: 5a 8d        	mv	s10, s6
   1c152: 01 a8        	j	0x1c162 <_vfiprintf_r+0x75a>
   1c154: 9b 05 27 00  	addiw	a1, a4, 2
   1c158: c1 06        	addi	a3, a3, 16
   1c15a: 2a 87        	mv	a4, a0
   1c15c: 41 34        	addiw	s0, s0, -16
   1c15e: 63 5e 89 02  	bge	s2, s0, 0x1c19a <_vfiprintf_r+0x792>
   1c162: 1b 05 17 00  	addiw	a0, a4, 1
   1c166: c1 07        	addi	a5, a5, 16
   1c168: 23 b0 66 01  	sd	s6, 0(a3)
   1c16c: 23 b4 26 01  	sd	s2, 8(a3)
   1c170: 3e e1        	sd	a5, 128(sp)
   1c172: aa dc        	sw	a0, 120(sp)
   1c174: e3 d0 ae fe  	bge	t4, a0, 0x1c154 <_vfiprintf_r+0x74c>
   1c178: a1 c7        	beqz	a5, 0x1c1c0 <_vfiprintf_r+0x7b8>
   1c17a: 90 18        	addi	a2, sp, 112
   1c17c: d2 85        	mv	a1, s4
   1c17e: 56 85        	mv	a0, s5
   1c180: ef f0 6f fd  	jal	0x1b956 <__sprint_r.part.0>
   1c184: e3 13 05 a0  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c188: 66 57        	lw	a4, 120(sp)
   1c18a: 41 34        	addiw	s0, s0, -16
   1c18c: 8a 67        	ld	a5, 128(sp)
   1c18e: ce 86        	mv	a3, s3
   1c190: 9b 05 17 00  	addiw	a1, a4, 1
   1c194: 9d 4e        	addi	t4, zero, 7
   1c196: e3 46 89 fc  	blt	s2, s0, 0x1c162 <_vfiprintf_r+0x75a>
   1c19a: e2 68        	ld	a7, 24(sp)
   1c19c: 02 7e        	ld	t3, 32(sp)
   1c19e: 22 73        	ld	t1, 40(sp)
   1c1a0: 6a 8f        	mv	t5, s10
   1c1a2: 22 8d        	mv	s10, s0
   1c1a4: 36 84        	mv	s0, a3
   1c1a6: ea 97        	add	a5, a5, s10
   1c1a8: 23 30 e4 01  	sd	t5, 0(s0)
   1c1ac: 23 34 a4 01  	sd	s10, 8(s0)
   1c1b0: 3e e1        	sd	a5, 128(sp)
   1c1b2: ae dc        	sw	a1, 120(sp)
   1c1b4: 1d 47        	addi	a4, zero, 7
   1c1b6: 63 41 b7 20  	blt	a4, a1, 0x1c3b8 <_vfiprintf_r+0x9b0>
   1c1ba: 41 04        	addi	s0, s0, 16
   1c1bc: 85 25        	addiw	a1, a1, 1
   1c1be: f1 bc        	j	0x1bc9a <_vfiprintf_r+0x292>
   1c1c0: 85 45        	addi	a1, zero, 1
   1c1c2: 01 47        	mv	a4, zero
   1c1c4: ce 86        	mv	a3, s3
   1c1c6: 59 bf        	j	0x1c15c <_vfiprintf_r+0x754>
   1c1c8: 85 46        	addi	a3, zero, 1
   1c1ca: 01 47        	mv	a4, zero
   1c1cc: 4e 86        	mv	a2, s3
   1c1ce: e1 bd        	j	0x1c0a6 <_vfiprintf_r+0x69e>
   1c1d0: 85 cb        	beqz	a5, 0x1c200 <_vfiprintf_r+0x7f8>
   1c1d2: 90 18        	addi	a2, sp, 112
   1c1d4: d2 85        	mv	a1, s4
   1c1d6: 56 85        	mv	a0, s5
   1c1d8: 9a e8        	sd	t1, 80(sp)
   1c1da: f2 e4        	sd	t3, 72(sp)
   1c1dc: 46 f4        	sd	a7, 40(sp)
   1c1de: 16 f0        	sd	t0, 32(sp)
   1c1e0: 7e ec        	sd	t6, 24(sp)
   1c1e2: ef f0 4f f7  	jal	0x1b956 <__sprint_r.part.0>
   1c1e6: e3 12 05 9a  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c1ea: 66 57        	lw	a4, 120(sp)
   1c1ec: 8a 67        	ld	a5, 128(sp)
   1c1ee: 46 63        	ld	t1, 80(sp)
   1c1f0: 26 6e        	ld	t3, 72(sp)
   1c1f2: a2 78        	ld	a7, 40(sp)
   1c1f4: 82 72        	ld	t0, 32(sp)
   1c1f6: e2 6f        	ld	t6, 24(sp)
   1c1f8: 4e 84        	mv	s0, s3
   1c1fa: 9b 05 17 00  	addiw	a1, a4, 1
   1c1fe: a9 b4        	j	0x1bc48 <_vfiprintf_r+0x240>
   1c200: 63 8f 0f 2c  	beqz	t6, 0x1c4de <_vfiprintf_r+0xad6>
   1c204: bc 10        	addi	a5, sp, 104
   1c206: be f9        	sd	a5, 240(sp)
   1c208: 89 47        	addi	a5, zero, 2
   1c20a: be fd        	sd	a5, 248(sp)
   1c20c: 85 45        	addi	a1, zero, 1
   1c20e: 4e 84        	mv	s0, s3
   1c210: 2e 87        	mv	a4, a1
   1c212: 41 04        	addi	s0, s0, 16
   1c214: 85 25        	addiw	a1, a1, 1
   1c216: 95 bc        	j	0x1bc8a <_vfiprintf_r+0x282>
   1c218: be 8b        	mv	s7, a5
   1c21a: 89 47        	addi	a5, zero, 2
   1c21c: e3 87 07 d8  	beqz	a5, 0x1bfaa <_vfiprintf_r+0x5a2>
   1c220: c2 66        	ld	a3, 16(sp)
   1c222: 13 03 c1 0e  	addi	t1, sp, 236
   1c226: 9a 84        	mv	s1, t1
   1c228: 93 77 f7 00  	andi	a5, a4, 15
   1c22c: b6 97        	add	a5, a5, a3
   1c22e: 83 c7 07 00  	lbu	a5, 0(a5)
   1c232: 11 83        	srli	a4, a4, 4
   1c234: fd 14        	addi	s1, s1, -1
   1c236: 23 80 f4 00  	sb	a5, 0(s1)
   1c23a: 7d f7        	bnez	a4, 0x1c228 <_vfiprintf_r+0x820>
   1c23c: 3b 03 93 40  	subw	t1, t1, s1
   1c240: 1b 0e 03 00  	sext.w	t3, t1
   1c244: e3 5a a3 b7  	bge	t1, s10, 0x1bdb8 <_vfiprintf_r+0x3b0>
   1c248: 1b 0e 0d 00  	sext.w	t3, s10
   1c24c: b5 b6        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c24e: 41 45        	addi	a0, zero, 16
   1c250: 63 52 d5 44  	bge	a0, a3, 0x1c694 <_vfiprintf_r+0xc8c>
   1c254: b7 2e 02 00  	lui	t4, 34
   1c258: 93 8e 0e 6e  	addi	t4, t4, 1760
   1c25c: 26 f4        	sd	s1, 40(sp)
   1c25e: a2 85        	mv	a1, s0
   1c260: 41 4f        	addi	t5, zero, 16
   1c262: 9d 43        	addi	t2, zero, 7
   1c264: 7e ec        	sd	t6, 24(sp)
   1c266: 16 f0        	sd	t0, 32(sp)
   1c268: c6 e4        	sd	a7, 72(sp)
   1c26a: f2 e8        	sd	t3, 80(sp)
   1c26c: 9a ec        	sd	t1, 88(sp)
   1c26e: 36 84        	mv	s0, a3
   1c270: f6 84        	mv	s1, t4
   1c272: 11 a8        	j	0x1c286 <_vfiprintf_r+0x87e>
   1c274: 1b 06 27 00  	addiw	a2, a4, 2
   1c278: c1 05        	addi	a1, a1, 16
   1c27a: 2a 87        	mv	a4, a0
   1c27c: 41 34        	addiw	s0, s0, -16
   1c27e: 63 50 8f 04  	bge	t5, s0, 0x1c2be <_vfiprintf_r+0x8b6>
   1c282: 1b 06 17 00  	addiw	a2, a4, 1
   1c286: c1 07        	addi	a5, a5, 16
   1c288: 84 e1        	sd	s1, 0(a1)
   1c28a: 23 b4 e5 01  	sd	t5, 8(a1)
   1c28e: 3e e1        	sd	a5, 128(sp)
   1c290: 1b 05 06 00  	sext.w	a0, a2
   1c294: b2 dc        	sw	a2, 120(sp)
   1c296: e3 df a3 fc  	bge	t2, a0, 0x1c274 <_vfiprintf_r+0x86c>
   1c29a: a9 cb        	beqz	a5, 0x1c2ec <_vfiprintf_r+0x8e4>
   1c29c: 90 18        	addi	a2, sp, 112
   1c29e: d2 85        	mv	a1, s4
   1c2a0: 56 85        	mv	a0, s5
   1c2a2: ef f0 4f eb  	jal	0x1b956 <__sprint_r.part.0>
   1c2a6: e3 12 05 8e  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c2aa: 66 57        	lw	a4, 120(sp)
   1c2ac: 41 4f        	addi	t5, zero, 16
   1c2ae: 41 34        	addiw	s0, s0, -16
   1c2b0: 8a 67        	ld	a5, 128(sp)
   1c2b2: ce 85        	mv	a1, s3
   1c2b4: 1b 06 17 00  	addiw	a2, a4, 1
   1c2b8: 9d 43        	addi	t2, zero, 7
   1c2ba: e3 44 8f fc  	blt	t5, s0, 0x1c282 <_vfiprintf_r+0x87a>
   1c2be: a6 8e        	mv	t4, s1
   1c2c0: e2 6f        	ld	t6, 24(sp)
   1c2c2: 82 72        	ld	t0, 32(sp)
   1c2c4: a6 68        	ld	a7, 72(sp)
   1c2c6: 46 6e        	ld	t3, 80(sp)
   1c2c8: 66 63        	ld	t1, 88(sp)
   1c2ca: a2 74        	ld	s1, 40(sp)
   1c2cc: a2 86        	mv	a3, s0
   1c2ce: 2e 84        	mv	s0, a1
   1c2d0: b6 97        	add	a5, a5, a3
   1c2d2: 23 30 d4 01  	sd	t4, 0(s0)
   1c2d6: 14 e4        	sd	a3, 8(s0)
   1c2d8: 3e e1        	sd	a5, 128(sp)
   1c2da: b2 dc        	sw	a2, 120(sp)
   1c2dc: 1d 47        	addi	a4, zero, 7
   1c2de: 63 49 c7 10  	blt	a4, a2, 0x1c3f0 <_vfiprintf_r+0x9e8>
   1c2e2: 41 04        	addi	s0, s0, 16
   1c2e4: 9b 05 16 00  	addiw	a1, a2, 1
   1c2e8: 32 87        	mv	a4, a2
   1c2ea: 35 ba        	j	0x1bc26 <_vfiprintf_r+0x21e>
   1c2ec: 01 47        	mv	a4, zero
   1c2ee: 05 46        	addi	a2, zero, 1
   1c2f0: ce 85        	mv	a1, s3
   1c2f2: 69 b7        	j	0x1c27c <_vfiprintf_r+0x874>
   1c2f4: 82 dc        	sw	zero, 120(sp)
   1c2f6: 93 fb 4b 00  	andi	s7, s7, 4
   1c2fa: 63 88 0b 0a  	beqz	s7, 0x1c3aa <_vfiprintf_r+0x9a2>
   1c2fe: bb 84 c8 41  	subw	s1, a7, t3
   1c302: 63 54 90 0a  	blez	s1, 0x1c3aa <_vfiprintf_r+0x9a2>
   1c306: 4e 84        	mv	s0, s3
   1c308: 41 47        	addi	a4, zero, 16
   1c30a: e6 56        	lw	a3, 120(sp)
   1c30c: 63 5e 97 38  	bge	a4, s1, 0x1c6a8 <_vfiprintf_r+0xca0>
   1c310: b7 2e 02 00  	lui	t4, 34
   1c314: 93 8e 0e 6e  	addi	t4, t4, 1760
   1c318: 22 86        	mv	a2, s0
   1c31a: c1 4b        	addi	s7, zero, 16
   1c31c: 26 84        	mv	s0, s1
   1c31e: 1d 4d        	addi	s10, zero, 7
   1c320: 46 ec        	sd	a7, 24(sp)
   1c322: 72 f0        	sd	t3, 32(sp)
   1c324: f6 84        	mv	s1, t4
   1c326: 01 a8        	j	0x1c336 <_vfiprintf_r+0x92e>
   1c328: 1b 87 26 00  	addiw	a4, a3, 2
   1c32c: 41 06        	addi	a2, a2, 16
   1c32e: ae 86        	mv	a3, a1
   1c330: 41 34        	addiw	s0, s0, -16
   1c332: 63 dc 8b 02  	bge	s7, s0, 0x1c36a <_vfiprintf_r+0x962>
   1c336: 9b 85 16 00  	addiw	a1, a3, 1
   1c33a: c1 07        	addi	a5, a5, 16
   1c33c: 04 e2        	sd	s1, 0(a2)
   1c33e: 23 34 76 01  	sd	s7, 8(a2)
   1c342: 3e e1        	sd	a5, 128(sp)
   1c344: ae dc        	sw	a1, 120(sp)
   1c346: e3 51 bd fe  	bge	s10, a1, 0x1c328 <_vfiprintf_r+0x920>
   1c34a: a1 cf        	beqz	a5, 0x1c3a2 <_vfiprintf_r+0x99a>
   1c34c: 90 18        	addi	a2, sp, 112
   1c34e: d2 85        	mv	a1, s4
   1c350: 56 85        	mv	a0, s5
   1c352: ef f0 4f e0  	jal	0x1b956 <__sprint_r.part.0>
   1c356: e3 1a 05 82  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c35a: e6 56        	lw	a3, 120(sp)
   1c35c: 41 34        	addiw	s0, s0, -16
   1c35e: 8a 67        	ld	a5, 128(sp)
   1c360: 4e 86        	mv	a2, s3
   1c362: 1b 87 16 00  	addiw	a4, a3, 1
   1c366: e3 c8 8b fc  	blt	s7, s0, 0x1c336 <_vfiprintf_r+0x92e>
   1c36a: e2 68        	ld	a7, 24(sp)
   1c36c: 02 7e        	ld	t3, 32(sp)
   1c36e: a6 8e        	mv	t4, s1
   1c370: a2 84        	mv	s1, s0
   1c372: 32 84        	mv	s0, a2
   1c374: a6 97        	add	a5, a5, s1
   1c376: 23 30 d4 01  	sd	t4, 0(s0)
   1c37a: 04 e4        	sd	s1, 8(s0)
   1c37c: 3e e1        	sd	a5, 128(sp)
   1c37e: ba dc        	sw	a4, 120(sp)
   1c380: 9d 46        	addi	a3, zero, 7
   1c382: e3 dc e6 94  	bge	a3, a4, 0x1bcda <_vfiprintf_r+0x2d2>
   1c386: 95 c3        	beqz	a5, 0x1c3aa <_vfiprintf_r+0x9a2>
   1c388: 90 18        	addi	a2, sp, 112
   1c38a: d2 85        	mv	a1, s4
   1c38c: 56 85        	mv	a0, s5
   1c38e: 72 f0        	sd	t3, 32(sp)
   1c390: 46 ec        	sd	a7, 24(sp)
   1c392: ef f0 4f dc  	jal	0x1b956 <__sprint_r.part.0>
   1c396: 63 1a 05 fe  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c39a: 8a 67        	ld	a5, 128(sp)
   1c39c: 02 7e        	ld	t3, 32(sp)
   1c39e: e2 68        	ld	a7, 24(sp)
   1c3a0: 2d ba        	j	0x1bcda <_vfiprintf_r+0x2d2>
   1c3a2: 05 47        	addi	a4, zero, 1
   1c3a4: 81 46        	mv	a3, zero
   1c3a6: 4e 86        	mv	a2, s3
   1c3a8: 61 b7        	j	0x1c330 <_vfiprintf_r+0x928>
   1c3aa: c6 87        	mv	a5, a7
   1c3ac: 63 d3 c8 01  	bge	a7, t3, 0x1c3b2 <_vfiprintf_r+0x9aa>
   1c3b0: f2 87        	mv	a5, t3
   1c3b2: bb 8c 97 01  	addw	s9, a5, s9
   1c3b6: 15 ba        	j	0x1bcea <_vfiprintf_r+0x2e2>
   1c3b8: e3 85 07 ca  	beqz	a5, 0x1c062 <_vfiprintf_r+0x65a>
   1c3bc: 90 18        	addi	a2, sp, 112
   1c3be: d2 85        	mv	a1, s4
   1c3c0: 56 85        	mv	a0, s5
   1c3c2: 1a f4        	sd	t1, 40(sp)
   1c3c4: 72 f0        	sd	t3, 32(sp)
   1c3c6: 46 ec        	sd	a7, 24(sp)
   1c3c8: ef f0 ef d8  	jal	0x1b956 <__sprint_r.part.0>
   1c3cc: 63 1f 05 fa  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c3d0: e6 55        	lw	a1, 120(sp)
   1c3d2: 8a 67        	ld	a5, 128(sp)
   1c3d4: 22 73        	ld	t1, 40(sp)
   1c3d6: 02 7e        	ld	t3, 32(sp)
   1c3d8: e2 68        	ld	a7, 24(sp)
   1c3da: 4e 84        	mv	s0, s3
   1c3dc: 85 25        	addiw	a1, a1, 1
   1c3de: 75 b8        	j	0x1bc9a <_vfiprintf_r+0x292>
   1c3e0: 82 dc        	sw	zero, 120(sp)
   1c3e2: 4e 84        	mv	s0, s3
   1c3e4: 6f f0 ef f0  	j	0x1baf2 <_vfiprintf_r+0xea>
   1c3e8: 85 45        	addi	a1, zero, 1
   1c3ea: 01 47        	mv	a4, zero
   1c3ec: 4e 84        	mv	s0, s3
   1c3ee: 71 b8        	j	0x1bc8a <_vfiprintf_r+0x282>
   1c3f0: 63 86 07 18  	beqz	a5, 0x1c57c <_vfiprintf_r+0xb74>
   1c3f4: 90 18        	addi	a2, sp, 112
   1c3f6: d2 85        	mv	a1, s4
   1c3f8: 56 85        	mv	a0, s5
   1c3fa: 9a e8        	sd	t1, 80(sp)
   1c3fc: f2 e4        	sd	t3, 72(sp)
   1c3fe: 46 f4        	sd	a7, 40(sp)
   1c400: 16 f0        	sd	t0, 32(sp)
   1c402: 7e ec        	sd	t6, 24(sp)
   1c404: ef f0 2f d5  	jal	0x1b956 <__sprint_r.part.0>
   1c408: 63 11 05 f8  	bnez	a0, 0x1bb8a <_vfiprintf_r+0x182>
   1c40c: 66 57        	lw	a4, 120(sp)
   1c40e: 8a 67        	ld	a5, 128(sp)
   1c410: 46 63        	ld	t1, 80(sp)
   1c412: 26 6e        	ld	t3, 72(sp)
   1c414: a2 78        	ld	a7, 40(sp)
   1c416: 82 72        	ld	t0, 32(sp)
   1c418: e2 6f        	ld	t6, 24(sp)
   1c41a: 4e 84        	mv	s0, s3
   1c41c: 9b 05 17 00  	addiw	a1, a4, 1
   1c420: 6f f0 7f 80  	j	0x1bc26 <_vfiprintf_r+0x21e>
   1c424: 01 4e        	mv	t3, zero
   1c426: 01 43        	mv	t1, zero
   1c428: e4 11        	addi	s1, sp, 236
   1c42a: 79 b2        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c42c: 13 03 c1 0e  	addi	t1, sp, 236
   1c430: 62 78        	ld	a6, 56(sp)
   1c432: 13 ff 0b 40  	andi	t5, s7, 1024
   1c436: 01 46        	mv	a2, zero
   1c438: 1a 87        	mv	a4, t1
   1c43a: a9 46        	addi	a3, zero, 10
   1c43c: 25 4e        	addi	t3, zero, 9
   1c43e: 93 0f f0 0f  	addi	t6, zero, 255
   1c442: 39 a0        	j	0x1c450 <_vfiprintf_r+0xa48>
   1c444: b3 57 dc 02  	divu	a5, s8, a3
   1c448: 63 77 8e 03  	bgeu	t3, s8, 0x1c476 <_vfiprintf_r+0xa6e>
   1c44c: 3e 8c        	mv	s8, a5
   1c44e: 26 87        	mv	a4, s1
   1c450: 93 04 f7 ff  	addi	s1, a4, -1
   1c454: 05 26        	addiw	a2, a2, 1
   1c456: b3 77 dc 02  	remu	a5, s8, a3
   1c45a: 9b 87 07 03  	addiw	a5, a5, 48
   1c45e: a3 0f f7 fe  	sb	a5, -1(a4)
   1c462: e3 01 0f fe  	beqz	t5, 0x1c444 <_vfiprintf_r+0xa3c>
   1c466: 83 47 08 00  	lbu	a5, 0(a6)
   1c46a: e3 9d c7 fc  	bne	a5, a2, 0x1c444 <_vfiprintf_r+0xa3c>
   1c46e: e3 8b f7 fd  	beq	a5, t6, 0x1c444 <_vfiprintf_r+0xa3c>
   1c472: 63 61 8e 13  	bltu	t3, s8, 0x1c594 <_vfiprintf_r+0xb8c>
   1c476: 3b 03 93 40  	subw	t1, t1, s1
   1c47a: 42 fc        	sd	a6, 56(sp)
   1c47c: 1b 0e 03 00  	sext.w	t3, t1
   1c480: 63 54 a3 01  	bge	t1, s10, 0x1c488 <_vfiprintf_r+0xa80>
   1c484: 1b 0e 0d 00  	sext.w	t3, s10
   1c488: 76 8c        	mv	s8, t4
   1c48a: 3d b2        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c48c: 01 4e        	mv	t3, zero
   1c48e: e4 11        	addi	s1, sp, 236
   1c490: 25 b2        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c492: 41 04        	addi	s0, s0, 16
   1c494: 9b 85 16 00  	addiw	a1, a3, 1
   1c498: 36 87        	mv	a4, a3
   1c49a: 6f f0 8f ff  	j	0x1bc92 <_vfiprintf_r+0x28a>
   1c49e: 93 77 03 04  	andi	a5, t1, 64
   1c4a2: 03 2c 0c 00  	lw	s8, 0(s8)
   1c4a6: 63 8f 07 16  	beqz	a5, 0x1c624 <_vfiprintf_r+0xc1c>
   1c4aa: 42 1c        	slli	s8, s8, 48
   1c4ac: 13 5c 0c 03  	srli	s8, s8, 48
   1c4b0: f9 b4        	j	0x1bf7e <_vfiprintf_r+0x576>
   1c4b2: 93 f6 07 04  	andi	a3, a5, 64
   1c4b6: 03 27 0c 00  	lw	a4, 0(s8)
   1c4ba: 63 8b 06 16  	beqz	a3, 0x1c630 <_vfiprintf_r+0xc28>
   1c4be: 42 17        	slli	a4, a4, 48
   1c4c0: 41 93        	srli	a4, a4, 48
   1c4c2: 25 bc        	j	0x1befa <_vfiprintf_r+0x4f2>
   1c4c4: 83 c7 0d 00  	lbu	a5, 0(s11)
   1c4c8: 63 06 07 e4  	beqz	a4, 0x1bb14 <_vfiprintf_r+0x10c>
   1c4cc: 03 47 07 00  	lbu	a4, 0(a4)
   1c4d0: 63 02 07 e4  	beqz	a4, 0x1bb14 <_vfiprintf_r+0x10c>
   1c4d4: 93 eb 0b 40  	ori	s7, s7, 1024
   1c4d8: 81 2b        	sext.w	s7, s7
   1c4da: 6f f0 af e3  	j	0x1bb14 <_vfiprintf_r+0x10c>
   1c4de: 01 47        	mv	a4, zero
   1c4e0: 85 45        	addi	a1, zero, 1
   1c4e2: 4e 84        	mv	s0, s3
   1c4e4: 6f f0 6f fa  	j	0x1bc8a <_vfiprintf_r+0x282>
   1c4e8: 1b 0e 0d 00  	sext.w	t3, s10
   1c4ec: b6 84        	mv	s1, a3
   1c4ee: e9 b0        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c4f0: de 87        	mv	a5, s7
   1c4f2: cd ba        	j	0x1bee4 <_vfiprintf_r+0x4dc>
   1c4f4: 37 27 02 00  	lui	a4, 34
   1c4f8: 13 07 07 bf  	addi	a4, a4, -1040
   1c4fc: 3a e8        	sd	a4, 16(sp)
   1c4fe: 13 f7 0b 02  	andi	a4, s7, 32
   1c502: 93 06 8c 00  	addi	a3, s8, 8
   1c506: 31 eb        	bnez	a4, 0x1c55a <_vfiprintf_r+0xb52>
   1c508: 13 f7 0b 01  	andi	a4, s7, 16
   1c50c: 39 e7        	bnez	a4, 0x1c55a <_vfiprintf_r+0xb52>
   1c50e: 13 f6 0b 04  	andi	a2, s7, 64
   1c512: 03 27 0c 00  	lw	a4, 0(s8)
   1c516: 63 01 06 10  	beqz	a2, 0x1c618 <_vfiprintf_r+0xc10>
   1c51a: 42 17        	slli	a4, a4, 48
   1c51c: 41 93        	srli	a4, a4, 48
   1c51e: 13 f6 1b 00  	andi	a2, s7, 1
   1c522: 19 ca        	beqz	a2, 0x1c538 <_vfiprintf_r+0xb30>
   1c524: 11 cb        	beqz	a4, 0x1c538 <_vfiprintf_r+0xb30>
   1c526: 93 eb 2b 00  	ori	s7, s7, 2
   1c52a: 13 06 00 03  	addi	a2, zero, 48
   1c52e: 23 04 c1 06  	sb	a2, 104(sp)
   1c532: a3 04 f1 06  	sb	a5, 105(sp)
   1c536: 81 2b        	sext.w	s7, s7
   1c538: 93 fb fb bf  	andi	s7, s7, -1025
   1c53c: 81 2b        	sext.w	s7, s7
   1c53e: 36 8c        	mv	s8, a3
   1c540: d9 bc        	j	0x1c016 <_vfiprintf_r+0x60e>
   1c542: 5e 83        	mv	t1, s7
   1c544: 15 b4        	j	0x1bf68 <_vfiprintf_r+0x560>
   1c546: 37 27 02 00  	lui	a4, 34
   1c54a: 13 07 87 c0  	addi	a4, a4, -1016
   1c54e: 3a e8        	sd	a4, 16(sp)
   1c550: 13 f7 0b 02  	andi	a4, s7, 32
   1c554: 93 06 8c 00  	addi	a3, s8, 8
   1c558: 45 db        	beqz	a4, 0x1c508 <_vfiprintf_r+0xb00>
   1c55a: 03 37 0c 00  	ld	a4, 0(s8)
   1c55e: c1 b7        	j	0x1c51e <_vfiprintf_r+0xb16>
   1c560: 26 85        	mv	a0, s1
   1c562: ef d0 cf f5  	jal	0x19cbe <strlen>
   1c566: 1b 03 05 00  	sext.w	t1, a0
   1c56a: 13 4e f3 ff  	not	t3, t1
   1c56e: 13 5e fe 43  	srai	t3, t3, 63
   1c572: e2 68        	ld	a7, 24(sp)
   1c574: 33 7e 6e 00  	and	t3, t3, t1
   1c578: 01 4d        	mv	s10, zero
   1c57a: 3d b8        	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c57c: 03 47 71 06  	lbu	a4, 103(sp)
   1c580: 79 cf        	beqz	a4, 0x1c65e <_vfiprintf_r+0xc56>
   1c582: 93 07 71 06  	addi	a5, sp, 103
   1c586: be f9        	sd	a5, 240(sp)
   1c588: 85 47        	addi	a5, zero, 1
   1c58a: be fd        	sd	a5, 248(sp)
   1c58c: 85 45        	addi	a1, zero, 1
   1c58e: 4e 84        	mv	s0, s3
   1c590: 6f f0 2f eb  	j	0x1bc42 <_vfiprintf_r+0x23a>
   1c594: c2 77        	ld	a5, 48(sp)
   1c596: 86 65        	ld	a1, 64(sp)
   1c598: 9a e4        	sd	t1, 72(sp)
   1c59a: 9d 8c        	sub	s1, s1, a5
   1c59c: 3e 86        	mv	a2, a5
   1c59e: 26 85        	mv	a0, s1
   1c5a0: 76 fc        	sd	t4, 56(sp)
   1c5a2: 7a f4        	sd	t5, 40(sp)
   1c5a4: 46 f0        	sd	a7, 32(sp)
   1c5a6: 42 ec        	sd	a6, 24(sp)
   1c5a8: ef d0 6f fa  	jal	0x19d4e <strncpy>
   1c5ac: 62 68        	ld	a6, 24(sp)
   1c5ae: a9 47        	addi	a5, zero, 10
   1c5b0: b3 57 fc 02  	divu	a5, s8, a5
   1c5b4: 03 47 18 00  	lbu	a4, 1(a6)
   1c5b8: 82 78        	ld	a7, 32(sp)
   1c5ba: 22 7f        	ld	t5, 40(sp)
   1c5bc: 33 37 e0 00  	snez	a4, a4
   1c5c0: e2 7e        	ld	t4, 56(sp)
   1c5c2: 26 63        	ld	t1, 72(sp)
   1c5c4: 3a 98        	add	a6, a6, a4
   1c5c6: 01 46        	mv	a2, zero
   1c5c8: a9 46        	addi	a3, zero, 10
   1c5ca: 25 4e        	addi	t3, zero, 9
   1c5cc: 93 0f f0 0f  	addi	t6, zero, 255
   1c5d0: b5 bd        	j	0x1c44c <_vfiprintf_r+0xa44>
   1c5d2: be 8b        	mv	s7, a5
   1c5d4: 81 47        	mv	a5, zero
   1c5d6: e3 8a 07 9c  	beqz	a5, 0x1bfaa <_vfiprintf_r+0x5a2>
   1c5da: 99 b1        	j	0x1c220 <_vfiprintf_r+0x818>
   1c5dc: 93 eb 0b 20  	ori	s7, s7, 512
   1c5e0: 83 c7 1d 00  	lbu	a5, 1(s11)
   1c5e4: 81 2b        	sext.w	s7, s7
   1c5e6: 85 0d        	addi	s11, s11, 1
   1c5e8: 6f f0 cf d2  	j	0x1bb14 <_vfiprintf_r+0x10c>
   1c5ec: 93 eb 0b 02  	ori	s7, s7, 32
   1c5f0: 83 c7 1d 00  	lbu	a5, 1(s11)
   1c5f4: 81 2b        	sext.w	s7, s7
   1c5f6: 85 0d        	addi	s11, s11, 1
   1c5f8: 6f f0 cf d1  	j	0x1bb14 <_vfiprintf_r+0x10c>
   1c5fc: 99 47        	addi	a5, zero, 6
   1c5fe: 1b 03 0d 00  	sext.w	t1, s10
   1c602: 63 ec a7 05  	bltu	a5, s10, 0x1c65a <_vfiprintf_r+0xc52>
   1c606: b7 24 02 00  	lui	s1, 34
   1c60a: 1a 8e        	mv	t3, t1
   1c60c: 93 84 04 c2  	addi	s1, s1, -992
   1c610: 6f f0 0f df  	j	0x1bc00 <_vfiprintf_r+0x1f8>
   1c614: 01 47        	mv	a4, zero
   1c616: 29 b1        	j	0x1c220 <_vfiprintf_r+0x818>
   1c618: 13 f6 0b 20  	andi	a2, s7, 512
   1c61c: 25 c6        	beqz	a2, 0x1c684 <_vfiprintf_r+0xc7c>
   1c61e: 13 77 f7 0f  	andi	a4, a4, 255
   1c622: f5 bd        	j	0x1c51e <_vfiprintf_r+0xb16>
   1c624: 93 77 03 20  	andi	a5, t1, 512
   1c628: b1 cb        	beqz	a5, 0x1c67c <_vfiprintf_r+0xc74>
   1c62a: 13 7c fc 0f  	andi	s8, s8, 255
   1c62e: 81 ba        	j	0x1bf7e <_vfiprintf_r+0x576>
   1c630: 93 f6 07 20  	andi	a3, a5, 512
   1c634: a9 c2        	beqz	a3, 0x1c676 <_vfiprintf_r+0xc6e>
   1c636: 13 77 f7 0f  	andi	a4, a4, 255
   1c63a: c1 b0        	j	0x1befa <_vfiprintf_r+0x4f2>
   1c63c: 93 f7 0b 20  	andi	a5, s7, 512
   1c640: 85 cb        	beqz	a5, 0x1c670 <_vfiprintf_r+0xc68>
   1c642: 1b 1c 8c 01  	slliw	s8, s8, 24
   1c646: 1b 5c 8c 41  	sraiw	s8, s8, 24
   1c64a: e2 87        	mv	a5, s8
   1c64c: 6f f0 0f ee  	j	0x1bd2c <_vfiprintf_r+0x324>
   1c650: 85 45        	addi	a1, zero, 1
   1c652: 01 47        	mv	a4, zero
   1c654: 4e 84        	mv	s0, s3
   1c656: 6f f0 cf e3  	j	0x1bc92 <_vfiprintf_r+0x28a>
   1c65a: 19 43        	addi	t1, zero, 6
   1c65c: 6d b7        	j	0x1c606 <_vfiprintf_r+0xbfe>
   1c65e: 85 45        	addi	a1, zero, 1
   1c660: 4e 84        	mv	s0, s3
   1c662: 6f f0 6f de  	j	0x1bc48 <_vfiprintf_r+0x240>
   1c666: 23 10 97 01  	sh	s9, 0(a4)
   1c66a: 21 0c        	addi	s8, s8, 8
   1c66c: 6f f0 4f c3  	j	0x1baa0 <_vfiprintf_r+0x98>
   1c670: e2 87        	mv	a5, s8
   1c672: 6f f0 af eb  	j	0x1bd2c <_vfiprintf_r+0x324>
   1c676: 02 17        	slli	a4, a4, 32
   1c678: 01 93        	srli	a4, a4, 32
   1c67a: 41 b0        	j	0x1befa <_vfiprintf_r+0x4f2>
   1c67c: 02 1c        	slli	s8, s8, 32
   1c67e: 13 5c 0c 02  	srli	s8, s8, 32
   1c682: f5 b8        	j	0x1bf7e <_vfiprintf_r+0x576>
   1c684: 02 17        	slli	a4, a4, 32
   1c686: 01 93        	srli	a4, a4, 32
   1c688: 59 bd        	j	0x1c51e <_vfiprintf_r+0xb16>
   1c68a: 6a 8e        	mv	t3, s10
   1c68c: 6a 83        	mv	t1, s10
   1c68e: 01 4d        	mv	s10, zero
   1c690: 6f f0 8f f2  	j	0x1bdb8 <_vfiprintf_r+0x3b0>
   1c694: b7 2e 02 00  	lui	t4, 34
   1c698: 93 8e 0e 6e  	addi	t4, t4, 1760
   1c69c: 15 b9        	j	0x1c2d0 <_vfiprintf_r+0x8c8>
   1c69e: 76 8c        	mv	s8, t4
   1c6a0: 93 07 00 03  	addi	a5, zero, 48
   1c6a4: 6f f0 0f f0  	j	0x1bda4 <_vfiprintf_r+0x39c>
   1c6a8: b7 2e 02 00  	lui	t4, 34
   1c6ac: 1b 87 16 00  	addiw	a4, a3, 1
   1c6b0: 93 8e 0e 6e  	addi	t4, t4, 1760
   1c6b4: c1 b1        	j	0x1c374 <_vfiprintf_r+0x96c>
   1c6b6: 37 2f 02 00  	lui	t5, 34
   1c6ba: ae 86        	mv	a3, a1
   1c6bc: 13 0f 0f 6d  	addi	t5, t5, 1744
   1c6c0: 15 bc        	j	0x1c0f4 <_vfiprintf_r+0x6ec>
   1c6c2: 7d 5c        	addi	s8, zero, -1
   1c6c4: 6f f0 4f cd  	j	0x1bb98 <_vfiprintf_r+0x190>
   1c6c8: 23 20 97 01  	sw	s9, 0(a4)
   1c6cc: 21 0c        	addi	s8, s8, 8
   1c6ce: 6f f0 2f bd  	j	0x1baa0 <_vfiprintf_r+0x98>
   1c6d2: 83 27 0c 00  	lw	a5, 0(s8)
   1c6d6: 21 0c        	addi	s8, s8, 8
   1c6d8: 1b 8d 07 00  	sext.w	s10, a5
   1c6dc: 63 d3 07 00  	bgez	a5, 0x1c6e2 <_vfiprintf_r+0xcda>
   1c6e0: 7d 5d        	addi	s10, zero, -1
   1c6e2: 83 c7 1d 00  	lbu	a5, 1(s11)
   1c6e6: ba 8d        	mv	s11, a4
   1c6e8: 6f f0 cf c2  	j	0x1bb14 <_vfiprintf_r+0x10c>

000000000001c6ec <vfiprintf>:
   1c6ec: aa 87        	mv	a5, a0
   1c6ee: 03 b5 81 93  	ld	a0, -1736(gp)
   1c6f2: b2 86        	mv	a3, a2
   1c6f4: 2e 86        	mv	a2, a1
   1c6f6: be 85        	mv	a1, a5
   1c6f8: 6f f0 0f b1  	j	0x1ba08 <_vfiprintf_r>

000000000001c6fc <__sbprintf>:
   1c6fc: 83 d7 05 01  	lhu	a5, 16(a1)
   1c700: 03 ae c5 0a  	lw	t3, 172(a1)
   1c704: 03 d3 25 01  	lhu	t1, 18(a1)
   1c708: 83 b8 05 03  	ld	a7, 48(a1)
   1c70c: 03 b8 05 04  	ld	a6, 64(a1)
   1c710: 13 01 01 b3  	addi	sp, sp, -1232
   1c714: f5 9b        	andi	a5, a5, -3
   1c716: 13 07 00 40  	addi	a4, zero, 1024
   1c71a: 23 30 81 4c  	sd	s0, 1216(sp)
   1c71e: 23 18 f1 00  	sh	a5, 16(sp)
   1c722: 2e 84        	mv	s0, a1
   1c724: 1c 19        	addi	a5, sp, 176
   1c726: 8a 85        	mv	a1, sp
   1c728: 23 3c 91 4a  	sd	s1, 1208(sp)
   1c72c: 23 38 21 4b  	sd	s2, 1200(sp)
   1c730: 23 34 11 4c  	sd	ra, 1224(sp)
   1c734: 2a 89        	mv	s2, a0
   1c736: 72 d7        	sw	t3, 172(sp)
   1c738: 23 19 61 00  	sh	t1, 18(sp)
   1c73c: 46 f8        	sd	a7, 48(sp)
   1c73e: c2 e0        	sd	a6, 64(sp)
   1c740: 3e e0        	sd	a5, 0(sp)
   1c742: 3e ec        	sd	a5, 24(sp)
   1c744: 3a c6        	sw	a4, 12(sp)
   1c746: 3a d0        	sw	a4, 32(sp)
   1c748: 02 d4        	sw	zero, 40(sp)
   1c74a: ef f0 ef ab  	jal	0x1ba08 <_vfiprintf_r>
   1c74e: aa 84        	mv	s1, a0
   1c750: 63 59 05 02  	bgez	a0, 0x1c782 <__sbprintf+0x86>
   1c754: 83 57 01 01  	lhu	a5, 16(sp)
   1c758: 93 f7 07 04  	andi	a5, a5, 64
   1c75c: 99 c7        	beqz	a5, 0x1c76a <__sbprintf+0x6e>
   1c75e: 83 57 04 01  	lhu	a5, 16(s0)
   1c762: 93 e7 07 04  	ori	a5, a5, 64
   1c766: 23 18 f4 00  	sh	a5, 16(s0)
   1c76a: 83 30 81 4c  	ld	ra, 1224(sp)
   1c76e: 03 34 01 4c  	ld	s0, 1216(sp)
   1c772: 03 39 01 4b  	ld	s2, 1200(sp)
   1c776: 26 85        	mv	a0, s1
   1c778: 83 34 81 4b  	ld	s1, 1208(sp)
   1c77c: 13 01 01 4d  	addi	sp, sp, 1232
   1c780: 82 80        	ret
   1c782: 8a 85        	mv	a1, sp
   1c784: 4a 85        	mv	a0, s2
   1c786: ef 80 bf cd  	jal	0x15460 <_fflush_r>
   1c78a: 69 d5        	beqz	a0, 0x1c754 <__sbprintf+0x58>
   1c78c: fd 54        	addi	s1, zero, -1
   1c78e: d9 b7        	j	0x1c754 <__sbprintf+0x58>

000000000001c790 <_wctomb_r>:
   1c790: b7 47 02 00  	lui	a5, 36
   1c794: 83 b7 87 84  	ld	a5, -1976(a5)
   1c798: 82 87        	jr	a5

000000000001c79a <__ascii_wctomb>:
   1c79a: 9b 07 06 00  	sext.w	a5, a2
   1c79e: 91 cd        	beqz	a1, 0x1c7ba <__ascii_wctomb+0x20>
   1c7a0: 13 07 f0 0f  	addi	a4, zero, 255
   1c7a4: 63 66 f7 00  	bltu	a4, a5, 0x1c7b0 <__ascii_wctomb+0x16>
   1c7a8: 23 80 c5 00  	sb	a2, 0(a1)
   1c7ac: 05 45        	addi	a0, zero, 1
   1c7ae: 82 80        	ret
   1c7b0: 93 07 a0 08  	addi	a5, zero, 138
   1c7b4: 1c c1        	sw	a5, 0(a0)
   1c7b6: 7d 55        	addi	a0, zero, -1
   1c7b8: 82 80        	ret
   1c7ba: 01 45        	mv	a0, zero
   1c7bc: 82 80        	ret

000000000001c7be <_write_r>:
   1c7be: 01 11        	addi	sp, sp, -32
   1c7c0: 2e 87        	mv	a4, a1
   1c7c2: 22 e8        	sd	s0, 16(sp)
   1c7c4: 26 e4        	sd	s1, 8(sp)
   1c7c6: b2 85        	mv	a1, a2
   1c7c8: 2a 84        	mv	s0, a0
   1c7ca: 36 86        	mv	a2, a3
   1c7cc: 3a 85        	mv	a0, a4
   1c7ce: 06 ec        	sd	ra, 24(sp)
   1c7d0: 23 ac 01 96  	sw	zero, -1672(gp)
   1c7d4: ef 20 60 2c  	jal	0x1ea9a <_write>
   1c7d8: fd 57        	addi	a5, zero, -1
   1c7da: 63 07 f5 00  	beq	a0, a5, 0x1c7e8 <_write_r+0x2a>
   1c7de: e2 60        	ld	ra, 24(sp)
   1c7e0: 42 64        	ld	s0, 16(sp)
   1c7e2: a2 64        	ld	s1, 8(sp)
   1c7e4: 05 61        	addi	sp, sp, 32
   1c7e6: 82 80        	ret
   1c7e8: 83 a7 81 97  	lw	a5, -1672(gp)
   1c7ec: ed db        	beqz	a5, 0x1c7de <_write_r+0x20>
   1c7ee: e2 60        	ld	ra, 24(sp)
   1c7f0: 1c c0        	sw	a5, 0(s0)
   1c7f2: 42 64        	ld	s0, 16(sp)
   1c7f4: a2 64        	ld	s1, 8(sp)
   1c7f6: 05 61        	addi	sp, sp, 32
   1c7f8: 82 80        	ret

000000000001c7fa <__assert_func>:
   1c7fa: 03 b7 81 93  	ld	a4, -1736(gp)
   1c7fe: 41 11        	addi	sp, sp, -16
   1c800: b6 87        	mv	a5, a3
   1c802: 32 88        	mv	a6, a2
   1c804: 06 e4        	sd	ra, 8(sp)
   1c806: 83 38 87 01  	ld	a7, 24(a4)
   1c80a: aa 86        	mv	a3, a0
   1c80c: 2e 87        	mv	a4, a1
   1c80e: 3e 86        	mv	a2, a5
   1c810: 63 0f 08 00  	beqz	a6, 0x1c82e <__assert_func+0x34>
   1c814: b7 27 02 00  	lui	a5, 34
   1c818: 93 87 07 6f  	addi	a5, a5, 1776
   1c81c: b7 25 02 00  	lui	a1, 34
   1c820: 93 85 05 70  	addi	a1, a1, 1792
   1c824: 46 85        	mv	a0, a7
   1c826: ef 00 e0 1e  	jal	0x1ca14 <fiprintf>
   1c82a: ef 10 50 4b  	jal	0x1e4de <abort>
   1c82e: 37 28 02 00  	lui	a6, 34
   1c832: 93 07 08 c0  	addi	a5, a6, -1024
   1c836: 13 08 08 c0  	addi	a6, a6, -1024
   1c83a: cd b7        	j	0x1c81c <__assert_func+0x22>

000000000001c83c <__assert>:
   1c83c: 41 11        	addi	sp, sp, -16
   1c83e: b2 86        	mv	a3, a2
   1c840: 01 46        	mv	a2, zero
   1c842: 06 e4        	sd	ra, 8(sp)
   1c844: ef f0 7f fb  	jal	0x1c7fa <__assert_func>

000000000001c848 <atexit>:
   1c848: aa 85        	mv	a1, a0
   1c84a: 81 46        	mv	a3, zero
   1c84c: 01 46        	mv	a2, zero
   1c84e: 01 45        	mv	a0, zero
   1c850: 6f 10 f0 49  	j	0x1e4ee <__register_exitproc>

000000000001c854 <_calloc_r>:
   1c854: b3 b7 c5 02  	mulhu	a5, a1, a2
   1c858: 41 11        	addi	sp, sp, -16
   1c85a: 06 e4        	sd	ra, 8(sp)
   1c85c: 22 e0        	sd	s0, 0(sp)
   1c85e: b3 85 c5 02  	mul	a1, a1, a2
   1c862: b5 ef        	bnez	a5, 0x1c8de <_calloc_r+0x8a>
   1c864: ef b0 df cc  	jal	0x18530 <_malloc_r>
   1c868: 2a 84        	mv	s0, a0
   1c86a: 35 cd        	beqz	a0, 0x1c8e6 <_calloc_r+0x92>
   1c86c: 03 36 85 ff  	ld	a2, -8(a0)
   1c870: 13 07 80 04  	addi	a4, zero, 72
   1c874: 71 9a        	andi	a2, a2, -4
   1c876: 61 16        	addi	a2, a2, -8
   1c878: 63 62 c7 04  	bltu	a4, a2, 0x1c8bc <_calloc_r+0x68>
   1c87c: 93 06 70 02  	addi	a3, zero, 39
   1c880: aa 87        	mv	a5, a0
   1c882: 63 f2 c6 02  	bgeu	a3, a2, 0x1c8a6 <_calloc_r+0x52>
   1c886: 23 30 05 00  	sd	zero, 0(a0)
   1c88a: 23 34 05 00  	sd	zero, 8(a0)
   1c88e: 93 07 70 03  	addi	a5, zero, 55
   1c892: 63 fd c7 02  	bgeu	a5, a2, 0x1c8cc <_calloc_r+0x78>
   1c896: 23 38 05 00  	sd	zero, 16(a0)
   1c89a: 23 3c 05 00  	sd	zero, 24(a0)
   1c89e: 93 07 05 02  	addi	a5, a0, 32
   1c8a2: 63 08 e6 04  	beq	a2, a4, 0x1c8f2 <_calloc_r+0x9e>
   1c8a6: 23 b0 07 00  	sd	zero, 0(a5)
   1c8aa: 23 b4 07 00  	sd	zero, 8(a5)
   1c8ae: 23 b8 07 00  	sd	zero, 16(a5)
   1c8b2: a2 60        	ld	ra, 8(sp)
   1c8b4: 22 85        	mv	a0, s0
   1c8b6: 02 64        	ld	s0, 0(sp)
   1c8b8: 41 01        	addi	sp, sp, 16
   1c8ba: 82 80        	ret
   1c8bc: 81 45        	mv	a1, zero
   1c8be: ef 60 9f b3  	jal	0x133f6 <memset>
   1c8c2: a2 60        	ld	ra, 8(sp)
   1c8c4: 22 85        	mv	a0, s0
   1c8c6: 02 64        	ld	s0, 0(sp)
   1c8c8: 41 01        	addi	sp, sp, 16
   1c8ca: 82 80        	ret
   1c8cc: 93 07 05 01  	addi	a5, a0, 16
   1c8d0: 23 b0 07 00  	sd	zero, 0(a5)
   1c8d4: 23 b4 07 00  	sd	zero, 8(a5)
   1c8d8: 23 b8 07 00  	sd	zero, 16(a5)
   1c8dc: d9 bf        	j	0x1c8b2 <_calloc_r+0x5e>
   1c8de: ef 00 80 05  	jal	0x1c936 <__errno>
   1c8e2: b1 47        	addi	a5, zero, 12
   1c8e4: 1c c1        	sw	a5, 0(a0)
   1c8e6: 01 44        	mv	s0, zero
   1c8e8: a2 60        	ld	ra, 8(sp)
   1c8ea: 22 85        	mv	a0, s0
   1c8ec: 02 64        	ld	s0, 0(sp)
   1c8ee: 41 01        	addi	sp, sp, 16
   1c8f0: 82 80        	ret
   1c8f2: 23 30 05 02  	sd	zero, 32(a0)
   1c8f6: 93 07 05 03  	addi	a5, a0, 48
   1c8fa: 23 34 05 02  	sd	zero, 40(a0)
   1c8fe: 65 b7        	j	0x1c8a6 <_calloc_r+0x52>

000000000001c900 <_close_r>:
   1c900: 01 11        	addi	sp, sp, -32
   1c902: 22 e8        	sd	s0, 16(sp)
   1c904: 26 e4        	sd	s1, 8(sp)
   1c906: 2a 84        	mv	s0, a0
   1c908: 2e 85        	mv	a0, a1
   1c90a: 06 ec        	sd	ra, 24(sp)
   1c90c: 23 ac 01 96  	sw	zero, -1672(gp)
   1c910: ef 20 20 01  	jal	0x1e922 <_close>
   1c914: fd 57        	addi	a5, zero, -1
   1c916: 63 07 f5 00  	beq	a0, a5, 0x1c924 <_close_r+0x24>
   1c91a: e2 60        	ld	ra, 24(sp)
   1c91c: 42 64        	ld	s0, 16(sp)
   1c91e: a2 64        	ld	s1, 8(sp)
   1c920: 05 61        	addi	sp, sp, 32
   1c922: 82 80        	ret
   1c924: 83 a7 81 97  	lw	a5, -1672(gp)
   1c928: ed db        	beqz	a5, 0x1c91a <_close_r+0x1a>
   1c92a: e2 60        	ld	ra, 24(sp)
   1c92c: 1c c0        	sw	a5, 0(s0)
   1c92e: 42 64        	ld	s0, 16(sp)
   1c930: a2 64        	ld	s1, 8(sp)
   1c932: 05 61        	addi	sp, sp, 32
   1c934: 82 80        	ret

000000000001c936 <__errno>:
   1c936: 03 b5 81 93  	ld	a0, -1736(gp)
   1c93a: 82 80        	ret

000000000001c93c <_fclose_r>:
   1c93c: 01 11        	addi	sp, sp, -32
   1c93e: 06 ec        	sd	ra, 24(sp)
   1c940: 22 e8        	sd	s0, 16(sp)
   1c942: 26 e4        	sd	s1, 8(sp)
   1c944: 4a e0        	sd	s2, 0(sp)
   1c946: 89 c9        	beqz	a1, 0x1c958 <_fclose_r+0x1c>
   1c948: 2e 84        	mv	s0, a1
   1c94a: aa 84        	mv	s1, a0
   1c94c: 19 c1        	beqz	a0, 0x1c952 <_fclose_r+0x16>
   1c94e: 3c 49        	lw	a5, 80(a0)
   1c950: a5 cf        	beqz	a5, 0x1c9c8 <_fclose_r+0x8c>
   1c952: 83 17 04 01  	lh	a5, 16(s0)
   1c956: 89 eb        	bnez	a5, 0x1c968 <_fclose_r+0x2c>
   1c958: e2 60        	ld	ra, 24(sp)
   1c95a: 42 64        	ld	s0, 16(sp)
   1c95c: 01 49        	mv	s2, zero
   1c95e: a2 64        	ld	s1, 8(sp)
   1c960: 4a 85        	mv	a0, s2
   1c962: 02 69        	ld	s2, 0(sp)
   1c964: 05 61        	addi	sp, sp, 32
   1c966: 82 80        	ret
   1c968: a2 85        	mv	a1, s0
   1c96a: 26 85        	mv	a0, s1
   1c96c: ef 80 3f 97  	jal	0x152de <__sflush_r>
   1c970: 3c 68        	ld	a5, 80(s0)
   1c972: 2a 89        	mv	s2, a0
   1c974: 91 c7        	beqz	a5, 0x1c980 <_fclose_r+0x44>
   1c976: 0c 78        	ld	a1, 48(s0)
   1c978: 26 85        	mv	a0, s1
   1c97a: 82 97        	jalr	a5
   1c97c: 63 4c 05 04  	bltz	a0, 0x1c9d4 <_fclose_r+0x98>
   1c980: 83 57 04 01  	lhu	a5, 16(s0)
   1c984: 93 f7 07 08  	andi	a5, a5, 128
   1c988: a1 ef        	bnez	a5, 0x1c9e0 <_fclose_r+0xa4>
   1c98a: 2c 6c        	ld	a1, 88(s0)
   1c98c: 91 c9        	beqz	a1, 0x1c9a0 <_fclose_r+0x64>
   1c98e: 93 07 44 07  	addi	a5, s0, 116
   1c992: 63 85 f5 00  	beq	a1, a5, 0x1c99c <_fclose_r+0x60>
   1c996: 26 85        	mv	a0, s1
   1c998: ef 80 ff ec  	jal	0x15866 <_free_r>
   1c99c: 23 3c 04 04  	sd	zero, 88(s0)
   1c9a0: 2c 7c        	ld	a1, 120(s0)
   1c9a2: 91 c5        	beqz	a1, 0x1c9ae <_fclose_r+0x72>
   1c9a4: 26 85        	mv	a0, s1
   1c9a6: ef 80 1f ec  	jal	0x15866 <_free_r>
   1c9aa: 23 3c 04 06  	sd	zero, 120(s0)
   1c9ae: ef 80 df d8  	jal	0x1573a <__sfp_lock_acquire>
   1c9b2: 23 18 04 00  	sh	zero, 16(s0)
   1c9b6: ef 80 7f d8  	jal	0x1573c <__sfp_lock_release>
   1c9ba: e2 60        	ld	ra, 24(sp)
   1c9bc: 42 64        	ld	s0, 16(sp)
   1c9be: a2 64        	ld	s1, 8(sp)
   1c9c0: 4a 85        	mv	a0, s2
   1c9c2: 02 69        	ld	s2, 0(sp)
   1c9c4: 05 61        	addi	sp, sp, 32
   1c9c6: 82 80        	ret
   1c9c8: ef 80 bf d6  	jal	0x15732 <__sinit>
   1c9cc: 83 17 04 01  	lh	a5, 16(s0)
   1c9d0: c1 d7        	beqz	a5, 0x1c958 <_fclose_r+0x1c>
   1c9d2: 59 bf        	j	0x1c968 <_fclose_r+0x2c>
   1c9d4: 83 57 04 01  	lhu	a5, 16(s0)
   1c9d8: 7d 59        	addi	s2, zero, -1
   1c9da: 93 f7 07 08  	andi	a5, a5, 128
   1c9de: d5 d7        	beqz	a5, 0x1c98a <_fclose_r+0x4e>
   1c9e0: 0c 6c        	ld	a1, 24(s0)
   1c9e2: 26 85        	mv	a0, s1
   1c9e4: ef 80 3f e8  	jal	0x15866 <_free_r>
   1c9e8: 4d b7        	j	0x1c98a <_fclose_r+0x4e>

000000000001c9ea <fclose>:
   1c9ea: aa 85        	mv	a1, a0
   1c9ec: 03 b5 81 93  	ld	a0, -1736(gp)
   1c9f0: 6f f0 df f4  	j	0x1c93c <_fclose_r>

000000000001c9f4 <_fiprintf_r>:
   1c9f4: 5d 71        	addi	sp, sp, -80
   1c9f6: 13 03 81 02  	addi	t1, sp, 40
   1c9fa: 36 f4        	sd	a3, 40(sp)
   1c9fc: 9a 86        	mv	a3, t1
   1c9fe: 06 ec        	sd	ra, 24(sp)
   1ca00: 3a f8        	sd	a4, 48(sp)
   1ca02: 3e fc        	sd	a5, 56(sp)
   1ca04: c2 e0        	sd	a6, 64(sp)
   1ca06: c6 e4        	sd	a7, 72(sp)
   1ca08: 1a e4        	sd	t1, 8(sp)
   1ca0a: ef e0 ff ff  	jal	0x1ba08 <_vfiprintf_r>
   1ca0e: e2 60        	ld	ra, 24(sp)
   1ca10: 61 61        	addi	sp, sp, 80
   1ca12: 82 80        	ret

000000000001ca14 <fiprintf>:
   1ca14: 2a 8e        	mv	t3, a0
   1ca16: 5d 71        	addi	sp, sp, -80
   1ca18: 03 b5 81 93  	ld	a0, -1736(gp)
   1ca1c: 13 03 01 02  	addi	t1, sp, 32
   1ca20: 32 f0        	sd	a2, 32(sp)
   1ca22: 36 f4        	sd	a3, 40(sp)
   1ca24: 2e 86        	mv	a2, a1
   1ca26: 9a 86        	mv	a3, t1
   1ca28: f2 85        	mv	a1, t3
   1ca2a: 06 ec        	sd	ra, 24(sp)
   1ca2c: 3a f8        	sd	a4, 48(sp)
   1ca2e: 3e fc        	sd	a5, 56(sp)
   1ca30: c2 e0        	sd	a6, 64(sp)
   1ca32: c6 e4        	sd	a7, 72(sp)
   1ca34: 1a e4        	sd	t1, 8(sp)
   1ca36: ef e0 3f fd  	jal	0x1ba08 <_vfiprintf_r>
   1ca3a: e2 60        	ld	ra, 24(sp)
   1ca3c: 61 61        	addi	sp, sp, 80
   1ca3e: 82 80        	ret

000000000001ca40 <__fputwc>:
   1ca40: 79 71        	addi	sp, sp, -48
   1ca42: 22 f0        	sd	s0, 32(sp)
   1ca44: 26 ec        	sd	s1, 24(sp)
   1ca46: 4a e8        	sd	s2, 16(sp)
   1ca48: 06 f4        	sd	ra, 40(sp)
   1ca4a: 2a 89        	mv	s2, a0
   1ca4c: ae 84        	mv	s1, a1
   1ca4e: 32 84        	mv	s0, a2
   1ca50: ef b0 7f 8e  	jal	0x18336 <__locale_mb_cur_max>
   1ca54: 85 47        	addi	a5, zero, 1
   1ca56: 63 1b f5 00  	bne	a0, a5, 0x1ca6c <__fputwc+0x2c>
   1ca5a: 9b 87 f4 ff  	addiw	a5, s1, -1
   1ca5e: 13 07 e0 0f  	addi	a4, zero, 254
   1ca62: 63 65 f7 00  	bltu	a4, a5, 0x1ca6c <__fputwc+0x2c>
   1ca66: 23 04 91 00  	sb	s1, 8(sp)
   1ca6a: 21 a8        	j	0x1ca82 <__fputwc+0x42>
   1ca6c: 93 06 44 0a  	addi	a3, s0, 164
   1ca70: 26 86        	mv	a2, s1
   1ca72: 2c 00        	addi	a1, sp, 8
   1ca74: 4a 85        	mv	a0, s2
   1ca76: ef 10 b0 1c  	jal	0x1e440 <_wcrtomb_r>
   1ca7a: fd 57        	addi	a5, zero, -1
   1ca7c: 63 0d f5 04  	beq	a0, a5, 0x1cad6 <__fputwc+0x96>
   1ca80: 0d c5        	beqz	a0, 0x1caaa <__fputwc+0x6a>
   1ca82: 5c 44        	lw	a5, 12(s0)
   1ca84: 83 45 81 00  	lbu	a1, 8(sp)
   1ca88: 1b 87 f7 ff  	addiw	a4, a5, -1
   1ca8c: 58 c4        	sw	a4, 12(s0)
   1ca8e: 63 58 07 00  	bgez	a4, 0x1ca9e <__fputwc+0x5e>
   1ca92: 1c 54        	lw	a5, 40(s0)
   1ca94: 63 43 f7 02  	blt	a4, a5, 0x1caba <__fputwc+0x7a>
   1ca98: a9 47        	addi	a5, zero, 10
   1ca9a: 63 80 f5 02  	beq	a1, a5, 0x1caba <__fputwc+0x7a>
   1ca9e: 1c 60        	ld	a5, 0(s0)
   1caa0: 13 87 17 00  	addi	a4, a5, 1
   1caa4: 18 e0        	sd	a4, 0(s0)
   1caa6: 23 80 b7 00  	sb	a1, 0(a5)
   1caaa: a2 70        	ld	ra, 40(sp)
   1caac: 02 74        	ld	s0, 32(sp)
   1caae: 42 69        	ld	s2, 16(sp)
   1cab0: 1b 85 04 00  	sext.w	a0, s1
   1cab4: e2 64        	ld	s1, 24(sp)
   1cab6: 45 61        	addi	sp, sp, 48
   1cab8: 82 80        	ret
   1caba: 22 86        	mv	a2, s0
   1cabc: 4a 85        	mv	a0, s2
   1cabe: ef 10 30 08  	jal	0x1e340 <__swbuf_r>
   1cac2: fd 57        	addi	a5, zero, -1
   1cac4: e3 13 f5 fe  	bne	a0, a5, 0x1caaa <__fputwc+0x6a>
   1cac8: 7d 55        	addi	a0, zero, -1
   1caca: a2 70        	ld	ra, 40(sp)
   1cacc: 02 74        	ld	s0, 32(sp)
   1cace: e2 64        	ld	s1, 24(sp)
   1cad0: 42 69        	ld	s2, 16(sp)
   1cad2: 45 61        	addi	sp, sp, 48
   1cad4: 82 80        	ret
   1cad6: 83 57 04 01  	lhu	a5, 16(s0)
   1cada: 7d 55        	addi	a0, zero, -1
   1cadc: 93 e7 07 04  	ori	a5, a5, 64
   1cae0: 23 18 f4 00  	sh	a5, 16(s0)
   1cae4: dd b7        	j	0x1caca <__fputwc+0x8a>

000000000001cae6 <_fputwc_r>:
   1cae6: 83 17 06 01  	lh	a5, 16(a2)
   1caea: 13 97 27 03  	slli	a4, a5, 50
   1caee: 63 4c 07 00  	bltz	a4, 0x1cb06 <_fputwc_r+0x20>
   1caf2: 03 27 c6 0a  	lw	a4, 172(a2)
   1caf6: 89 66        	lui	a3, 2
   1caf8: d5 8f        	or	a5, a5, a3
   1cafa: 89 66        	lui	a3, 2
   1cafc: 55 8f        	or	a4, a4, a3
   1cafe: 23 18 f6 00  	sh	a5, 16(a2)
   1cb02: 23 26 e6 0a  	sw	a4, 172(a2)
   1cb06: 6f f0 bf f3  	j	0x1ca40 <__fputwc>

000000000001cb0a <fputwc>:
   1cb0a: 01 11        	addi	sp, sp, -32
   1cb0c: 22 e8        	sd	s0, 16(sp)
   1cb0e: 03 b4 81 93  	ld	s0, -1736(gp)
   1cb12: 06 ec        	sd	ra, 24(sp)
   1cb14: 2e 86        	mv	a2, a1
   1cb16: aa 85        	mv	a1, a0
   1cb18: 19 c0        	beqz	s0, 0x1cb1e <fputwc+0x14>
   1cb1a: 3c 48        	lw	a5, 80(s0)
   1cb1c: 9d c7        	beqz	a5, 0x1cb4a <fputwc+0x40>
   1cb1e: 83 17 06 01  	lh	a5, 16(a2)
   1cb22: 13 97 27 03  	slli	a4, a5, 50
   1cb26: 63 4c 07 00  	bltz	a4, 0x1cb3e <fputwc+0x34>
   1cb2a: 03 27 c6 0a  	lw	a4, 172(a2)
   1cb2e: 89 66        	lui	a3, 2
   1cb30: d5 8f        	or	a5, a5, a3
   1cb32: 89 66        	lui	a3, 2
   1cb34: 55 8f        	or	a4, a4, a3
   1cb36: 23 18 f6 00  	sh	a5, 16(a2)
   1cb3a: 23 26 e6 0a  	sw	a4, 172(a2)
   1cb3e: 22 85        	mv	a0, s0
   1cb40: 42 64        	ld	s0, 16(sp)
   1cb42: e2 60        	ld	ra, 24(sp)
   1cb44: 05 61        	addi	sp, sp, 32
   1cb46: 6f f0 bf ef  	j	0x1ca40 <__fputwc>
   1cb4a: 2a e0        	sd	a0, 0(sp)
   1cb4c: 22 85        	mv	a0, s0
   1cb4e: 32 e4        	sd	a2, 8(sp)
   1cb50: ef 80 3f be  	jal	0x15732 <__sinit>
   1cb54: 22 66        	ld	a2, 8(sp)
   1cb56: 82 65        	ld	a1, 0(sp)
   1cb58: d9 b7        	j	0x1cb1e <fputwc+0x14>

000000000001cb5a <_fstat_r>:
   1cb5a: 01 11        	addi	sp, sp, -32
   1cb5c: 2e 87        	mv	a4, a1
   1cb5e: 22 e8        	sd	s0, 16(sp)
   1cb60: 26 e4        	sd	s1, 8(sp)
   1cb62: 2a 84        	mv	s0, a0
   1cb64: b2 85        	mv	a1, a2
   1cb66: 3a 85        	mv	a0, a4
   1cb68: 06 ec        	sd	ra, 24(sp)
   1cb6a: 23 ac 01 96  	sw	zero, -1672(gp)
   1cb6e: ef 10 30 60  	jal	0x1e970 <_fstat>
   1cb72: fd 57        	addi	a5, zero, -1
   1cb74: 63 07 f5 00  	beq	a0, a5, 0x1cb82 <_fstat_r+0x28>
   1cb78: e2 60        	ld	ra, 24(sp)
   1cb7a: 42 64        	ld	s0, 16(sp)
   1cb7c: a2 64        	ld	s1, 8(sp)
   1cb7e: 05 61        	addi	sp, sp, 32
   1cb80: 82 80        	ret
   1cb82: 83 a7 81 97  	lw	a5, -1672(gp)
   1cb86: ed db        	beqz	a5, 0x1cb78 <_fstat_r+0x1e>
   1cb88: e2 60        	ld	ra, 24(sp)
   1cb8a: 1c c0        	sw	a5, 0(s0)
   1cb8c: 42 64        	ld	s0, 16(sp)
   1cb8e: a2 64        	ld	s1, 8(sp)
   1cb90: 05 61        	addi	sp, sp, 32
   1cb92: 82 80        	ret

000000000001cb94 <__sfvwrite_r>:
   1cb94: 1c 6a        	ld	a5, 16(a2)
   1cb96: 63 8a 07 1a  	beqz	a5, 0x1cd4a <__sfvwrite_r+0x1b6>
   1cb9a: 03 97 05 01  	lh	a4, 16(a1)
   1cb9e: 1d 71        	addi	sp, sp, -96
   1cba0: a2 e8        	sd	s0, 80(sp)
   1cba2: 52 f8        	sd	s4, 48(sp)
   1cba4: 5a f0        	sd	s6, 32(sp)
   1cba6: 86 ec        	sd	ra, 88(sp)
   1cba8: a6 e4        	sd	s1, 72(sp)
   1cbaa: ca e0        	sd	s2, 64(sp)
   1cbac: 4e fc        	sd	s3, 56(sp)
   1cbae: 56 f4        	sd	s5, 40(sp)
   1cbb0: 5e ec        	sd	s7, 24(sp)
   1cbb2: 62 e8        	sd	s8, 16(sp)
   1cbb4: 66 e4        	sd	s9, 8(sp)
   1cbb6: 93 77 87 00  	andi	a5, a4, 8
   1cbba: 32 8b        	mv	s6, a2
   1cbbc: 2a 8a        	mv	s4, a0
   1cbbe: 2e 84        	mv	s0, a1
   1cbc0: b5 c3        	beqz	a5, 0x1cc24 <__sfvwrite_r+0x90>
   1cbc2: 9c 6d        	ld	a5, 24(a1)
   1cbc4: a5 c3        	beqz	a5, 0x1cc24 <__sfvwrite_r+0x90>
   1cbc6: 93 77 27 00  	andi	a5, a4, 2
   1cbca: 83 34 0b 00  	ld	s1, 0(s6)
   1cbce: a5 cb        	beqz	a5, 0x1cc3e <__sfvwrite_r+0xaa>
   1cbd0: 3c 60        	ld	a5, 64(s0)
   1cbd2: 0c 78        	ld	a1, 48(s0)
   1cbd4: b7 0a 00 80  	lui	s5, 524288
   1cbd8: 81 49        	mv	s3, zero
   1cbda: 01 49        	mv	s2, zero
   1cbdc: 93 ca 0a c0  	xori	s5, s5, -1024
   1cbe0: 4e 86        	mv	a2, s3
   1cbe2: 52 85        	mv	a0, s4
   1cbe4: 63 0a 09 02  	beqz	s2, 0x1cc18 <__sfvwrite_r+0x84>
   1cbe8: ca 86        	mv	a3, s2
   1cbea: 63 f3 2a 01  	bgeu	s5, s2, 0x1cbf0 <__sfvwrite_r+0x5c>
   1cbee: d6 86        	mv	a3, s5
   1cbf0: 81 26        	sext.w	a3, a3
   1cbf2: 82 97        	jalr	a5
   1cbf4: 63 53 a0 14  	blez	a0, 0x1cd3a <__sfvwrite_r+0x1a6>
   1cbf8: 83 37 0b 01  	ld	a5, 16(s6)
   1cbfc: aa 99        	add	s3, s3, a0
   1cbfe: 33 09 a9 40  	sub	s2, s2, a0
   1cc02: 89 8f        	sub	a5, a5, a0
   1cc04: 23 38 fb 00  	sd	a5, 16(s6)
   1cc08: 63 86 07 10  	beqz	a5, 0x1cd14 <__sfvwrite_r+0x180>
   1cc0c: 3c 60        	ld	a5, 64(s0)
   1cc0e: 0c 78        	ld	a1, 48(s0)
   1cc10: 4e 86        	mv	a2, s3
   1cc12: 52 85        	mv	a0, s4
   1cc14: e3 1a 09 fc  	bnez	s2, 0x1cbe8 <__sfvwrite_r+0x54>
   1cc18: 83 b9 04 00  	ld	s3, 0(s1)
   1cc1c: 03 b9 84 00  	ld	s2, 8(s1)
   1cc20: c1 04        	addi	s1, s1, 16
   1cc22: 7d bf        	j	0x1cbe0 <__sfvwrite_r+0x4c>
   1cc24: a2 85        	mv	a1, s0
   1cc26: 52 85        	mv	a0, s4
   1cc28: ef 80 8f d1  	jal	0x15140 <__swsetup_r>
   1cc2c: 63 1d 05 10  	bnez	a0, 0x1cd46 <__sfvwrite_r+0x1b2>
   1cc30: 03 17 04 01  	lh	a4, 16(s0)
   1cc34: 83 34 0b 00  	ld	s1, 0(s6)
   1cc38: 93 77 27 00  	andi	a5, a4, 2
   1cc3c: d1 fb        	bnez	a5, 0x1cbd0 <__sfvwrite_r+0x3c>
   1cc3e: 93 77 17 00  	andi	a5, a4, 1
   1cc42: a5 eb        	bnez	a5, 0x1ccb2 <__sfvwrite_r+0x11e>
   1cc44: 83 2b c4 00  	lw	s7, 12(s0)
   1cc48: 1c 60        	ld	a5, 0(s0)
   1cc4a: b7 0a 00 80  	lui	s5, 524288
   1cc4e: 01 4c        	mv	s8, zero
   1cc50: 01 49        	mv	s2, zero
   1cc52: 93 ca fa ff  	not	s5, s5
   1cc56: 3e 85        	mv	a0, a5
   1cc58: 63 07 09 04  	beqz	s2, 0x1cca6 <__sfvwrite_r+0x112>
   1cc5c: 93 76 07 20  	andi	a3, a4, 512
   1cc60: 63 89 06 12  	beqz	a3, 0x1cd92 <__sfvwrite_r+0x1fe>
   1cc64: 63 78 79 19  	bgeu	s2, s7, 0x1cdf4 <__sfvwrite_r+0x260>
   1cc68: ca 8c        	mv	s9, s2
   1cc6a: ca 8b        	mv	s7, s2
   1cc6c: 5e 86        	mv	a2, s7
   1cc6e: e2 85        	mv	a1, s8
   1cc70: ef 00 a0 3c  	jal	0x1d03a <memmove>
   1cc74: 58 44        	lw	a4, 12(s0)
   1cc76: 1c 60        	ld	a5, 0(s0)
   1cc78: ca 89        	mv	s3, s2
   1cc7a: 3b 07 97 41  	subw	a4, a4, s9
   1cc7e: de 97        	add	a5, a5, s7
   1cc80: 58 c4        	sw	a4, 12(s0)
   1cc82: 1c e0        	sd	a5, 0(s0)
   1cc84: 01 49        	mv	s2, zero
   1cc86: 83 37 0b 01  	ld	a5, 16(s6)
   1cc8a: 4e 9c        	add	s8, s8, s3
   1cc8c: b3 87 37 41  	sub	a5, a5, s3
   1cc90: 23 38 fb 00  	sd	a5, 16(s6)
   1cc94: c1 c3        	beqz	a5, 0x1cd14 <__sfvwrite_r+0x180>
   1cc96: 1c 60        	ld	a5, 0(s0)
   1cc98: 03 17 04 01  	lh	a4, 16(s0)
   1cc9c: 83 2b c4 00  	lw	s7, 12(s0)
   1cca0: 3e 85        	mv	a0, a5
   1cca2: e3 1d 09 fa  	bnez	s2, 0x1cc5c <__sfvwrite_r+0xc8>
   1cca6: 03 bc 04 00  	ld	s8, 0(s1)
   1ccaa: 03 b9 84 00  	ld	s2, 8(s1)
   1ccae: c1 04        	addi	s1, s1, 16
   1ccb0: 5d b7        	j	0x1cc56 <__sfvwrite_r+0xc2>
   1ccb2: 81 49        	mv	s3, zero
   1ccb4: 01 45        	mv	a0, zero
   1ccb6: 81 4c        	mv	s9, zero
   1ccb8: 01 4c        	mv	s8, zero
   1ccba: 63 0a 0c 08  	beqz	s8, 0x1cd4e <__sfvwrite_r+0x1ba>
   1ccbe: 45 c1        	beqz	a0, 0x1cd5e <__sfvwrite_r+0x1ca>
   1ccc0: ce 8b        	mv	s7, s3
   1ccc2: 63 73 3c 01  	bgeu	s8, s3, 0x1ccc8 <__sfvwrite_r+0x134>
   1ccc6: e2 8b        	mv	s7, s8
   1ccc8: 08 60        	ld	a0, 0(s0)
   1ccca: 18 6c        	ld	a4, 24(s0)
   1cccc: 5c 44        	lw	a5, 12(s0)
   1ccce: 14 50        	lw	a3, 32(s0)
   1ccd0: 5e 89        	mv	s2, s7
   1ccd2: 63 76 a7 00  	bgeu	a4, a0, 0x1ccde <__sfvwrite_r+0x14a>
   1ccd6: bb 8a d7 00  	addw	s5, a5, a3
   1ccda: 63 cd 7a 09  	blt	s5, s7, 0x1cd74 <__sfvwrite_r+0x1e0>
   1ccde: 63 c7 db 18  	blt	s7, a3, 0x1ce6c <__sfvwrite_r+0x2d8>
   1cce2: 3c 60        	ld	a5, 64(s0)
   1cce4: 0c 78        	ld	a1, 48(s0)
   1cce6: 66 86        	mv	a2, s9
   1cce8: 52 85        	mv	a0, s4
   1ccea: 82 97        	jalr	a5
   1ccec: 2a 89        	mv	s2, a0
   1ccee: 63 56 a0 04  	blez	a0, 0x1cd3a <__sfvwrite_r+0x1a6>
   1ccf2: 9b 07 05 00  	sext.w	a5, a0
   1ccf6: bb 89 f9 40  	subw	s3, s3, a5
   1ccfa: 05 45        	addi	a0, zero, 1
   1ccfc: 63 8a 09 02  	beqz	s3, 0x1cd30 <__sfvwrite_r+0x19c>
   1cd00: 83 37 0b 01  	ld	a5, 16(s6)
   1cd04: ca 9c        	add	s9, s9, s2
   1cd06: 33 0c 2c 41  	sub	s8, s8, s2
   1cd0a: b3 87 27 41  	sub	a5, a5, s2
   1cd0e: 23 38 fb 00  	sd	a5, 16(s6)
   1cd12: c5 f7        	bnez	a5, 0x1ccba <__sfvwrite_r+0x126>
   1cd14: 01 45        	mv	a0, zero
   1cd16: e6 60        	ld	ra, 88(sp)
   1cd18: 46 64        	ld	s0, 80(sp)
   1cd1a: a6 64        	ld	s1, 72(sp)
   1cd1c: 06 69        	ld	s2, 64(sp)
   1cd1e: e2 79        	ld	s3, 56(sp)
   1cd20: 42 7a        	ld	s4, 48(sp)
   1cd22: a2 7a        	ld	s5, 40(sp)
   1cd24: 02 7b        	ld	s6, 32(sp)
   1cd26: e2 6b        	ld	s7, 24(sp)
   1cd28: 42 6c        	ld	s8, 16(sp)
   1cd2a: a2 6c        	ld	s9, 8(sp)
   1cd2c: 25 61        	addi	sp, sp, 96
   1cd2e: 82 80        	ret
   1cd30: a2 85        	mv	a1, s0
   1cd32: 52 85        	mv	a0, s4
   1cd34: ef 80 cf f2  	jal	0x15460 <_fflush_r>
   1cd38: 61 d5        	beqz	a0, 0x1cd00 <__sfvwrite_r+0x16c>
   1cd3a: 83 17 04 01  	lh	a5, 16(s0)
   1cd3e: 93 e7 07 04  	ori	a5, a5, 64
   1cd42: 23 18 f4 00  	sh	a5, 16(s0)
   1cd46: 7d 55        	addi	a0, zero, -1
   1cd48: f9 b7        	j	0x1cd16 <__sfvwrite_r+0x182>
   1cd4a: 01 45        	mv	a0, zero
   1cd4c: 82 80        	ret
   1cd4e: 03 bc 84 00  	ld	s8, 8(s1)
   1cd52: a6 87        	mv	a5, s1
   1cd54: c1 04        	addi	s1, s1, 16
   1cd56: e3 0c 0c fe  	beqz	s8, 0x1cd4e <__sfvwrite_r+0x1ba>
   1cd5a: 83 bc 07 00  	ld	s9, 0(a5)
   1cd5e: 62 86        	mv	a2, s8
   1cd60: a9 45        	addi	a1, zero, 10
   1cd62: 66 85        	mv	a0, s9
   1cd64: ef b0 9f e0  	jal	0x18b6c <memchr>
   1cd68: 63 04 05 14  	beqz	a0, 0x1ceb0 <__sfvwrite_r+0x31c>
   1cd6c: 05 05        	addi	a0, a0, 1
   1cd6e: bb 09 95 41  	subw	s3, a0, s9
   1cd72: b9 b7        	j	0x1ccc0 <__sfvwrite_r+0x12c>
   1cd74: e6 85        	mv	a1, s9
   1cd76: 56 86        	mv	a2, s5
   1cd78: ef 00 20 2c  	jal	0x1d03a <memmove>
   1cd7c: 1c 60        	ld	a5, 0(s0)
   1cd7e: a2 85        	mv	a1, s0
   1cd80: 52 85        	mv	a0, s4
   1cd82: d6 97        	add	a5, a5, s5
   1cd84: 1c e0        	sd	a5, 0(s0)
   1cd86: ef 80 af ed  	jal	0x15460 <_fflush_r>
   1cd8a: 45 f9        	bnez	a0, 0x1cd3a <__sfvwrite_r+0x1a6>
   1cd8c: d6 87        	mv	a5, s5
   1cd8e: 56 89        	mv	s2, s5
   1cd90: 9d b7        	j	0x1ccf6 <__sfvwrite_r+0x162>
   1cd92: 18 6c        	ld	a4, 24(s0)
   1cd94: 63 67 f7 02  	bltu	a4, a5, 0x1cdc2 <__sfvwrite_r+0x22e>
   1cd98: 18 50        	lw	a4, 32(s0)
   1cd9a: 63 64 e9 02  	bltu	s2, a4, 0x1cdc2 <__sfvwrite_r+0x22e>
   1cd9e: ca 86        	mv	a3, s2
   1cda0: 63 f3 2a 01  	bgeu	s5, s2, 0x1cda6 <__sfvwrite_r+0x212>
   1cda4: d6 86        	mv	a3, s5
   1cda6: 3b e7 e6 02  	remw	a4, a3, a4
   1cdaa: 3c 60        	ld	a5, 64(s0)
   1cdac: 0c 78        	ld	a1, 48(s0)
   1cdae: 62 86        	mv	a2, s8
   1cdb0: 52 85        	mv	a0, s4
   1cdb2: 99 9e        	subw	a3, a3, a4
   1cdb4: 82 97        	jalr	a5
   1cdb6: aa 89        	mv	s3, a0
   1cdb8: e3 51 a0 f8  	blez	a0, 0x1cd3a <__sfvwrite_r+0x1a6>
   1cdbc: 33 09 39 41  	sub	s2, s2, s3
   1cdc0: d9 b5        	j	0x1cc86 <__sfvwrite_r+0xf2>
   1cdc2: de 89        	mv	s3, s7
   1cdc4: 63 73 79 01  	bgeu	s2, s7, 0x1cdca <__sfvwrite_r+0x236>
   1cdc8: ca 89        	mv	s3, s2
   1cdca: 3e 85        	mv	a0, a5
   1cdcc: 4e 86        	mv	a2, s3
   1cdce: e2 85        	mv	a1, s8
   1cdd0: ef 00 a0 26  	jal	0x1d03a <memmove>
   1cdd4: 58 44        	lw	a4, 12(s0)
   1cdd6: 1c 60        	ld	a5, 0(s0)
   1cdd8: bb 06 37 41  	subw	a3, a4, s3
   1cddc: ce 97        	add	a5, a5, s3
   1cdde: 54 c4        	sw	a3, 12(s0)
   1cde0: 1c e0        	sd	a5, 0(s0)
   1cde2: e9 fe        	bnez	a3, 0x1cdbc <__sfvwrite_r+0x228>
   1cde4: a2 85        	mv	a1, s0
   1cde6: 52 85        	mv	a0, s4
   1cde8: ef 80 8f e7  	jal	0x15460 <_fflush_r>
   1cdec: 39 f5        	bnez	a0, 0x1cd3a <__sfvwrite_r+0x1a6>
   1cdee: 33 09 39 41  	sub	s2, s2, s3
   1cdf2: 51 bd        	j	0x1cc86 <__sfvwrite_r+0xf2>
   1cdf4: 93 76 07 48  	andi	a3, a4, 1152
   1cdf8: c1 ca        	beqz	a3, 0x1ce88 <__sfvwrite_r+0x2f4>
   1cdfa: 10 50        	lw	a2, 32(s0)
   1cdfc: 0c 6c        	ld	a1, 24(s0)
   1cdfe: 9b 16 16 00  	slliw	a3, a2, 1
   1ce02: b1 9e        	addw	a3, a3, a2
   1ce04: 8d 8f        	sub	a5, a5, a1
   1ce06: 9b d9 f6 01  	srliw	s3, a3, 31
   1ce0a: 9b 8b 07 00  	sext.w	s7, a5
   1ce0e: bb 89 d9 00  	addw	s3, s3, a3
   1ce12: 93 86 1b 00  	addi	a3, s7, 1
   1ce16: 9b d9 19 40  	sraiw	s3, s3, 1
   1ce1a: ca 96        	add	a3, a3, s2
   1ce1c: 63 f5 d9 00  	bgeu	s3, a3, 0x1ce26 <__sfvwrite_r+0x292>
   1ce20: 85 27        	addiw	a5, a5, 1
   1ce22: bb 89 27 01  	addw	s3, a5, s2
   1ce26: 13 77 07 40  	andi	a4, a4, 1024
   1ce2a: 2d c3        	beqz	a4, 0x1ce8c <__sfvwrite_r+0x2f8>
   1ce2c: ce 85        	mv	a1, s3
   1ce2e: 52 85        	mv	a0, s4
   1ce30: ef b0 0f f0  	jal	0x18530 <_malloc_r>
   1ce34: aa 8c        	mv	s9, a0
   1ce36: 41 c1        	beqz	a0, 0x1ceb6 <__sfvwrite_r+0x322>
   1ce38: 0c 6c        	ld	a1, 24(s0)
   1ce3a: 5e 86        	mv	a2, s7
   1ce3c: ef 00 80 0f  	jal	0x1cf34 <memcpy>
   1ce40: 83 57 04 01  	lhu	a5, 16(s0)
   1ce44: 93 f7 f7 b7  	andi	a5, a5, -1153
   1ce48: 93 e7 07 08  	ori	a5, a5, 128
   1ce4c: 23 18 f4 00  	sh	a5, 16(s0)
   1ce50: 33 85 7c 01  	add	a0, s9, s7
   1ce54: bb 8b 79 41  	subw	s7, s3, s7
   1ce58: 23 3c 94 01  	sd	s9, 24(s0)
   1ce5c: 23 26 74 01  	sw	s7, 12(s0)
   1ce60: 08 e0        	sd	a0, 0(s0)
   1ce62: 23 20 34 03  	sw	s3, 32(s0)
   1ce66: ca 8c        	mv	s9, s2
   1ce68: ca 8b        	mv	s7, s2
   1ce6a: 09 b5        	j	0x1cc6c <__sfvwrite_r+0xd8>
   1ce6c: 5e 86        	mv	a2, s7
   1ce6e: e6 85        	mv	a1, s9
   1ce70: ef 00 a0 1c  	jal	0x1d03a <memmove>
   1ce74: 54 44        	lw	a3, 12(s0)
   1ce76: 18 60        	ld	a4, 0(s0)
   1ce78: 9b 87 0b 00  	sext.w	a5, s7
   1ce7c: bb 86 76 41  	subw	a3, a3, s7
   1ce80: 5e 97        	add	a4, a4, s7
   1ce82: 54 c4        	sw	a3, 12(s0)
   1ce84: 18 e0        	sd	a4, 0(s0)
   1ce86: 85 bd        	j	0x1ccf6 <__sfvwrite_r+0x162>
   1ce88: de 8c        	mv	s9, s7
   1ce8a: cd b3        	j	0x1cc6c <__sfvwrite_r+0xd8>
   1ce8c: 4e 86        	mv	a2, s3
   1ce8e: 52 85        	mv	a0, s4
   1ce90: ef 00 e0 2b  	jal	0x1d14e <_realloc_r>
   1ce94: aa 8c        	mv	s9, a0
   1ce96: 4d fd        	bnez	a0, 0x1ce50 <__sfvwrite_r+0x2bc>
   1ce98: 0c 6c        	ld	a1, 24(s0)
   1ce9a: 52 85        	mv	a0, s4
   1ce9c: ef 80 bf 9c  	jal	0x15866 <_free_r>
   1cea0: 83 17 04 01  	lh	a5, 16(s0)
   1cea4: 31 47        	addi	a4, zero, 12
   1cea6: 23 20 ea 00  	sw	a4, 0(s4)
   1ceaa: 93 f7 f7 f7  	andi	a5, a5, -129
   1ceae: 41 bd        	j	0x1cd3e <__sfvwrite_r+0x1aa>
   1ceb0: 9b 09 1c 00  	addiw	s3, s8, 1
   1ceb4: 31 b5        	j	0x1ccc0 <__sfvwrite_r+0x12c>
   1ceb6: 31 47        	addi	a4, zero, 12
   1ceb8: 83 17 04 01  	lh	a5, 16(s0)
   1cebc: 23 20 ea 00  	sw	a4, 0(s4)
   1cec0: bd bd        	j	0x1cd3e <__sfvwrite_r+0x1aa>

000000000001cec2 <_isatty_r>:
   1cec2: 01 11        	addi	sp, sp, -32
   1cec4: 22 e8        	sd	s0, 16(sp)
   1cec6: 26 e4        	sd	s1, 8(sp)
   1cec8: 2a 84        	mv	s0, a0
   1ceca: 2e 85        	mv	a0, a1
   1cecc: 06 ec        	sd	ra, 24(sp)
   1cece: 23 ac 01 96  	sw	zero, -1672(gp)
   1ced2: ef 10 30 2e  	jal	0x1e9b4 <_isatty>
   1ced6: fd 57        	addi	a5, zero, -1
   1ced8: 63 07 f5 00  	beq	a0, a5, 0x1cee6 <_isatty_r+0x24>
   1cedc: e2 60        	ld	ra, 24(sp)
   1cede: 42 64        	ld	s0, 16(sp)
   1cee0: a2 64        	ld	s1, 8(sp)
   1cee2: 05 61        	addi	sp, sp, 32
   1cee4: 82 80        	ret
   1cee6: 83 a7 81 97  	lw	a5, -1672(gp)
   1ceea: ed db        	beqz	a5, 0x1cedc <_isatty_r+0x1a>
   1ceec: e2 60        	ld	ra, 24(sp)
   1ceee: 1c c0        	sw	a5, 0(s0)
   1cef0: 42 64        	ld	s0, 16(sp)
   1cef2: a2 64        	ld	s1, 8(sp)
   1cef4: 05 61        	addi	sp, sp, 32
   1cef6: 82 80        	ret

000000000001cef8 <_lseek_r>:
   1cef8: 01 11        	addi	sp, sp, -32
   1cefa: 2e 87        	mv	a4, a1
   1cefc: 22 e8        	sd	s0, 16(sp)
   1cefe: 26 e4        	sd	s1, 8(sp)
   1cf00: b2 85        	mv	a1, a2
   1cf02: 2a 84        	mv	s0, a0
   1cf04: 36 86        	mv	a2, a3
   1cf06: 3a 85        	mv	a0, a4
   1cf08: 06 ec        	sd	ra, 24(sp)
   1cf0a: 23 ac 01 96  	sw	zero, -1672(gp)
   1cf0e: ef 10 10 2e  	jal	0x1e9ee <_lseek>
   1cf12: fd 57        	addi	a5, zero, -1
   1cf14: 63 07 f5 00  	beq	a0, a5, 0x1cf22 <_lseek_r+0x2a>
   1cf18: e2 60        	ld	ra, 24(sp)
   1cf1a: 42 64        	ld	s0, 16(sp)
   1cf1c: a2 64        	ld	s1, 8(sp)
   1cf1e: 05 61        	addi	sp, sp, 32
   1cf20: 82 80        	ret
   1cf22: 83 a7 81 97  	lw	a5, -1672(gp)
   1cf26: ed db        	beqz	a5, 0x1cf18 <_lseek_r+0x20>
   1cf28: e2 60        	ld	ra, 24(sp)
   1cf2a: 1c c0        	sw	a5, 0(s0)
   1cf2c: 42 64        	ld	s0, 16(sp)
   1cf2e: a2 64        	ld	s1, 8(sp)
   1cf30: 05 61        	addi	sp, sp, 32
   1cf32: 82 80        	ret

000000000001cf34 <memcpy>:
   1cf34: b3 47 b5 00  	xor	a5, a0, a1
   1cf38: 9d 8b        	andi	a5, a5, 7
   1cf3a: b3 08 c5 00  	add	a7, a0, a2
   1cf3e: f9 eb        	bnez	a5, 0x1d014 <memcpy+0xe0>
   1cf40: 9d 47        	addi	a5, zero, 7
   1cf42: 63 f9 c7 0c  	bgeu	a5, a2, 0x1d014 <memcpy+0xe0>
   1cf46: 93 77 75 00  	andi	a5, a0, 7
   1cf4a: 2a 87        	mv	a4, a0
   1cf4c: a9 eb        	bnez	a5, 0x1cf9e <memcpy+0x6a>
   1cf4e: 93 f7 88 ff  	andi	a5, a7, -8
   1cf52: 33 86 e7 40  	sub	a2, a5, a4
   1cf56: 93 06 00 04  	addi	a3, zero, 64
   1cf5a: 93 02 00 04  	addi	t0, zero, 64
   1cf5e: 63 c3 c6 06  	blt	a3, a2, 0x1cfc4 <memcpy+0x90>
   1cf62: 63 7b f7 02  	bgeu	a4, a5, 0x1cf98 <memcpy+0x64>
   1cf66: fd 17        	addi	a5, a5, -1
   1cf68: 99 8f        	sub	a5, a5, a4
   1cf6a: 8d 83        	srli	a5, a5, 3
   1cf6c: 13 88 17 00  	addi	a6, a5, 1
   1cf70: ba 86        	mv	a3, a4
   1cf72: 2e 86        	mv	a2, a1
   1cf74: c2 87        	mv	a5, a6
   1cf76: 73 23 20 c2  	csrr	t1, vlenb
   1cf7a: 57 fe 87 05  	<unknown>
   1cf7e: 07 7c 06 02  	<unknown>
   1cf82: b3 87 c7 41  	sub	a5, a5, t3
   1cf86: 27 fc 06 02  	<unknown>
   1cf8a: 1a 96        	add	a2, a2, t1
   1cf8c: 9a 96        	add	a3, a3, t1
   1cf8e: f5 f7        	bnez	a5, 0x1cf7a <memcpy+0x46>
   1cf90: 93 17 38 00  	slli	a5, a6, 3
   1cf94: 3e 97        	add	a4, a4, a5
   1cf96: be 95        	add	a1, a1, a5
   1cf98: 63 61 17 09  	bltu	a4, a7, 0x1d01a <memcpy+0xe6>
   1cf9c: 82 80        	ret
   1cf9e: 83 c6 05 00  	lbu	a3, 0(a1)
   1cfa2: 05 07        	addi	a4, a4, 1
   1cfa4: 93 77 77 00  	andi	a5, a4, 7
   1cfa8: a3 0f d7 fe  	sb	a3, -1(a4)
   1cfac: 85 05        	addi	a1, a1, 1
   1cfae: c5 d3        	beqz	a5, 0x1cf4e <memcpy+0x1a>
   1cfb0: 83 c6 05 00  	lbu	a3, 0(a1)
   1cfb4: 05 07        	addi	a4, a4, 1
   1cfb6: 93 77 77 00  	andi	a5, a4, 7
   1cfba: a3 0f d7 fe  	sb	a3, -1(a4)
   1cfbe: 85 05        	addi	a1, a1, 1
   1cfc0: f9 ff        	bnez	a5, 0x1cf9e <memcpy+0x6a>
   1cfc2: 71 b7        	j	0x1cf4e <memcpy+0x1a>
   1cfc4: 90 65        	ld	a2, 8(a1)
   1cfc6: 94 7d        	ld	a3, 56(a1)
   1cfc8: 83 bf 05 00  	ld	t6, 0(a1)
   1cfcc: 03 bf 05 01  	ld	t5, 16(a1)
   1cfd0: 83 be 85 01  	ld	t4, 24(a1)
   1cfd4: 03 be 05 02  	ld	t3, 32(a1)
   1cfd8: 03 b3 85 02  	ld	t1, 40(a1)
   1cfdc: 03 b8 05 03  	ld	a6, 48(a1)
   1cfe0: 10 e7        	sd	a2, 8(a4)
   1cfe2: b0 61        	ld	a2, 64(a1)
   1cfe4: 23 30 f7 01  	sd	t6, 0(a4)
   1cfe8: 23 38 e7 01  	sd	t5, 16(a4)
   1cfec: 23 3c d7 01  	sd	t4, 24(a4)
   1cff0: 23 30 c7 03  	sd	t3, 32(a4)
   1cff4: 23 34 67 02  	sd	t1, 40(a4)
   1cff8: 23 38 07 03  	sd	a6, 48(a4)
   1cffc: 14 ff        	sd	a3, 56(a4)
   1cffe: 13 07 87 04  	addi	a4, a4, 72
   1d002: b3 86 e7 40  	sub	a3, a5, a4
   1d006: 23 3c c7 fe  	sd	a2, -8(a4)
   1d00a: 93 85 85 04  	addi	a1, a1, 72
   1d00e: e3 cb d2 fa  	blt	t0, a3, 0x1cfc4 <memcpy+0x90>
   1d012: 81 bf        	j	0x1cf62 <memcpy+0x2e>
   1d014: 2a 87        	mv	a4, a0
   1d016: 63 71 15 03  	bgeu	a0, a7, 0x1d038 <memcpy+0x104>
   1d01a: b3 87 e8 40  	sub	a5, a7, a4
   1d01e: f3 26 20 c2  	csrr	a3, vlenb
   1d022: 57 f6 07 04  	<unknown>
   1d026: 07 8c 05 02  	<unknown>
   1d02a: 91 8f        	sub	a5, a5, a2
   1d02c: 27 0c 07 02  	<unknown>
   1d030: b6 95        	add	a1, a1, a3
   1d032: 36 97        	add	a4, a4, a3
   1d034: fd f7        	bnez	a5, 0x1d022 <memcpy+0xee>
   1d036: 82 80        	ret
   1d038: 82 80        	ret

000000000001d03a <memmove>:
   1d03a: 63 f2 a5 02  	bgeu	a1, a0, 0x1d05e <memmove+0x24>
   1d03e: 33 87 c5 00  	add	a4, a1, a2
   1d042: 63 7e e5 00  	bgeu	a0, a4, 0x1d05e <memmove+0x24>
   1d046: b3 07 c5 00  	add	a5, a0, a2
   1d04a: 1d ca        	beqz	a2, 0x1d080 <memmove+0x46>
   1d04c: 83 46 f7 ff  	lbu	a3, -1(a4)
   1d050: fd 17        	addi	a5, a5, -1
   1d052: 7d 17        	addi	a4, a4, -1
   1d054: 23 80 d7 00  	sb	a3, 0(a5)
   1d058: e3 1a f5 fe  	bne	a0, a5, 0x1d04c <memmove+0x12>
   1d05c: 82 80        	ret
   1d05e: fd 47        	addi	a5, zero, 31
   1d060: 63 e1 c7 02  	bltu	a5, a2, 0x1d082 <memmove+0x48>
   1d064: aa 87        	mv	a5, a0
   1d066: 93 06 f6 ff  	addi	a3, a2, -1
   1d06a: 4d c2        	beqz	a2, 0x1d10c <memmove+0xd2>
   1d06c: 85 06        	addi	a3, a3, 1
   1d06e: be 96        	add	a3, a3, a5
   1d070: 03 c7 05 00  	lbu	a4, 0(a1)
   1d074: 85 07        	addi	a5, a5, 1
   1d076: 85 05        	addi	a1, a1, 1
   1d078: a3 8f e7 fe  	sb	a4, -1(a5)
   1d07c: e3 9a d7 fe  	bne	a5, a3, 0x1d070 <memmove+0x36>
   1d080: 82 80        	ret
   1d082: b3 67 b5 00  	or	a5, a0, a1
   1d086: 9d 8b        	andi	a5, a5, 7
   1d088: b5 ef        	bnez	a5, 0x1d104 <memmove+0xca>
   1d08a: 93 08 06 fe  	addi	a7, a2, -32
   1d08e: 93 f8 08 fe  	andi	a7, a7, -32
   1d092: 93 88 08 02  	addi	a7, a7, 32
   1d096: b3 07 15 01  	add	a5, a0, a7
   1d09a: ae 86        	mv	a3, a1
   1d09c: 2a 87        	mv	a4, a0
   1d09e: 03 b8 06 00  	ld	a6, 0(a3)
   1d0a2: 93 86 06 02  	addi	a3, a3, 32
   1d0a6: 13 07 07 02  	addi	a4, a4, 32
   1d0aa: 23 30 07 ff  	sd	a6, -32(a4)
   1d0ae: 03 b8 86 fe  	ld	a6, -24(a3)
   1d0b2: 23 34 07 ff  	sd	a6, -24(a4)
   1d0b6: 03 b8 06 ff  	ld	a6, -16(a3)
   1d0ba: 23 38 07 ff  	sd	a6, -16(a4)
   1d0be: 03 b8 86 ff  	ld	a6, -8(a3)
   1d0c2: 23 3c 07 ff  	sd	a6, -8(a4)
   1d0c6: e3 1c f7 fc  	bne	a4, a5, 0x1d09e <memmove+0x64>
   1d0ca: 13 77 86 01  	andi	a4, a2, 24
   1d0ce: c6 95        	add	a1, a1, a7
   1d0d0: 13 78 f6 01  	andi	a6, a2, 31
   1d0d4: 0d cf        	beqz	a4, 0x1d10e <memmove+0xd4>
   1d0d6: ae 86        	mv	a3, a1
   1d0d8: 3e 87        	mv	a4, a5
   1d0da: b3 8e 07 01  	add	t4, a5, a6
   1d0de: 1d 4e        	addi	t3, zero, 7
   1d0e0: 03 b3 06 00  	ld	t1, 0(a3)
   1d0e4: 21 07        	addi	a4, a4, 8
   1d0e6: b3 88 ee 40  	sub	a7, t4, a4
   1d0ea: 23 3c 67 fe  	sd	t1, -8(a4)
   1d0ee: a1 06        	addi	a3, a3, 8
   1d0f0: e3 68 1e ff  	bltu	t3, a7, 0x1d0e0 <memmove+0xa6>
   1d0f4: 13 07 88 ff  	addi	a4, a6, -8
   1d0f8: 61 9b        	andi	a4, a4, -8
   1d0fa: 21 07        	addi	a4, a4, 8
   1d0fc: 1d 8a        	andi	a2, a2, 7
   1d0fe: ba 97        	add	a5, a5, a4
   1d100: ba 95        	add	a1, a1, a4
   1d102: 95 b7        	j	0x1d066 <memmove+0x2c>
   1d104: 93 06 f6 ff  	addi	a3, a2, -1
   1d108: aa 87        	mv	a5, a0
   1d10a: 8d b7        	j	0x1d06c <memmove+0x32>
   1d10c: 82 80        	ret
   1d10e: 42 86        	mv	a2, a6
   1d110: 99 bf        	j	0x1d066 <memmove+0x2c>

000000000001d112 <_read_r>:
   1d112: 01 11        	addi	sp, sp, -32
   1d114: 2e 87        	mv	a4, a1
   1d116: 22 e8        	sd	s0, 16(sp)
   1d118: 26 e4        	sd	s1, 8(sp)
   1d11a: b2 85        	mv	a1, a2
   1d11c: 2a 84        	mv	s0, a0
   1d11e: 36 86        	mv	a2, a3
   1d120: 3a 85        	mv	a0, a4
   1d122: 06 ec        	sd	ra, 24(sp)
   1d124: 23 ac 01 96  	sw	zero, -1672(gp)
   1d128: ef 10 30 0f  	jal	0x1ea1a <_read>
   1d12c: fd 57        	addi	a5, zero, -1
   1d12e: 63 07 f5 00  	beq	a0, a5, 0x1d13c <_read_r+0x2a>
   1d132: e2 60        	ld	ra, 24(sp)
   1d134: 42 64        	ld	s0, 16(sp)
   1d136: a2 64        	ld	s1, 8(sp)
   1d138: 05 61        	addi	sp, sp, 32
   1d13a: 82 80        	ret
   1d13c: 83 a7 81 97  	lw	a5, -1672(gp)
   1d140: ed db        	beqz	a5, 0x1d132 <_read_r+0x20>
   1d142: e2 60        	ld	ra, 24(sp)
   1d144: 1c c0        	sw	a5, 0(s0)
   1d146: 42 64        	ld	s0, 16(sp)
   1d148: a2 64        	ld	s1, 8(sp)
   1d14a: 05 61        	addi	sp, sp, 32
   1d14c: 82 80        	ret

000000000001d14e <_realloc_r>:
   1d14e: 5d 71        	addi	sp, sp, -80
   1d150: 4a f8        	sd	s2, 48(sp)
   1d152: 86 e4        	sd	ra, 72(sp)
   1d154: a2 e0        	sd	s0, 64(sp)
   1d156: 26 fc        	sd	s1, 56(sp)
   1d158: 4e f4        	sd	s3, 40(sp)
   1d15a: 52 f0        	sd	s4, 32(sp)
   1d15c: 56 ec        	sd	s5, 24(sp)
   1d15e: 5a e8        	sd	s6, 16(sp)
   1d160: 5e e4        	sd	s7, 8(sp)
   1d162: 62 e0        	sd	s8, 0(sp)
   1d164: 32 89        	mv	s2, a2
   1d166: 63 89 05 14  	beqz	a1, 0x1d2b8 <_realloc_r+0x16a>
   1d16a: 2e 84        	mv	s0, a1
   1d16c: aa 89        	mv	s3, a0
   1d16e: ef b0 5f a9  	jal	0x18c02 <__malloc_lock>
   1d172: 83 37 84 ff  	ld	a5, -8(s0)
   1d176: 93 04 79 01  	addi	s1, s2, 23
   1d17a: 13 07 e0 02  	addi	a4, zero, 46
   1d17e: 93 0a 04 ff  	addi	s5, s0, -16
   1d182: 13 fa c7 ff  	andi	s4, a5, -4
   1d186: 63 7e 97 0a  	bgeu	a4, s1, 0x1d242 <_realloc_r+0xf4>
   1d18a: 37 07 00 80  	lui	a4, 524288
   1d18e: c1 98        	andi	s1, s1, -16
   1d190: 13 47 f7 ff  	not	a4, a4
   1d194: 63 6b 97 0a  	bltu	a4, s1, 0x1d24a <_realloc_r+0xfc>
   1d198: 63 e9 24 0b  	bltu	s1, s2, 0x1d24a <_realloc_r+0xfc>
   1d19c: 63 51 9a 0c  	bge	s4, s1, 0x1d25e <_realloc_r+0x110>
   1d1a0: 37 47 02 00  	lui	a4, 36
   1d1a4: 13 0c 07 91  	addi	s8, a4, -1776
   1d1a8: 03 36 0c 01  	ld	a2, 16(s8)
   1d1ac: b3 86 4a 01  	add	a3, s5, s4
   1d1b0: 98 66        	ld	a4, 8(a3)
   1d1b2: 63 02 d6 14  	beq	a2, a3, 0x1d2f6 <_realloc_r+0x1a8>
   1d1b6: 13 76 e7 ff  	andi	a2, a4, -2
   1d1ba: 36 96        	add	a2, a2, a3
   1d1bc: 10 66        	ld	a2, 8(a2)
   1d1be: 05 8a        	andi	a2, a2, 1
   1d1c0: 75 e2        	bnez	a2, 0x1d2a4 <_realloc_r+0x156>
   1d1c2: 71 9b        	andi	a4, a4, -4
   1d1c4: 33 06 ea 00  	add	a2, s4, a4
   1d1c8: 63 56 96 08  	bge	a2, s1, 0x1d254 <_realloc_r+0x106>
   1d1cc: 85 8b        	andi	a5, a5, 1
   1d1ce: 8d e3        	bnez	a5, 0x1d1f0 <_realloc_r+0xa2>
   1d1d0: 83 3b 04 ff  	ld	s7, -16(s0)
   1d1d4: b3 8b 7a 41  	sub	s7, s5, s7
   1d1d8: 83 b7 8b 00  	ld	a5, 8(s7)
   1d1dc: f1 9b        	andi	a5, a5, -4
   1d1de: 3e 97        	add	a4, a4, a5
   1d1e0: 33 0b 47 01  	add	s6, a4, s4
   1d1e4: 63 51 9b 24  	bge	s6, s1, 0x1d426 <_realloc_r+0x2d8>
   1d1e8: 33 0b fa 00  	add	s6, s4, a5
   1d1ec: 63 56 9b 1c  	bge	s6, s1, 0x1d3b8 <_realloc_r+0x26a>
   1d1f0: ca 85        	mv	a1, s2
   1d1f2: 4e 85        	mv	a0, s3
   1d1f4: ef b0 cf b3  	jal	0x18530 <_malloc_r>
   1d1f8: 2a 89        	mv	s2, a0
   1d1fa: 63 05 05 2e  	beqz	a0, 0x1d4e4 <_realloc_r+0x396>
   1d1fe: 83 37 84 ff  	ld	a5, -8(s0)
   1d202: 13 07 05 ff  	addi	a4, a0, -16
   1d206: f9 9b        	andi	a5, a5, -2
   1d208: d6 97        	add	a5, a5, s5
   1d20a: 63 82 e7 1a  	beq	a5, a4, 0x1d3ae <_realloc_r+0x260>
   1d20e: 13 06 8a ff  	addi	a2, s4, -8
   1d212: 93 07 80 04  	addi	a5, zero, 72
   1d216: 63 e4 c7 20  	bltu	a5, a2, 0x1d41e <_realloc_r+0x2d0>
   1d21a: 13 07 70 02  	addi	a4, zero, 39
   1d21e: 63 6b c7 16  	bltu	a4, a2, 0x1d394 <_realloc_r+0x246>
   1d222: aa 87        	mv	a5, a0
   1d224: 22 87        	mv	a4, s0
   1d226: 14 63        	ld	a3, 0(a4)
   1d228: 94 e3        	sd	a3, 0(a5)
   1d22a: 14 67        	ld	a3, 8(a4)
   1d22c: 94 e7        	sd	a3, 8(a5)
   1d22e: 18 6b        	ld	a4, 16(a4)
   1d230: 98 eb        	sd	a4, 16(a5)
   1d232: 4e 85        	mv	a0, s3
   1d234: a2 85        	mv	a1, s0
   1d236: ef 80 0f e3  	jal	0x15866 <_free_r>
   1d23a: 4e 85        	mv	a0, s3
   1d23c: ef b0 9f 9c  	jal	0x18c04 <__malloc_unlock>
   1d240: a9 a0        	j	0x1d28a <_realloc_r+0x13c>
   1d242: 93 04 00 02  	addi	s1, zero, 32
   1d246: e3 fb 24 f5  	bgeu	s1, s2, 0x1d19c <_realloc_r+0x4e>
   1d24a: b1 47        	addi	a5, zero, 12
   1d24c: 23 a0 f9 00  	sw	a5, 0(s3)
   1d250: 01 49        	mv	s2, zero
   1d252: 25 a8        	j	0x1d28a <_realloc_r+0x13c>
   1d254: 9c 6e        	ld	a5, 24(a3)
   1d256: 98 6a        	ld	a4, 16(a3)
   1d258: 32 8a        	mv	s4, a2
   1d25a: 1c ef        	sd	a5, 24(a4)
   1d25c: 98 eb        	sd	a4, 16(a5)
   1d25e: 83 b7 8a 00  	ld	a5, 8(s5)
   1d262: b3 06 9a 40  	sub	a3, s4, s1
   1d266: 7d 46        	addi	a2, zero, 31
   1d268: 85 8b        	andi	a5, a5, 1
   1d26a: 33 87 4a 01  	add	a4, s5, s4
   1d26e: 63 63 d6 06  	bltu	a2, a3, 0x1d2d4 <_realloc_r+0x186>
   1d272: b3 e7 47 01  	or	a5, a5, s4
   1d276: 23 b4 fa 00  	sd	a5, 8(s5)
   1d27a: 1c 67        	ld	a5, 8(a4)
   1d27c: 93 e7 17 00  	ori	a5, a5, 1
   1d280: 1c e7        	sd	a5, 8(a4)
   1d282: 4e 85        	mv	a0, s3
   1d284: ef b0 1f 98  	jal	0x18c04 <__malloc_unlock>
   1d288: 22 89        	mv	s2, s0
   1d28a: a6 60        	ld	ra, 72(sp)
   1d28c: 06 64        	ld	s0, 64(sp)
   1d28e: e2 74        	ld	s1, 56(sp)
   1d290: a2 79        	ld	s3, 40(sp)
   1d292: 02 7a        	ld	s4, 32(sp)
   1d294: e2 6a        	ld	s5, 24(sp)
   1d296: 42 6b        	ld	s6, 16(sp)
   1d298: a2 6b        	ld	s7, 8(sp)
   1d29a: 02 6c        	ld	s8, 0(sp)
   1d29c: 4a 85        	mv	a0, s2
   1d29e: 42 79        	ld	s2, 48(sp)
   1d2a0: 61 61        	addi	sp, sp, 80
   1d2a2: 82 80        	ret
   1d2a4: 85 8b        	andi	a5, a5, 1
   1d2a6: a9 f7        	bnez	a5, 0x1d1f0 <_realloc_r+0xa2>
   1d2a8: 83 3b 04 ff  	ld	s7, -16(s0)
   1d2ac: b3 8b 7a 41  	sub	s7, s5, s7
   1d2b0: 83 b7 8b 00  	ld	a5, 8(s7)
   1d2b4: f1 9b        	andi	a5, a5, -4
   1d2b6: 0d bf        	j	0x1d1e8 <_realloc_r+0x9a>
   1d2b8: 06 64        	ld	s0, 64(sp)
   1d2ba: a6 60        	ld	ra, 72(sp)
   1d2bc: e2 74        	ld	s1, 56(sp)
   1d2be: 42 79        	ld	s2, 48(sp)
   1d2c0: a2 79        	ld	s3, 40(sp)
   1d2c2: 02 7a        	ld	s4, 32(sp)
   1d2c4: e2 6a        	ld	s5, 24(sp)
   1d2c6: 42 6b        	ld	s6, 16(sp)
   1d2c8: a2 6b        	ld	s7, 8(sp)
   1d2ca: 02 6c        	ld	s8, 0(sp)
   1d2cc: b2 85        	mv	a1, a2
   1d2ce: 61 61        	addi	sp, sp, 80
   1d2d0: 6f b0 0f a6  	j	0x18530 <_malloc_r>
   1d2d4: c5 8f        	or	a5, a5, s1
   1d2d6: 23 b4 fa 00  	sd	a5, 8(s5)
   1d2da: b3 85 9a 00  	add	a1, s5, s1
   1d2de: 93 e6 16 00  	ori	a3, a3, 1
   1d2e2: 94 e5        	sd	a3, 8(a1)
   1d2e4: 1c 67        	ld	a5, 8(a4)
   1d2e6: c1 05        	addi	a1, a1, 16
   1d2e8: 4e 85        	mv	a0, s3
   1d2ea: 93 e7 17 00  	ori	a5, a5, 1
   1d2ee: 1c e7        	sd	a5, 8(a4)
   1d2f0: ef 80 6f d7  	jal	0x15866 <_free_r>
   1d2f4: 79 b7        	j	0x1d282 <_realloc_r+0x134>
   1d2f6: 71 9b        	andi	a4, a4, -4
   1d2f8: b3 06 47 01  	add	a3, a4, s4
   1d2fc: 13 86 04 02  	addi	a2, s1, 32
   1d300: 63 da c6 1a  	bge	a3, a2, 0x1d4b4 <_realloc_r+0x366>
   1d304: 85 8b        	andi	a5, a5, 1
   1d306: e3 95 07 ee  	bnez	a5, 0x1d1f0 <_realloc_r+0xa2>
   1d30a: 83 3b 04 ff  	ld	s7, -16(s0)
   1d30e: b3 8b 7a 41  	sub	s7, s5, s7
   1d312: 83 b7 8b 00  	ld	a5, 8(s7)
   1d316: f1 9b        	andi	a5, a5, -4
   1d318: 3e 97        	add	a4, a4, a5
   1d31a: 33 0b 47 01  	add	s6, a4, s4
   1d31e: e3 45 cb ec  	blt	s6, a2, 0x1d1e8 <_realloc_r+0x9a>
   1d322: 83 b7 8b 01  	ld	a5, 24(s7)
   1d326: 03 b7 0b 01  	ld	a4, 16(s7)
   1d32a: 13 06 8a ff  	addi	a2, s4, -8
   1d32e: 93 06 80 04  	addi	a3, zero, 72
   1d332: 1c ef        	sd	a5, 24(a4)
   1d334: 98 eb        	sd	a4, 16(a5)
   1d336: 13 89 0b 01  	addi	s2, s7, 16
   1d33a: 63 e3 c6 1c  	bltu	a3, a2, 0x1d500 <_realloc_r+0x3b2>
   1d33e: 13 07 70 02  	addi	a4, zero, 39
   1d342: ca 87        	mv	a5, s2
   1d344: 63 7f c7 00  	bgeu	a4, a2, 0x1d362 <_realloc_r+0x214>
   1d348: 18 60        	ld	a4, 0(s0)
   1d34a: 93 07 70 03  	addi	a5, zero, 55
   1d34e: 23 b8 eb 00  	sd	a4, 16(s7)
   1d352: 18 64        	ld	a4, 8(s0)
   1d354: 23 bc eb 00  	sd	a4, 24(s7)
   1d358: 63 e9 c7 1a  	bltu	a5, a2, 0x1d50a <_realloc_r+0x3bc>
   1d35c: 41 04        	addi	s0, s0, 16
   1d35e: 93 87 0b 02  	addi	a5, s7, 32
   1d362: 18 60        	ld	a4, 0(s0)
   1d364: 98 e3        	sd	a4, 0(a5)
   1d366: 18 64        	ld	a4, 8(s0)
   1d368: 98 e7        	sd	a4, 8(a5)
   1d36a: 18 68        	ld	a4, 16(s0)
   1d36c: 98 eb        	sd	a4, 16(a5)
   1d36e: 33 87 9b 00  	add	a4, s7, s1
   1d372: b3 07 9b 40  	sub	a5, s6, s1
   1d376: 23 38 ec 00  	sd	a4, 16(s8)
   1d37a: 93 e7 17 00  	ori	a5, a5, 1
   1d37e: 1c e7        	sd	a5, 8(a4)
   1d380: 83 b7 8b 00  	ld	a5, 8(s7)
   1d384: 4e 85        	mv	a0, s3
   1d386: 85 8b        	andi	a5, a5, 1
   1d388: c5 8f        	or	a5, a5, s1
   1d38a: 23 b4 fb 00  	sd	a5, 8(s7)
   1d38e: ef b0 7f 87  	jal	0x18c04 <__malloc_unlock>
   1d392: e5 bd        	j	0x1d28a <_realloc_r+0x13c>
   1d394: 14 60        	ld	a3, 0(s0)
   1d396: 13 07 70 03  	addi	a4, zero, 55
   1d39a: 14 e1        	sd	a3, 0(a0)
   1d39c: 14 64        	ld	a3, 8(s0)
   1d39e: 14 e5        	sd	a3, 8(a0)
   1d3a0: 63 6f c7 0e  	bltu	a4, a2, 0x1d49e <_realloc_r+0x350>
   1d3a4: 13 07 04 01  	addi	a4, s0, 16
   1d3a8: 93 07 05 01  	addi	a5, a0, 16
   1d3ac: ad bd        	j	0x1d226 <_realloc_r+0xd8>
   1d3ae: 83 37 85 ff  	ld	a5, -8(a0)
   1d3b2: f1 9b        	andi	a5, a5, -4
   1d3b4: 3e 9a        	add	s4, s4, a5
   1d3b6: 65 b5        	j	0x1d25e <_realloc_r+0x110>
   1d3b8: 83 b7 8b 01  	ld	a5, 24(s7)
   1d3bc: 03 b7 0b 01  	ld	a4, 16(s7)
   1d3c0: 13 06 8a ff  	addi	a2, s4, -8
   1d3c4: 93 06 80 04  	addi	a3, zero, 72
   1d3c8: 1c ef        	sd	a5, 24(a4)
   1d3ca: 98 eb        	sd	a4, 16(a5)
   1d3cc: 13 89 0b 01  	addi	s2, s7, 16
   1d3d0: 63 e2 c6 0c  	bltu	a3, a2, 0x1d494 <_realloc_r+0x346>
   1d3d4: 13 07 70 02  	addi	a4, zero, 39
   1d3d8: ca 87        	mv	a5, s2
   1d3da: 63 78 c7 02  	bgeu	a4, a2, 0x1d40a <_realloc_r+0x2bc>
   1d3de: 18 60        	ld	a4, 0(s0)
   1d3e0: 93 07 70 03  	addi	a5, zero, 55
   1d3e4: 23 b8 eb 00  	sd	a4, 16(s7)
   1d3e8: 18 64        	ld	a4, 8(s0)
   1d3ea: 23 bc eb 00  	sd	a4, 24(s7)
   1d3ee: 63 f7 c7 0e  	bgeu	a5, a2, 0x1d4dc <_realloc_r+0x38e>
   1d3f2: 1c 68        	ld	a5, 16(s0)
   1d3f4: 23 b0 fb 02  	sd	a5, 32(s7)
   1d3f8: 1c 6c        	ld	a5, 24(s0)
   1d3fa: 23 b4 fb 02  	sd	a5, 40(s7)
   1d3fe: 63 0f d6 06  	beq	a2, a3, 0x1d47c <_realloc_r+0x32e>
   1d402: 13 04 04 02  	addi	s0, s0, 32
   1d406: 93 87 0b 03  	addi	a5, s7, 48
   1d40a: 18 60        	ld	a4, 0(s0)
   1d40c: 98 e3        	sd	a4, 0(a5)
   1d40e: 18 64        	ld	a4, 8(s0)
   1d410: 98 e7        	sd	a4, 8(a5)
   1d412: 18 68        	ld	a4, 16(s0)
   1d414: 98 eb        	sd	a4, 16(a5)
   1d416: 4a 84        	mv	s0, s2
   1d418: 5a 8a        	mv	s4, s6
   1d41a: de 8a        	mv	s5, s7
   1d41c: 89 b5        	j	0x1d25e <_realloc_r+0x110>
   1d41e: a2 85        	mv	a1, s0
   1d420: ef f0 bf c1  	jal	0x1d03a <memmove>
   1d424: 39 b5        	j	0x1d232 <_realloc_r+0xe4>
   1d426: 9c 6e        	ld	a5, 24(a3)
   1d428: 98 6a        	ld	a4, 16(a3)
   1d42a: 13 06 8a ff  	addi	a2, s4, -8
   1d42e: 93 06 80 04  	addi	a3, zero, 72
   1d432: 1c ef        	sd	a5, 24(a4)
   1d434: 98 eb        	sd	a4, 16(a5)
   1d436: 03 b7 0b 01  	ld	a4, 16(s7)
   1d43a: 83 b7 8b 01  	ld	a5, 24(s7)
   1d43e: 13 89 0b 01  	addi	s2, s7, 16
   1d442: 1c ef        	sd	a5, 24(a4)
   1d444: 98 eb        	sd	a4, 16(a5)
   1d446: 63 e7 c6 04  	bltu	a3, a2, 0x1d494 <_realloc_r+0x346>
   1d44a: 13 07 70 02  	addi	a4, zero, 39
   1d44e: ca 87        	mv	a5, s2
   1d450: e3 7d c7 fa  	bgeu	a4, a2, 0x1d40a <_realloc_r+0x2bc>
   1d454: 18 60        	ld	a4, 0(s0)
   1d456: 93 07 70 03  	addi	a5, zero, 55
   1d45a: 23 b8 eb 00  	sd	a4, 16(s7)
   1d45e: 18 64        	ld	a4, 8(s0)
   1d460: 23 bc eb 00  	sd	a4, 24(s7)
   1d464: 63 fc c7 06  	bgeu	a5, a2, 0x1d4dc <_realloc_r+0x38e>
   1d468: 18 68        	ld	a4, 16(s0)
   1d46a: 93 07 80 04  	addi	a5, zero, 72
   1d46e: 23 b0 eb 02  	sd	a4, 32(s7)
   1d472: 18 6c        	ld	a4, 24(s0)
   1d474: 23 b4 eb 02  	sd	a4, 40(s7)
   1d478: e3 15 f6 f8  	bne	a2, a5, 0x1d402 <_realloc_r+0x2b4>
   1d47c: 18 70        	ld	a4, 32(s0)
   1d47e: 93 87 0b 04  	addi	a5, s7, 64
   1d482: 13 04 04 03  	addi	s0, s0, 48
   1d486: 23 b8 eb 02  	sd	a4, 48(s7)
   1d48a: 03 37 84 ff  	ld	a4, -8(s0)
   1d48e: 23 bc eb 02  	sd	a4, 56(s7)
   1d492: a5 bf        	j	0x1d40a <_realloc_r+0x2bc>
   1d494: a2 85        	mv	a1, s0
   1d496: 4a 85        	mv	a0, s2
   1d498: ef f0 3f ba  	jal	0x1d03a <memmove>
   1d49c: ad bf        	j	0x1d416 <_realloc_r+0x2c8>
   1d49e: 18 68        	ld	a4, 16(s0)
   1d4a0: 18 e9        	sd	a4, 16(a0)
   1d4a2: 18 6c        	ld	a4, 24(s0)
   1d4a4: 18 ed        	sd	a4, 24(a0)
   1d4a6: 63 04 f6 04  	beq	a2, a5, 0x1d4ee <_realloc_r+0x3a0>
   1d4aa: 13 07 04 02  	addi	a4, s0, 32
   1d4ae: 93 07 05 02  	addi	a5, a0, 32
   1d4b2: 95 bb        	j	0x1d226 <_realloc_r+0xd8>
   1d4b4: a6 9a        	add	s5, s5, s1
   1d4b6: b3 87 96 40  	sub	a5, a3, s1
   1d4ba: 23 38 5c 01  	sd	s5, 16(s8)
   1d4be: 93 e7 17 00  	ori	a5, a5, 1
   1d4c2: 23 b4 fa 00  	sd	a5, 8(s5)
   1d4c6: 83 37 84 ff  	ld	a5, -8(s0)
   1d4ca: 4e 85        	mv	a0, s3
   1d4cc: 22 89        	mv	s2, s0
   1d4ce: 85 8b        	andi	a5, a5, 1
   1d4d0: c5 8f        	or	a5, a5, s1
   1d4d2: 23 3c f4 fe  	sd	a5, -8(s0)
   1d4d6: ef b0 ef f2  	jal	0x18c04 <__malloc_unlock>
   1d4da: 45 bb        	j	0x1d28a <_realloc_r+0x13c>
   1d4dc: 41 04        	addi	s0, s0, 16
   1d4de: 93 87 0b 02  	addi	a5, s7, 32
   1d4e2: 25 b7        	j	0x1d40a <_realloc_r+0x2bc>
   1d4e4: 4e 85        	mv	a0, s3
   1d4e6: ef b0 ef f1  	jal	0x18c04 <__malloc_unlock>
   1d4ea: 01 49        	mv	s2, zero
   1d4ec: 79 bb        	j	0x1d28a <_realloc_r+0x13c>
   1d4ee: 14 70        	ld	a3, 32(s0)
   1d4f0: 13 07 04 03  	addi	a4, s0, 48
   1d4f4: 93 07 05 03  	addi	a5, a0, 48
   1d4f8: 14 f1        	sd	a3, 32(a0)
   1d4fa: 14 74        	ld	a3, 40(s0)
   1d4fc: 14 f5        	sd	a3, 40(a0)
   1d4fe: 25 b3        	j	0x1d226 <_realloc_r+0xd8>
   1d500: a2 85        	mv	a1, s0
   1d502: 4a 85        	mv	a0, s2
   1d504: ef f0 7f b3  	jal	0x1d03a <memmove>
   1d508: 9d b5        	j	0x1d36e <_realloc_r+0x220>
   1d50a: 1c 68        	ld	a5, 16(s0)
   1d50c: 23 b0 fb 02  	sd	a5, 32(s7)
   1d510: 1c 6c        	ld	a5, 24(s0)
   1d512: 23 b4 fb 02  	sd	a5, 40(s7)
   1d516: 63 07 d6 00  	beq	a2, a3, 0x1d524 <_realloc_r+0x3d6>
   1d51a: 13 04 04 02  	addi	s0, s0, 32
   1d51e: 93 87 0b 03  	addi	a5, s7, 48
   1d522: 81 b5        	j	0x1d362 <_realloc_r+0x214>
   1d524: 18 70        	ld	a4, 32(s0)
   1d526: 93 87 0b 04  	addi	a5, s7, 64
   1d52a: 13 04 04 03  	addi	s0, s0, 48
   1d52e: 23 b8 eb 02  	sd	a4, 48(s7)
   1d532: 03 37 84 ff  	ld	a4, -8(s0)
   1d536: 23 bc eb 02  	sd	a4, 56(s7)
   1d53a: 25 b5        	j	0x1d362 <_realloc_r+0x214>

000000000001d53c <cleanup_glue>:
   1d53c: 79 71        	addi	sp, sp, -48
   1d53e: 4a e8        	sd	s2, 16(sp)
   1d540: 03 b9 05 00  	ld	s2, 0(a1)
   1d544: 22 f0        	sd	s0, 32(sp)
   1d546: 26 ec        	sd	s1, 24(sp)
   1d548: 06 f4        	sd	ra, 40(sp)
   1d54a: 4e e4        	sd	s3, 8(sp)
   1d54c: 52 e0        	sd	s4, 0(sp)
   1d54e: 2e 84        	mv	s0, a1
   1d550: aa 84        	mv	s1, a0
   1d552: 63 0b 09 02  	beqz	s2, 0x1d588 <cleanup_glue+0x4c>
   1d556: 83 39 09 00  	ld	s3, 0(s2)
   1d55a: 63 83 09 02  	beqz	s3, 0x1d580 <cleanup_glue+0x44>
   1d55e: 03 ba 09 00  	ld	s4, 0(s3)
   1d562: 63 0b 0a 00  	beqz	s4, 0x1d578 <cleanup_glue+0x3c>
   1d566: 83 35 0a 00  	ld	a1, 0(s4)
   1d56a: 99 c1        	beqz	a1, 0x1d570 <cleanup_glue+0x34>
   1d56c: ef f0 1f fd  	jal	0x1d53c <cleanup_glue>
   1d570: d2 85        	mv	a1, s4
   1d572: 26 85        	mv	a0, s1
   1d574: ef 80 2f af  	jal	0x15866 <_free_r>
   1d578: ce 85        	mv	a1, s3
   1d57a: 26 85        	mv	a0, s1
   1d57c: ef 80 af ae  	jal	0x15866 <_free_r>
   1d580: ca 85        	mv	a1, s2
   1d582: 26 85        	mv	a0, s1
   1d584: ef 80 2f ae  	jal	0x15866 <_free_r>
   1d588: a2 85        	mv	a1, s0
   1d58a: 02 74        	ld	s0, 32(sp)
   1d58c: a2 70        	ld	ra, 40(sp)
   1d58e: 42 69        	ld	s2, 16(sp)
   1d590: a2 69        	ld	s3, 8(sp)
   1d592: 02 6a        	ld	s4, 0(sp)
   1d594: 26 85        	mv	a0, s1
   1d596: e2 64        	ld	s1, 24(sp)
   1d598: 45 61        	addi	sp, sp, 48
   1d59a: 6f 80 cf ac  	j	0x15866 <_free_r>

000000000001d59e <_reclaim_reent>:
   1d59e: 83 b7 81 93  	ld	a5, -1736(gp)
   1d5a2: 63 85 a7 0a  	beq	a5, a0, 0x1d64c <_reclaim_reent+0xae>
   1d5a6: 2c 7d        	ld	a1, 120(a0)
   1d5a8: 79 71        	addi	sp, sp, -48
   1d5aa: 26 ec        	sd	s1, 24(sp)
   1d5ac: 06 f4        	sd	ra, 40(sp)
   1d5ae: 22 f0        	sd	s0, 32(sp)
   1d5b0: 4a e8        	sd	s2, 16(sp)
   1d5b2: 4e e4        	sd	s3, 8(sp)
   1d5b4: aa 84        	mv	s1, a0
   1d5b6: 8d c5        	beqz	a1, 0x1d5e0 <_reclaim_reent+0x42>
   1d5b8: 01 49        	mv	s2, zero
   1d5ba: 93 09 00 20  	addi	s3, zero, 512
   1d5be: b3 87 25 01  	add	a5, a1, s2
   1d5c2: 80 63        	ld	s0, 0(a5)
   1d5c4: 01 c8        	beqz	s0, 0x1d5d4 <_reclaim_reent+0x36>
   1d5c6: a2 85        	mv	a1, s0
   1d5c8: 00 60        	ld	s0, 0(s0)
   1d5ca: 26 85        	mv	a0, s1
   1d5cc: ef 80 af a9  	jal	0x15866 <_free_r>
   1d5d0: 7d f8        	bnez	s0, 0x1d5c6 <_reclaim_reent+0x28>
   1d5d2: ac 7c        	ld	a1, 120(s1)
   1d5d4: 21 09        	addi	s2, s2, 8
   1d5d6: e3 14 39 ff  	bne	s2, s3, 0x1d5be <_reclaim_reent+0x20>
   1d5da: 26 85        	mv	a0, s1
   1d5dc: ef 80 af a8  	jal	0x15866 <_free_r>
   1d5e0: ac 70        	ld	a1, 96(s1)
   1d5e2: 81 c5        	beqz	a1, 0x1d5ea <_reclaim_reent+0x4c>
   1d5e4: 26 85        	mv	a0, s1
   1d5e6: ef 80 0f a8  	jal	0x15866 <_free_r>
   1d5ea: 03 b4 84 1f  	ld	s0, 504(s1)
   1d5ee: 01 cc        	beqz	s0, 0x1d606 <_reclaim_reent+0x68>
   1d5f0: 13 89 04 20  	addi	s2, s1, 512
   1d5f4: 63 09 24 01  	beq	s0, s2, 0x1d606 <_reclaim_reent+0x68>
   1d5f8: a2 85        	mv	a1, s0
   1d5fa: 00 60        	ld	s0, 0(s0)
   1d5fc: 26 85        	mv	a0, s1
   1d5fe: ef 80 8f a6  	jal	0x15866 <_free_r>
   1d602: e3 1b 89 fe  	bne	s2, s0, 0x1d5f8 <_reclaim_reent+0x5a>
   1d606: cc 64        	ld	a1, 136(s1)
   1d608: 81 c5        	beqz	a1, 0x1d610 <_reclaim_reent+0x72>
   1d60a: 26 85        	mv	a0, s1
   1d60c: ef 80 af a5  	jal	0x15866 <_free_r>
   1d610: bc 48        	lw	a5, 80(s1)
   1d612: 95 c7        	beqz	a5, 0x1d63e <_reclaim_reent+0xa0>
   1d614: bc 6c        	ld	a5, 88(s1)
   1d616: 26 85        	mv	a0, s1
   1d618: 82 97        	jalr	a5
   1d61a: 03 b4 04 52  	ld	s0, 1312(s1)
   1d61e: 05 c0        	beqz	s0, 0x1d63e <_reclaim_reent+0xa0>
   1d620: 0c 60        	ld	a1, 0(s0)
   1d622: 81 c5        	beqz	a1, 0x1d62a <_reclaim_reent+0x8c>
   1d624: 26 85        	mv	a0, s1
   1d626: ef f0 7f f1  	jal	0x1d53c <cleanup_glue>
   1d62a: a2 85        	mv	a1, s0
   1d62c: 02 74        	ld	s0, 32(sp)
   1d62e: a2 70        	ld	ra, 40(sp)
   1d630: 42 69        	ld	s2, 16(sp)
   1d632: a2 69        	ld	s3, 8(sp)
   1d634: 26 85        	mv	a0, s1
   1d636: e2 64        	ld	s1, 24(sp)
   1d638: 45 61        	addi	sp, sp, 48
   1d63a: 6f 80 cf a2  	j	0x15866 <_free_r>
   1d63e: a2 70        	ld	ra, 40(sp)
   1d640: 02 74        	ld	s0, 32(sp)
   1d642: e2 64        	ld	s1, 24(sp)
   1d644: 42 69        	ld	s2, 16(sp)
   1d646: a2 69        	ld	s3, 8(sp)
   1d648: 45 61        	addi	sp, sp, 48
   1d64a: 82 80        	ret
   1d64c: 82 80        	ret

000000000001d64e <__ssprint_r>:
   1d64e: 1c 6a        	ld	a5, 16(a2)
   1d650: 5d 71        	addi	sp, sp, -80
   1d652: 4e f4        	sd	s3, 40(sp)
   1d654: 56 ec        	sd	s5, 24(sp)
   1d656: 86 e4        	sd	ra, 72(sp)
   1d658: a2 e0        	sd	s0, 64(sp)
   1d65a: 26 fc        	sd	s1, 56(sp)
   1d65c: 4a f8        	sd	s2, 48(sp)
   1d65e: 52 f0        	sd	s4, 32(sp)
   1d660: 5a e8        	sd	s6, 16(sp)
   1d662: 5e e4        	sd	s7, 8(sp)
   1d664: 62 e0        	sd	s8, 0(sp)
   1d666: 83 39 06 00  	ld	s3, 0(a2)
   1d66a: b2 8a        	mv	s5, a2
   1d66c: 63 8d 07 10  	beqz	a5, 0x1d786 <__ssprint_r+0x138>
   1d670: d8 45        	lw	a4, 12(a1)
   1d672: 9c 61        	ld	a5, 0(a1)
   1d674: 2a 8b        	mv	s6, a0
   1d676: ae 84        	mv	s1, a1
   1d678: 01 4a        	mv	s4, zero
   1d67a: 01 44        	mv	s0, zero
   1d67c: 3a 89        	mv	s2, a4
   1d67e: 3e 85        	mv	a0, a5
   1d680: 29 c0        	beqz	s0, 0x1d6c2 <__ssprint_r+0x74>
   1d682: 63 76 e4 04  	bgeu	s0, a4, 0x1d6ce <__ssprint_r+0x80>
   1d686: 1b 07 04 00  	sext.w	a4, s0
   1d68a: 3a 89        	mv	s2, a4
   1d68c: ba 8b        	mv	s7, a4
   1d68e: 3e 85        	mv	a0, a5
   1d690: d2 85        	mv	a1, s4
   1d692: 4a 86        	mv	a2, s2
   1d694: ef f0 7f 9a  	jal	0x1d03a <memmove>
   1d698: d4 44        	lw	a3, 12(s1)
   1d69a: 9c 60        	ld	a5, 0(s1)
   1d69c: 03 b7 0a 01  	ld	a4, 16(s5)
   1d6a0: bb 86 26 41  	subw	a3, a3, s2
   1d6a4: ca 97        	add	a5, a5, s2
   1d6a6: d4 c4        	sw	a3, 12(s1)
   1d6a8: 9c e0        	sd	a5, 0(s1)
   1d6aa: 33 07 77 41  	sub	a4, a4, s7
   1d6ae: 23 b8 ea 00  	sd	a4, 16(s5)
   1d6b2: 5e 9a        	add	s4, s4, s7
   1d6b4: 33 04 74 41  	sub	s0, s0, s7
   1d6b8: 79 c7        	beqz	a4, 0x1d786 <__ssprint_r+0x138>
   1d6ba: d8 44        	lw	a4, 12(s1)
   1d6bc: 3e 85        	mv	a0, a5
   1d6be: 3a 89        	mv	s2, a4
   1d6c0: 69 f0        	bnez	s0, 0x1d682 <__ssprint_r+0x34>
   1d6c2: 03 ba 09 00  	ld	s4, 0(s3)
   1d6c6: 03 b4 89 00  	ld	s0, 8(s3)
   1d6ca: c1 09        	addi	s3, s3, 16
   1d6cc: 45 bf        	j	0x1d67c <__ssprint_r+0x2e>
   1d6ce: 03 97 04 01  	lh	a4, 16(s1)
   1d6d2: 9b 0b 04 00  	sext.w	s7, s0
   1d6d6: 93 76 07 48  	andi	a3, a4, 1152
   1d6da: dd da        	beqz	a3, 0x1d690 <__ssprint_r+0x42>
   1d6dc: 90 50        	lw	a2, 32(s1)
   1d6de: 8c 6c        	ld	a1, 24(s1)
   1d6e0: 9b 16 16 00  	slliw	a3, a2, 1
   1d6e4: b1 9e        	addw	a3, a3, a2
   1d6e6: 8d 8f        	sub	a5, a5, a1
   1d6e8: 1b d9 f6 01  	srliw	s2, a3, 31
   1d6ec: 9b 8b 07 00  	sext.w	s7, a5
   1d6f0: 3b 09 d9 00  	addw	s2, s2, a3
   1d6f4: 93 86 1b 00  	addi	a3, s7, 1
   1d6f8: 1b 59 19 40  	sraiw	s2, s2, 1
   1d6fc: a2 96        	add	a3, a3, s0
   1d6fe: 63 75 d9 00  	bgeu	s2, a3, 0x1d708 <__ssprint_r+0xba>
   1d702: 85 27        	addiw	a5, a5, 1
   1d704: 3b 89 87 00  	addw	s2, a5, s0
   1d708: 13 77 07 40  	andi	a4, a4, 1024
   1d70c: 31 c7        	beqz	a4, 0x1d758 <__ssprint_r+0x10a>
   1d70e: ca 85        	mv	a1, s2
   1d710: 5a 85        	mv	a0, s6
   1d712: ef a0 ff e1  	jal	0x18530 <_malloc_r>
   1d716: 2a 8c        	mv	s8, a0
   1d718: 31 c9        	beqz	a0, 0x1d76c <__ssprint_r+0x11e>
   1d71a: 8c 6c        	ld	a1, 24(s1)
   1d71c: 5e 86        	mv	a2, s7
   1d71e: ef f0 7f 81  	jal	0x1cf34 <memcpy>
   1d722: 83 d7 04 01  	lhu	a5, 16(s1)
   1d726: 93 f7 f7 b7  	andi	a5, a5, -1153
   1d72a: 93 e7 07 08  	ori	a5, a5, 128
   1d72e: 23 98 f4 00  	sh	a5, 16(s1)
   1d732: 33 05 7c 01  	add	a0, s8, s7
   1d736: bb 0b 79 41  	subw	s7, s2, s7
   1d73a: 1b 07 04 00  	sext.w	a4, s0
   1d73e: 23 a0 24 03  	sw	s2, 32(s1)
   1d742: 23 a6 74 01  	sw	s7, 12(s1)
   1d746: 23 bc 84 01  	sd	s8, 24(s1)
   1d74a: 88 e0        	sd	a0, 0(s1)
   1d74c: 3a 89        	mv	s2, a4
   1d74e: ba 8b        	mv	s7, a4
   1d750: aa 87        	mv	a5, a0
   1d752: e3 7f e4 f2  	bgeu	s0, a4, 0x1d690 <__ssprint_r+0x42>
   1d756: 15 bf        	j	0x1d68a <__ssprint_r+0x3c>
   1d758: 4a 86        	mv	a2, s2
   1d75a: 5a 85        	mv	a0, s6
   1d75c: ef f0 3f 9f  	jal	0x1d14e <_realloc_r>
   1d760: 2a 8c        	mv	s8, a0
   1d762: 61 f9        	bnez	a0, 0x1d732 <__ssprint_r+0xe4>
   1d764: 8c 6c        	ld	a1, 24(s1)
   1d766: 5a 85        	mv	a0, s6
   1d768: ef 80 ef 8f  	jal	0x15866 <_free_r>
   1d76c: 83 d7 04 01  	lhu	a5, 16(s1)
   1d770: 31 47        	addi	a4, zero, 12
   1d772: 23 20 eb 00  	sw	a4, 0(s6)
   1d776: 93 e7 07 04  	ori	a5, a5, 64
   1d77a: 23 98 f4 00  	sh	a5, 16(s1)
   1d77e: 23 b8 0a 00  	sd	zero, 16(s5)
   1d782: 7d 55        	addi	a0, zero, -1
   1d784: 11 a0        	j	0x1d788 <__ssprint_r+0x13a>
   1d786: 01 45        	mv	a0, zero
   1d788: a6 60        	ld	ra, 72(sp)
   1d78a: 06 64        	ld	s0, 64(sp)
   1d78c: 23 a4 0a 00  	sw	zero, 8(s5)
   1d790: e2 74        	ld	s1, 56(sp)
   1d792: 42 79        	ld	s2, 48(sp)
   1d794: a2 79        	ld	s3, 40(sp)
   1d796: 02 7a        	ld	s4, 32(sp)
   1d798: e2 6a        	ld	s5, 24(sp)
   1d79a: 42 6b        	ld	s6, 16(sp)
   1d79c: a2 6b        	ld	s7, 8(sp)
   1d79e: 02 6c        	ld	s8, 0(sp)
   1d7a0: 61 61        	addi	sp, sp, 80
   1d7a2: 82 80        	ret

000000000001d7a4 <_svfiprintf_r>:
   1d7a4: 03 d7 05 01  	lhu	a4, 16(a1)
   1d7a8: 05 71        	addi	sp, sp, -480
   1d7aa: 52 fb        	sd	s4, 432(sp)
   1d7ac: 5a f3        	sd	s6, 416(sp)
   1d7ae: 62 eb        	sd	s8, 400(sp)
   1d7b0: ee fe        	sd	s11, 376(sp)
   1d7b2: 86 ef        	sd	ra, 472(sp)
   1d7b4: a2 eb        	sd	s0, 464(sp)
   1d7b6: a6 e7        	sd	s1, 456(sp)
   1d7b8: ca e3        	sd	s2, 448(sp)
   1d7ba: 4e ff        	sd	s3, 440(sp)
   1d7bc: 56 f7        	sd	s5, 424(sp)
   1d7be: 5e ef        	sd	s7, 408(sp)
   1d7c0: 66 e7        	sd	s9, 392(sp)
   1d7c2: 6a e3        	sd	s10, 384(sp)
   1d7c4: 13 77 07 08  	andi	a4, a4, 128
   1d7c8: 2e 8a        	mv	s4, a1
   1d7ca: 2a 8b        	mv	s6, a0
   1d7cc: b2 8d        	mv	s11, a2
   1d7ce: 36 8c        	mv	s8, a3
   1d7d0: 01 c7        	beqz	a4, 0x1d7d8 <_svfiprintf_r+0x34>
   1d7d2: 98 6d        	ld	a4, 24(a1)
   1d7d4: e3 0d 07 20  	beqz	a4, 0x1e1ee <_svfiprintf_r+0xa4a>
   1d7d8: b7 37 02 00  	lui	a5, 35
   1d7dc: 93 0a 01 0f  	addi	s5, sp, 240
   1d7e0: 81 4c        	mv	s9, zero
   1d7e2: 93 87 47 83  	addi	a5, a5, -1996
   1d7e6: b7 3b 02 00  	lui	s7, 35
   1d7ea: d6 f8        	sd	s5, 112(sp)
   1d7ec: 02 e1        	sd	zero, 128(sp)
   1d7ee: 82 dc        	sw	zero, 120(sp)
   1d7f0: 81 44        	mv	s1, zero
   1d7f2: 56 84        	mv	s0, s5
   1d7f4: 02 e8        	sd	zero, 16(sp)
   1d7f6: 02 fc        	sd	zero, 56(sp)
   1d7f8: 02 f8        	sd	zero, 48(sp)
   1d7fa: 82 e0        	sd	zero, 64(sp)
   1d7fc: 3e e4        	sd	a5, 8(sp)
   1d7fe: c1 49        	addi	s3, zero, 16
   1d800: 93 8b 0b 9a  	addi	s7, s7, -1632
   1d804: 66 8d        	mv	s10, s9
   1d806: e2 86        	mv	a3, s8
   1d808: 83 c7 0d 00  	lbu	a5, 0(s11)
   1d80c: 63 8a 07 16  	beqz	a5, 0x1d980 <_svfiprintf_r+0x1dc>
   1d810: 6e 89        	mv	s2, s11
   1d812: 13 07 50 02  	addi	a4, zero, 37
   1d816: 63 8b e7 1c  	beq	a5, a4, 0x1d9ec <_svfiprintf_r+0x248>
   1d81a: 83 47 19 00  	lbu	a5, 1(s2)
   1d81e: 05 09        	addi	s2, s2, 1
   1d820: fd fb        	bnez	a5, 0x1d816 <_svfiprintf_r+0x72>
   1d822: 3b 0c b9 41  	subw	s8, s2, s11
   1d826: 63 0d 0c 14  	beqz	s8, 0x1d980 <_svfiprintf_r+0x1dc>
   1d82a: e6 57        	lw	a5, 120(sp)
   1d82c: e2 94        	add	s1, s1, s8
   1d82e: 23 30 b4 01  	sd	s11, 0(s0)
   1d832: 1b 87 17 00  	addiw	a4, a5, 1
   1d836: 23 34 84 01  	sd	s8, 8(s0)
   1d83a: 26 e1        	sd	s1, 128(sp)
   1d83c: ba dc        	sw	a4, 120(sp)
   1d83e: 9d 47        	addi	a5, zero, 7
   1d840: 41 04        	addi	s0, s0, 16
   1d842: 63 cc e7 1a  	blt	a5, a4, 0x1d9fa <_svfiprintf_r+0x256>
   1d846: 83 47 09 00  	lbu	a5, 0(s2)
   1d84a: 3b 0d 8d 01  	addw	s10, s10, s8
   1d84e: 63 89 07 12  	beqz	a5, 0x1d980 <_svfiprintf_r+0x1dc>
   1d852: 83 47 19 00  	lbu	a5, 1(s2)
   1d856: 93 0d 19 00  	addi	s11, s2, 1
   1d85a: a3 03 01 06  	sb	zero, 103(sp)
   1d85e: 7d 5c        	addi	s8, zero, -1
   1d860: 81 48        	mv	a7, zero
   1d862: 81 4c        	mv	s9, zero
   1d864: 13 09 a0 05  	addi	s2, zero, 90
   1d868: 85 0d        	addi	s11, s11, 1
   1d86a: 81 27        	sext.w	a5, a5
   1d86c: 1b 87 07 fe  	addiw	a4, a5, -32
   1d870: 1b 06 07 00  	sext.w	a2, a4
   1d874: 63 65 c9 04  	bltu	s2, a2, 0x1d8be <_svfiprintf_r+0x11a>
   1d878: 13 16 07 02  	slli	a2, a4, 32
   1d87c: 13 57 e6 01  	srli	a4, a2, 30
   1d880: 22 66        	ld	a2, 8(sp)
   1d882: 32 97        	add	a4, a4, a2
   1d884: 18 43        	lw	a4, 0(a4)
   1d886: 02 87        	jr	a4
   1d888: 81 48        	mv	a7, zero
   1d88a: a5 45        	addi	a1, zero, 9
   1d88c: 03 c6 0d 00  	lbu	a2, 0(s11)
   1d890: 1b 97 28 00  	slliw	a4, a7, 2
   1d894: bb 08 17 01  	addw	a7, a4, a7
   1d898: 9b 87 07 fd  	addiw	a5, a5, -48
   1d89c: 9b 98 18 00  	slliw	a7, a7, 1
   1d8a0: 1b 07 06 fd  	addiw	a4, a2, -48
   1d8a4: bb 88 17 01  	addw	a7, a5, a7
   1d8a8: 85 0d        	addi	s11, s11, 1
   1d8aa: 9b 07 06 00  	sext.w	a5, a2
   1d8ae: e3 ff e5 fc  	bgeu	a1, a4, 0x1d88c <_svfiprintf_r+0xe8>
   1d8b2: 1b 87 07 fe  	addiw	a4, a5, -32
   1d8b6: 1b 06 07 00  	sext.w	a2, a4
   1d8ba: e3 7f c9 fa  	bgeu	s2, a2, 0x1d878 <_svfiprintf_r+0xd4>
   1d8be: e9 c3        	beqz	a5, 0x1d980 <_svfiprintf_r+0x1dc>
   1d8c0: 23 04 f1 08  	sb	a5, 136(sp)
   1d8c4: a3 03 01 06  	sb	zero, 103(sp)
   1d8c8: 05 4e        	addi	t3, zero, 1
   1d8ca: 05 43        	addi	t1, zero, 1
   1d8cc: 13 09 81 08  	addi	s2, sp, 136
   1d8d0: 01 4c        	mv	s8, zero
   1d8d2: 93 f2 2c 00  	andi	t0, s9, 2
   1d8d6: 63 83 02 00  	beqz	t0, 0x1d8dc <_svfiprintf_r+0x138>
   1d8da: 09 2e        	addiw	t3, t3, 2
   1d8dc: 93 ff 4c 08  	andi	t6, s9, 132
   1d8e0: e6 57        	lw	a5, 120(sp)
   1d8e2: 63 96 0f 00  	bnez	t6, 0x1d8ee <_svfiprintf_r+0x14a>
   1d8e6: 3b 87 c8 41  	subw	a4, a7, t3
   1d8ea: 63 49 e0 5e  	bgtz	a4, 0x1dedc <_svfiprintf_r+0x738>
   1d8ee: 03 47 71 06  	lbu	a4, 103(sp)
   1d8f2: 11 cf        	beqz	a4, 0x1d90e <_svfiprintf_r+0x16a>
   1d8f4: 13 06 71 06  	addi	a2, sp, 103
   1d8f8: 10 e0        	sd	a2, 0(s0)
   1d8fa: 85 27        	addiw	a5, a5, 1
   1d8fc: 85 04        	addi	s1, s1, 1
   1d8fe: 05 46        	addi	a2, zero, 1
   1d900: 10 e4        	sd	a2, 8(s0)
   1d902: 26 e1        	sd	s1, 128(sp)
   1d904: be dc        	sw	a5, 120(sp)
   1d906: 1d 47        	addi	a4, zero, 7
   1d908: 41 04        	addi	s0, s0, 16
   1d90a: 63 4a f7 12  	blt	a4, a5, 0x1da3e <_svfiprintf_r+0x29a>
   1d90e: 63 8e 02 00  	beqz	t0, 0x1d92a <_svfiprintf_r+0x186>
   1d912: b0 10        	addi	a2, sp, 104
   1d914: 10 e0        	sd	a2, 0(s0)
   1d916: 85 27        	addiw	a5, a5, 1
   1d918: 89 04        	addi	s1, s1, 2
   1d91a: 09 46        	addi	a2, zero, 2
   1d91c: 10 e4        	sd	a2, 8(s0)
   1d91e: 26 e1        	sd	s1, 128(sp)
   1d920: be dc        	sw	a5, 120(sp)
   1d922: 1d 47        	addi	a4, zero, 7
   1d924: 41 04        	addi	s0, s0, 16
   1d926: 63 47 f7 66  	blt	a4, a5, 0x1df94 <_svfiprintf_r+0x7f0>
   1d92a: 13 07 00 08  	addi	a4, zero, 128
   1d92e: 63 81 ef 3e  	beq	t6, a4, 0x1dd10 <_svfiprintf_r+0x56c>
   1d932: 3b 0c 6c 40  	subw	s8, s8, t1
   1d936: 63 48 80 49  	bgtz	s8, 0x1ddc6 <_svfiprintf_r+0x622>
   1d93a: 1b 87 17 00  	addiw	a4, a5, 1
   1d93e: 9a 94        	add	s1, s1, t1
   1d940: 23 30 24 01  	sd	s2, 0(s0)
   1d944: 23 34 64 00  	sd	t1, 8(s0)
   1d948: 26 e1        	sd	s1, 128(sp)
   1d94a: ba dc        	sw	a4, 120(sp)
   1d94c: 9d 47        	addi	a5, zero, 7
   1d94e: 41 04        	addi	s0, s0, 16
   1d950: 63 c0 e7 52  	blt	a5, a4, 0x1de70 <_svfiprintf_r+0x6cc>
   1d954: 93 fc 4c 00  	andi	s9, s9, 4
   1d958: 63 86 0c 00  	beqz	s9, 0x1d964 <_svfiprintf_r+0x1c0>
   1d95c: 3b 89 c8 41  	subw	s2, a7, t3
   1d960: 63 4f 20 65  	bgtz	s2, 0x1dfbe <_svfiprintf_r+0x81a>
   1d964: c6 87        	mv	a5, a7
   1d966: 63 d3 c8 01  	bge	a7, t3, 0x1d96c <_svfiprintf_r+0x1c8>
   1d96a: f2 87        	mv	a5, t3
   1d96c: 3b 8d a7 01  	addw	s10, a5, s10
   1d970: 63 90 04 52  	bnez	s1, 0x1de90 <_svfiprintf_r+0x6ec>
   1d974: 83 c7 0d 00  	lbu	a5, 0(s11)
   1d978: 82 dc        	sw	zero, 120(sp)
   1d97a: 56 84        	mv	s0, s5
   1d97c: e3 9a 07 e8  	bnez	a5, 0x1d810 <_svfiprintf_r+0x6c>
   1d980: ea 8c        	mv	s9, s10
   1d982: c1 c8        	beqz	s1, 0x1da12 <_svfiprintf_r+0x26e>
   1d984: 90 18        	addi	a2, sp, 112
   1d986: d2 85        	mv	a1, s4
   1d988: 5a 85        	mv	a0, s6
   1d98a: ef f0 5f cc  	jal	0x1d64e <__ssprint_r>
   1d98e: 51 a0        	j	0x1da12 <_svfiprintf_r+0x26e>
   1d990: 93 ec 0c 01  	ori	s9, s9, 16
   1d994: 83 c7 0d 00  	lbu	a5, 0(s11)
   1d998: 81 2c        	sext.w	s9, s9
   1d99a: f9 b5        	j	0x1d868 <_svfiprintf_r+0xc4>
   1d99c: 93 ec 0c 01  	ori	s9, s9, 16
   1d9a0: 81 2c        	sext.w	s9, s9
   1d9a2: 93 f7 0c 02  	andi	a5, s9, 32
   1d9a6: 93 8e 86 00  	addi	t4, a3, 8
   1d9aa: e1 c3        	beqz	a5, 0x1da6a <_svfiprintf_r+0x2c6>
   1d9ac: 98 62        	ld	a4, 0(a3)
   1d9ae: ba 87        	mv	a5, a4
   1d9b0: 63 4c 07 0c  	bltz	a4, 0x1da88 <_svfiprintf_r+0x2e4>
   1d9b4: 7d 57        	addi	a4, zero, -1
   1d9b6: 63 05 ec 0e  	beq	s8, a4, 0x1daa0 <_svfiprintf_r+0x2fc>
   1d9ba: 93 fc fc f7  	andi	s9, s9, -129
   1d9be: 81 2c        	sext.w	s9, s9
   1d9c0: e5 e3        	bnez	a5, 0x1daa0 <_svfiprintf_r+0x2fc>
   1d9c2: e3 13 0c 14  	bnez	s8, 0x1e308 <_svfiprintf_r+0xb64>
   1d9c6: 01 4e        	mv	t3, zero
   1d9c8: 01 43        	mv	t1, zero
   1d9ca: f6 86        	mv	a3, t4
   1d9cc: 13 09 c1 0e  	addi	s2, sp, 236
   1d9d0: d5 a8        	j	0x1dac4 <_svfiprintf_r+0x320>
   1d9d2: 83 a8 06 00  	lw	a7, 0(a3)
   1d9d6: a1 06        	addi	a3, a3, 8
   1d9d8: 63 dd 08 2a  	bgez	a7, 0x1dc92 <_svfiprintf_r+0x4ee>
   1d9dc: bb 08 10 41  	negw	a7, a7
   1d9e0: 93 ec 4c 00  	ori	s9, s9, 4
   1d9e4: 83 c7 0d 00  	lbu	a5, 0(s11)
   1d9e8: 81 2c        	sext.w	s9, s9
   1d9ea: bd bd        	j	0x1d868 <_svfiprintf_r+0xc4>
   1d9ec: 3b 0c b9 41  	subw	s8, s2, s11
   1d9f0: e3 1d 0c e2  	bnez	s8, 0x1d82a <_svfiprintf_r+0x86>
   1d9f4: 83 47 09 00  	lbu	a5, 0(s2)
   1d9f8: 99 bd        	j	0x1d84e <_svfiprintf_r+0xaa>
   1d9fa: 90 18        	addi	a2, sp, 112
   1d9fc: d2 85        	mv	a1, s4
   1d9fe: 5a 85        	mv	a0, s6
   1da00: 36 ec        	sd	a3, 24(sp)
   1da02: ef f0 df c4  	jal	0x1d64e <__ssprint_r>
   1da06: 09 e5        	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1da08: 8a 64        	ld	s1, 128(sp)
   1da0a: e2 66        	ld	a3, 24(sp)
   1da0c: 56 84        	mv	s0, s5
   1da0e: 25 bd        	j	0x1d846 <_svfiprintf_r+0xa2>
   1da10: ea 8c        	mv	s9, s10
   1da12: 83 57 0a 01  	lhu	a5, 16(s4)
   1da16: 93 f7 07 04  	andi	a5, a5, 64
   1da1a: e3 94 07 10  	bnez	a5, 0x1e322 <_svfiprintf_r+0xb7e>
   1da1e: fe 60        	ld	ra, 472(sp)
   1da20: 5e 64        	ld	s0, 464(sp)
   1da22: be 64        	ld	s1, 456(sp)
   1da24: 1e 69        	ld	s2, 448(sp)
   1da26: fa 79        	ld	s3, 440(sp)
   1da28: 5a 7a        	ld	s4, 432(sp)
   1da2a: ba 7a        	ld	s5, 424(sp)
   1da2c: 1a 7b        	ld	s6, 416(sp)
   1da2e: fa 6b        	ld	s7, 408(sp)
   1da30: 5a 6c        	ld	s8, 400(sp)
   1da32: 1a 6d        	ld	s10, 384(sp)
   1da34: f6 7d        	ld	s11, 376(sp)
   1da36: 66 85        	mv	a0, s9
   1da38: ba 6c        	ld	s9, 392(sp)
   1da3a: 3d 61        	addi	sp, sp, 480
   1da3c: 82 80        	ret
   1da3e: 90 18        	addi	a2, sp, 112
   1da40: d2 85        	mv	a1, s4
   1da42: 5a 85        	mv	a0, s6
   1da44: b6 ec        	sd	a3, 88(sp)
   1da46: 9a e8        	sd	t1, 80(sp)
   1da48: f2 e4        	sd	t3, 72(sp)
   1da4a: 46 f4        	sd	a7, 40(sp)
   1da4c: 7e f0        	sd	t6, 32(sp)
   1da4e: 16 ec        	sd	t0, 24(sp)
   1da50: ef f0 ff bf  	jal	0x1d64e <__ssprint_r>
   1da54: 55 fd        	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1da56: 8a 64        	ld	s1, 128(sp)
   1da58: e6 57        	lw	a5, 120(sp)
   1da5a: e6 66        	ld	a3, 88(sp)
   1da5c: 46 63        	ld	t1, 80(sp)
   1da5e: 26 6e        	ld	t3, 72(sp)
   1da60: a2 78        	ld	a7, 40(sp)
   1da62: 82 7f        	ld	t6, 32(sp)
   1da64: e2 62        	ld	t0, 24(sp)
   1da66: 56 84        	mv	s0, s5
   1da68: 5d b5        	j	0x1d90e <_svfiprintf_r+0x16a>
   1da6a: 93 f7 0c 01  	andi	a5, s9, 16
   1da6e: 9d ff        	bnez	a5, 0x1d9ac <_svfiprintf_r+0x208>
   1da70: 13 f7 0c 04  	andi	a4, s9, 64
   1da74: 9c 42        	lw	a5, 0(a3)
   1da76: e3 05 07 04  	beqz	a4, 0x1e2c0 <_svfiprintf_r+0xb1c>
   1da7a: 9b 97 07 01  	slliw	a5, a5, 16
   1da7e: 9b d7 07 41  	sraiw	a5, a5, 16
   1da82: 3e 87        	mv	a4, a5
   1da84: e3 58 07 f2  	bgez	a4, 0x1d9b4 <_svfiprintf_r+0x210>
   1da88: 13 07 d0 02  	addi	a4, zero, 45
   1da8c: a3 03 e1 06  	sb	a4, 103(sp)
   1da90: 7d 57        	addi	a4, zero, -1
   1da92: b3 07 f0 40  	neg	a5, a5
   1da96: 63 05 ec 00  	beq	s8, a4, 0x1daa0 <_svfiprintf_r+0x2fc>
   1da9a: 93 fc fc f7  	andi	s9, s9, -129
   1da9e: 81 2c        	sext.w	s9, s9
   1daa0: 25 47        	addi	a4, zero, 9
   1daa2: 63 6a f7 60  	bltu	a4, a5, 0x1e0b6 <_svfiprintf_r+0x912>
   1daa6: 9b 87 07 03  	addiw	a5, a5, 48
   1daaa: 93 f7 f7 0f  	andi	a5, a5, 255
   1daae: f6 86        	mv	a3, t4
   1dab0: a3 05 f1 0e  	sb	a5, 235(sp)
   1dab4: 1b 0e 0c 00  	sext.w	t3, s8
   1dab8: 63 43 80 01  	bgtz	s8, 0x1dabe <_svfiprintf_r+0x31a>
   1dabc: 05 4e        	addi	t3, zero, 1
   1dabe: 05 43        	addi	t1, zero, 1
   1dac0: 13 09 b1 0e  	addi	s2, sp, 235
   1dac4: 83 47 71 06  	lbu	a5, 103(sp)
   1dac8: e3 85 07 e0  	beqz	a5, 0x1d8d2 <_svfiprintf_r+0x12e>
   1dacc: 05 2e        	addiw	t3, t3, 1
   1dace: 11 b5        	j	0x1d8d2 <_svfiprintf_r+0x12e>
   1dad0: 03 b9 06 00  	ld	s2, 0(a3)
   1dad4: a3 03 01 06  	sb	zero, 103(sp)
   1dad8: a1 06        	addi	a3, a3, 8
   1dada: 63 02 09 7a  	beqz	s2, 0x1e27e <_svfiprintf_r+0xada>
   1dade: fd 57        	addi	a5, zero, -1
   1dae0: 36 f0        	sd	a3, 32(sp)
   1dae2: 46 ec        	sd	a7, 24(sp)
   1dae4: 63 02 fc 6e  	beq	s8, a5, 0x1e1c8 <_svfiprintf_r+0xa24>
   1dae8: 62 86        	mv	a2, s8
   1daea: 81 45        	mv	a1, zero
   1daec: 4a 85        	mv	a0, s2
   1daee: ef b0 ef 87  	jal	0x18b6c <memchr>
   1daf2: e2 68        	ld	a7, 24(sp)
   1daf4: 82 76        	ld	a3, 32(sp)
   1daf6: e3 04 05 00  	beqz	a0, 0x1e2fe <_svfiprintf_r+0xb5a>
   1dafa: 3b 03 25 41  	subw	t1, a0, s2
   1dafe: 13 4e f3 ff  	not	t3, t1
   1db02: 13 5e fe 43  	srai	t3, t3, 63
   1db06: 33 7e 6e 00  	and	t3, t3, t1
   1db0a: 01 4c        	mv	s8, zero
   1db0c: 65 bf        	j	0x1dac4 <_svfiprintf_r+0x320>
   1db0e: 9c 42        	lw	a5, 0(a3)
   1db10: a3 03 01 06  	sb	zero, 103(sp)
   1db14: a1 06        	addi	a3, a3, 8
   1db16: 23 04 f1 08  	sb	a5, 136(sp)
   1db1a: 05 4e        	addi	t3, zero, 1
   1db1c: 05 43        	addi	t1, zero, 1
   1db1e: 13 09 81 08  	addi	s2, sp, 136
   1db22: 7d b3        	j	0x1d8d0 <_svfiprintf_r+0x12c>
   1db24: 93 ec 0c 02  	ori	s9, s9, 32
   1db28: 83 c7 0d 00  	lbu	a5, 0(s11)
   1db2c: 81 2c        	sext.w	s9, s9
   1db2e: 2d bb        	j	0x1d868 <_svfiprintf_r+0xc4>
   1db30: e1 77        	lui	a5, 1048568
   1db32: 93 c7 07 83  	xori	a5, a5, -2000
   1db36: 23 14 f1 06  	sh	a5, 104(sp)
   1db3a: b7 27 02 00  	lui	a5, 34
   1db3e: 93 ec 2c 00  	ori	s9, s9, 2
   1db42: 93 87 87 c0  	addi	a5, a5, -1016
   1db46: 98 62        	ld	a4, 0(a3)
   1db48: 81 2c        	sext.w	s9, s9
   1db4a: a1 06        	addi	a3, a3, 8
   1db4c: 3e e8        	sd	a5, 16(sp)
   1db4e: a3 03 01 06  	sb	zero, 103(sp)
   1db52: fd 57        	addi	a5, zero, -1
   1db54: 63 0d fc 34  	beq	s8, a5, 0x1deae <_svfiprintf_r+0x70a>
   1db58: 93 f7 fc f7  	andi	a5, s9, -129
   1db5c: 81 27        	sext.w	a5, a5
   1db5e: 63 14 07 34  	bnez	a4, 0x1dea6 <_svfiprintf_r+0x702>
   1db62: 66 83        	mv	t1, s9
   1db64: 09 47        	addi	a4, zero, 2
   1db66: be 8c        	mv	s9, a5
   1db68: 63 1f 0c 04  	bnez	s8, 0x1dbc6 <_svfiprintf_r+0x422>
   1db6c: 63 10 07 54  	bnez	a4, 0x1e0ac <_svfiprintf_r+0x908>
   1db70: 13 73 13 00  	andi	t1, t1, 1
   1db74: 63 0f 03 58  	beqz	t1, 0x1e112 <_svfiprintf_r+0x96e>
   1db78: 93 07 00 03  	addi	a5, zero, 48
   1db7c: a3 05 f1 0e  	sb	a5, 235(sp)
   1db80: 1a 8e        	mv	t3, t1
   1db82: 13 09 b1 0e  	addi	s2, sp, 235
   1db86: 3d bf        	j	0x1dac4 <_svfiprintf_r+0x320>
   1db88: 93 e7 0c 01  	ori	a5, s9, 16
   1db8c: 81 27        	sext.w	a5, a5
   1db8e: 13 f7 07 02  	andi	a4, a5, 32
   1db92: 13 86 86 00  	addi	a2, a3, 8
   1db96: 09 e7        	bnez	a4, 0x1dba0 <_svfiprintf_r+0x3fc>
   1db98: 13 f7 07 01  	andi	a4, a5, 16
   1db9c: 63 04 07 5a  	beqz	a4, 0x1e144 <_svfiprintf_r+0x9a0>
   1dba0: 98 62        	ld	a4, 0(a3)
   1dba2: 93 fc f7 bf  	andi	s9, a5, -1025
   1dba6: a3 03 01 06  	sb	zero, 103(sp)
   1dbaa: fd 56        	addi	a3, zero, -1
   1dbac: 81 2c        	sext.w	s9, s9
   1dbae: 63 06 dc 4a  	beq	s8, a3, 0x1e05a <_svfiprintf_r+0x8b6>
   1dbb2: 93 f7 f7 b7  	andi	a5, a5, -1153
   1dbb6: 81 27        	sext.w	a5, a5
   1dbb8: 63 1d 07 68  	bnez	a4, 0x1e252 <_svfiprintf_r+0xaae>
   1dbbc: 66 83        	mv	t1, s9
   1dbbe: b2 86        	mv	a3, a2
   1dbc0: be 8c        	mv	s9, a5
   1dbc2: e3 05 0c fa  	beqz	s8, 0x1db6c <_svfiprintf_r+0x3c8>
   1dbc6: 85 47        	addi	a5, zero, 1
   1dbc8: 63 01 f7 74  	beq	a4, a5, 0x1e30a <_svfiprintf_r+0xb66>
   1dbcc: 89 47        	addi	a5, zero, 2
   1dbce: 63 05 f7 6c  	beq	a4, a5, 0x1e298 <_svfiprintf_r+0xaf4>
   1dbd2: 01 47        	mv	a4, zero
   1dbd4: 61 a1        	j	0x1e05c <_svfiprintf_r+0x8b8>
   1dbd6: 13 e3 0c 01  	ori	t1, s9, 16
   1dbda: 01 23        	sext.w	t1, t1
   1dbdc: 93 77 03 02  	andi	a5, t1, 32
   1dbe0: 93 8e 86 00  	addi	t4, a3, 8
   1dbe4: 89 e7        	bnez	a5, 0x1dbee <_svfiprintf_r+0x44a>
   1dbe6: 93 77 03 01  	andi	a5, t1, 16
   1dbea: 63 85 07 54  	beqz	a5, 0x1e134 <_svfiprintf_r+0x990>
   1dbee: 9c 62        	ld	a5, 0(a3)
   1dbf0: a3 03 01 06  	sb	zero, 103(sp)
   1dbf4: 7d 57        	addi	a4, zero, -1
   1dbf6: 9a 8c        	mv	s9, t1
   1dbf8: e3 04 ec ea  	beq	s8, a4, 0x1daa0 <_svfiprintf_r+0x2fc>
   1dbfc: 93 7c f3 f7  	andi	s9, t1, -129
   1dc00: 81 2c        	sext.w	s9, s9
   1dc02: e3 9f 07 e8  	bnez	a5, 0x1daa0 <_svfiprintf_r+0x2fc>
   1dc06: f6 86        	mv	a3, t4
   1dc08: 05 47        	addi	a4, zero, 1
   1dc0a: b9 bf        	j	0x1db68 <_svfiprintf_r+0x3c4>
   1dc0c: 93 ec 0c 08  	ori	s9, s9, 128
   1dc10: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dc14: 81 2c        	sext.w	s9, s9
   1dc16: 89 b9        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dc18: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dc1c: 93 05 a0 02  	addi	a1, zero, 42
   1dc20: 13 87 1d 00  	addi	a4, s11, 1
   1dc24: 63 82 b7 70  	beq	a5, a1, 0x1e328 <_svfiprintf_r+0xb84>
   1dc28: 1b 86 07 fd  	addiw	a2, a5, -48
   1dc2c: 25 45        	addi	a0, zero, 9
   1dc2e: ba 8d        	mv	s11, a4
   1dc30: 01 4c        	mv	s8, zero
   1dc32: a5 45        	addi	a1, zero, 9
   1dc34: e3 6c c5 c2  	bltu	a0, a2, 0x1d86c <_svfiprintf_r+0xc8>
   1dc38: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dc3c: 1b 17 2c 00  	slliw	a4, s8, 2
   1dc40: 3b 0c 87 01  	addw	s8, a4, s8
   1dc44: 1b 1c 1c 00  	slliw	s8, s8, 1
   1dc48: 3b 0c cc 00  	addw	s8, s8, a2
   1dc4c: 1b 86 07 fd  	addiw	a2, a5, -48
   1dc50: 85 0d        	addi	s11, s11, 1
   1dc52: e3 f3 c5 fe  	bgeu	a1, a2, 0x1dc38 <_svfiprintf_r+0x494>
   1dc56: 19 b9        	j	0x1d86c <_svfiprintf_r+0xc8>
   1dc58: 13 07 b0 02  	addi	a4, zero, 43
   1dc5c: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dc60: a3 03 e1 06  	sb	a4, 103(sp)
   1dc64: 11 b1        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dc66: 5a 85        	mv	a0, s6
   1dc68: 36 f0        	sd	a3, 32(sp)
   1dc6a: 46 ec        	sd	a7, 24(sp)
   1dc6c: ef a0 2f e6  	jal	0x182ce <_localeconv_r>
   1dc70: 1c 65        	ld	a5, 8(a0)
   1dc72: 3e 85        	mv	a0, a5
   1dc74: be e0        	sd	a5, 64(sp)
   1dc76: ef c0 8f 84  	jal	0x19cbe <strlen>
   1dc7a: aa 87        	mv	a5, a0
   1dc7c: 5a 85        	mv	a0, s6
   1dc7e: 3e f8        	sd	a5, 48(sp)
   1dc80: ef a0 ef e4  	jal	0x182ce <_localeconv_r>
   1dc84: 18 69        	ld	a4, 16(a0)
   1dc86: c2 77        	ld	a5, 48(sp)
   1dc88: e2 68        	ld	a7, 24(sp)
   1dc8a: 3a fc        	sd	a4, 56(sp)
   1dc8c: 82 76        	ld	a3, 32(sp)
   1dc8e: 63 96 07 48  	bnez	a5, 0x1e11a <_svfiprintf_r+0x976>
   1dc92: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dc96: c9 be        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dc98: 93 ec 1c 00  	ori	s9, s9, 1
   1dc9c: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dca0: 81 2c        	sext.w	s9, s9
   1dca2: d9 b6        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dca4: 03 47 71 06  	lbu	a4, 103(sp)
   1dca8: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dcac: e3 1e 07 ba  	bnez	a4, 0x1d868 <_svfiprintf_r+0xc4>
   1dcb0: 13 07 00 02  	addi	a4, zero, 32
   1dcb4: a3 03 e1 06  	sb	a4, 103(sp)
   1dcb8: 45 be        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dcba: 93 f7 0c 02  	andi	a5, s9, 32
   1dcbe: 98 62        	ld	a4, 0(a3)
   1dcc0: 85 e3        	bnez	a5, 0x1dce0 <_svfiprintf_r+0x53c>
   1dcc2: 93 f7 0c 01  	andi	a5, s9, 16
   1dcc6: 89 ef        	bnez	a5, 0x1dce0 <_svfiprintf_r+0x53c>
   1dcc8: 93 f7 0c 04  	andi	a5, s9, 64
   1dccc: 63 96 07 60  	bnez	a5, 0x1e2d8 <_svfiprintf_r+0xb34>
   1dcd0: 93 fc 0c 20  	andi	s9, s9, 512
   1dcd4: 63 8f 0c 62  	beqz	s9, 0x1e312 <_svfiprintf_r+0xb6e>
   1dcd8: 23 00 a7 01  	sb	s10, 0(a4)
   1dcdc: a1 06        	addi	a3, a3, 8
   1dcde: 2d b6        	j	0x1d808 <_svfiprintf_r+0x64>
   1dce0: 23 30 a7 01  	sd	s10, 0(a4)
   1dce4: a1 06        	addi	a3, a3, 8
   1dce6: 0d b6        	j	0x1d808 <_svfiprintf_r+0x64>
   1dce8: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dcec: 13 07 c0 06  	addi	a4, zero, 108
   1dcf0: 63 8f e7 56  	beq	a5, a4, 0x1e26e <_svfiprintf_r+0xaca>
   1dcf4: 93 ec 0c 01  	ori	s9, s9, 16
   1dcf8: 81 2c        	sext.w	s9, s9
   1dcfa: bd b6        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dcfc: 83 c7 0d 00  	lbu	a5, 0(s11)
   1dd00: 13 07 80 06  	addi	a4, zero, 104
   1dd04: 63 8d e7 54  	beq	a5, a4, 0x1e25e <_svfiprintf_r+0xaba>
   1dd08: 93 ec 0c 04  	ori	s9, s9, 64
   1dd0c: 81 2c        	sext.w	s9, s9
   1dd0e: a9 be        	j	0x1d868 <_svfiprintf_r+0xc4>
   1dd10: 3b 87 c8 41  	subw	a4, a7, t3
   1dd14: e3 5f e0 c0  	blez	a4, 0x1d932 <_svfiprintf_r+0x18e>
   1dd18: 37 3f 02 00  	lui	t5, 35
   1dd1c: 41 46        	addi	a2, zero, 16
   1dd1e: 13 0f 0f 9a  	addi	t5, t5, -1632
   1dd22: 63 51 e6 06  	bge	a2, a4, 0x1dd84 <_svfiprintf_r+0x5e0>
   1dd26: 26 86        	mv	a2, s1
   1dd28: b6 e4        	sd	a3, 72(sp)
   1dd2a: c1 4e        	addi	t4, zero, 16
   1dd2c: a2 86        	mv	a3, s0
   1dd2e: 9d 4f        	addi	t6, zero, 7
   1dd30: 46 ec        	sd	a7, 24(sp)
   1dd32: 72 f0        	sd	t3, 32(sp)
   1dd34: 1a f4        	sd	t1, 40(sp)
   1dd36: 3a 84        	mv	s0, a4
   1dd38: fa 84        	mv	s1, t5
   1dd3a: 21 a0        	j	0x1dd42 <_svfiprintf_r+0x59e>
   1dd3c: 41 34        	addiw	s0, s0, -16
   1dd3e: 63 db 8e 02  	bge	t4, s0, 0x1dd74 <_svfiprintf_r+0x5d0>
   1dd42: 85 27        	addiw	a5, a5, 1
   1dd44: 41 06        	addi	a2, a2, 16
   1dd46: 84 e2        	sd	s1, 0(a3)
   1dd48: 23 b4 d6 01  	sd	t4, 8(a3)
   1dd4c: 32 e1        	sd	a2, 128(sp)
   1dd4e: be dc        	sw	a5, 120(sp)
   1dd50: c1 06        	addi	a3, a3, 16
   1dd52: e3 d5 ff fe  	bge	t6, a5, 0x1dd3c <_svfiprintf_r+0x598>
   1dd56: 90 18        	addi	a2, sp, 112
   1dd58: d2 85        	mv	a1, s4
   1dd5a: 5a 85        	mv	a0, s6
   1dd5c: ef f0 3f 8f  	jal	0x1d64e <__ssprint_r>
   1dd60: e3 18 05 ca  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1dd64: c1 4e        	addi	t4, zero, 16
   1dd66: 41 34        	addiw	s0, s0, -16
   1dd68: 0a 66        	ld	a2, 128(sp)
   1dd6a: e6 57        	lw	a5, 120(sp)
   1dd6c: d6 86        	mv	a3, s5
   1dd6e: 9d 4f        	addi	t6, zero, 7
   1dd70: e3 c9 8e fc  	blt	t4, s0, 0x1dd42 <_svfiprintf_r+0x59e>
   1dd74: 22 87        	mv	a4, s0
   1dd76: e2 68        	ld	a7, 24(sp)
   1dd78: 36 84        	mv	s0, a3
   1dd7a: 02 7e        	ld	t3, 32(sp)
   1dd7c: 22 73        	ld	t1, 40(sp)
   1dd7e: a6 66        	ld	a3, 72(sp)
   1dd80: 26 8f        	mv	t5, s1
   1dd82: b2 84        	mv	s1, a2
   1dd84: ba 94        	add	s1, s1, a4
   1dd86: 85 27        	addiw	a5, a5, 1
   1dd88: 18 e4        	sd	a4, 8(s0)
   1dd8a: 23 30 e4 01  	sd	t5, 0(s0)
   1dd8e: 26 e1        	sd	s1, 128(sp)
   1dd90: be dc        	sw	a5, 120(sp)
   1dd92: 1d 47        	addi	a4, zero, 7
   1dd94: 41 04        	addi	s0, s0, 16
   1dd96: e3 5e f7 b8  	bge	a4, a5, 0x1d932 <_svfiprintf_r+0x18e>
   1dd9a: 90 18        	addi	a2, sp, 112
   1dd9c: d2 85        	mv	a1, s4
   1dd9e: 5a 85        	mv	a0, s6
   1dda0: b6 e4        	sd	a3, 72(sp)
   1dda2: 1a f4        	sd	t1, 40(sp)
   1dda4: 72 f0        	sd	t3, 32(sp)
   1dda6: 46 ec        	sd	a7, 24(sp)
   1dda8: ef f0 7f 8a  	jal	0x1d64e <__ssprint_r>
   1ddac: e3 12 05 c6  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1ddb0: 22 73        	ld	t1, 40(sp)
   1ddb2: 8a 64        	ld	s1, 128(sp)
   1ddb4: e6 57        	lw	a5, 120(sp)
   1ddb6: 3b 0c 6c 40  	subw	s8, s8, t1
   1ddba: a6 66        	ld	a3, 72(sp)
   1ddbc: 02 7e        	ld	t3, 32(sp)
   1ddbe: e2 68        	ld	a7, 24(sp)
   1ddc0: 56 84        	mv	s0, s5
   1ddc2: e3 5c 80 b7  	blez	s8, 0x1d93a <_svfiprintf_r+0x196>
   1ddc6: 37 3f 02 00  	lui	t5, 35
   1ddca: 13 0f 0f 9a  	addi	t5, t5, -1632
   1ddce: 63 d2 89 07  	bge	s3, s8, 0x1de32 <_svfiprintf_r+0x68e>
   1ddd2: 36 86        	mv	a2, a3
   1ddd4: 22 87        	mv	a4, s0
   1ddd6: a6 86        	mv	a3, s1
   1ddd8: 62 84        	mv	s0, s8
   1ddda: 9d 4e        	addi	t4, zero, 7
   1dddc: 46 ec        	sd	a7, 24(sp)
   1ddde: 72 f0        	sd	t3, 32(sp)
   1dde0: 1a f4        	sd	t1, 40(sp)
   1dde2: 5e 8c        	mv	s8, s7
   1dde4: b2 84        	mv	s1, a2
   1dde6: 21 a0        	j	0x1ddee <_svfiprintf_r+0x64a>
   1dde8: 41 34        	addiw	s0, s0, -16
   1ddea: 63 db 89 02  	bge	s3, s0, 0x1de20 <_svfiprintf_r+0x67c>
   1ddee: 85 27        	addiw	a5, a5, 1
   1ddf0: c1 06        	addi	a3, a3, 16
   1ddf2: 23 30 77 01  	sd	s7, 0(a4)
   1ddf6: 23 34 37 01  	sd	s3, 8(a4)
   1ddfa: 36 e1        	sd	a3, 128(sp)
   1ddfc: be dc        	sw	a5, 120(sp)
   1ddfe: 41 07        	addi	a4, a4, 16
   1de00: e3 d4 fe fe  	bge	t4, a5, 0x1dde8 <_svfiprintf_r+0x644>
   1de04: 90 18        	addi	a2, sp, 112
   1de06: d2 85        	mv	a1, s4
   1de08: 5a 85        	mv	a0, s6
   1de0a: ef f0 5f 84  	jal	0x1d64e <__ssprint_r>
   1de0e: e3 11 05 c0  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1de12: 41 34        	addiw	s0, s0, -16
   1de14: 8a 66        	ld	a3, 128(sp)
   1de16: e6 57        	lw	a5, 120(sp)
   1de18: 56 87        	mv	a4, s5
   1de1a: 9d 4e        	addi	t4, zero, 7
   1de1c: e3 c9 89 fc  	blt	s3, s0, 0x1ddee <_svfiprintf_r+0x64a>
   1de20: e2 68        	ld	a7, 24(sp)
   1de22: 02 7e        	ld	t3, 32(sp)
   1de24: 22 73        	ld	t1, 40(sp)
   1de26: 26 86        	mv	a2, s1
   1de28: 62 8f        	mv	t5, s8
   1de2a: b6 84        	mv	s1, a3
   1de2c: 22 8c        	mv	s8, s0
   1de2e: b2 86        	mv	a3, a2
   1de30: 3a 84        	mv	s0, a4
   1de32: 85 27        	addiw	a5, a5, 1
   1de34: e2 94        	add	s1, s1, s8
   1de36: 23 30 e4 01  	sd	t5, 0(s0)
   1de3a: 23 34 84 01  	sd	s8, 8(s0)
   1de3e: 26 e1        	sd	s1, 128(sp)
   1de40: be dc        	sw	a5, 120(sp)
   1de42: 1d 47        	addi	a4, zero, 7
   1de44: 41 04        	addi	s0, s0, 16
   1de46: e3 5a f7 ae  	bge	a4, a5, 0x1d93a <_svfiprintf_r+0x196>
   1de4a: 90 18        	addi	a2, sp, 112
   1de4c: d2 85        	mv	a1, s4
   1de4e: 5a 85        	mv	a0, s6
   1de50: b6 e4        	sd	a3, 72(sp)
   1de52: 1a f4        	sd	t1, 40(sp)
   1de54: 72 f0        	sd	t3, 32(sp)
   1de56: 46 ec        	sd	a7, 24(sp)
   1de58: ef f0 6f ff  	jal	0x1d64e <__ssprint_r>
   1de5c: e3 1a 05 ba  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1de60: 8a 64        	ld	s1, 128(sp)
   1de62: e6 57        	lw	a5, 120(sp)
   1de64: a6 66        	ld	a3, 72(sp)
   1de66: 22 73        	ld	t1, 40(sp)
   1de68: 02 7e        	ld	t3, 32(sp)
   1de6a: e2 68        	ld	a7, 24(sp)
   1de6c: 56 84        	mv	s0, s5
   1de6e: f1 b4        	j	0x1d93a <_svfiprintf_r+0x196>
   1de70: 90 18        	addi	a2, sp, 112
   1de72: d2 85        	mv	a1, s4
   1de74: 5a 85        	mv	a0, s6
   1de76: 36 f4        	sd	a3, 40(sp)
   1de78: 72 f0        	sd	t3, 32(sp)
   1de7a: 46 ec        	sd	a7, 24(sp)
   1de7c: ef f0 2f fd  	jal	0x1d64e <__ssprint_r>
   1de80: e3 18 05 b8  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1de84: 8a 64        	ld	s1, 128(sp)
   1de86: a2 76        	ld	a3, 40(sp)
   1de88: 02 7e        	ld	t3, 32(sp)
   1de8a: e2 68        	ld	a7, 24(sp)
   1de8c: 56 84        	mv	s0, s5
   1de8e: d9 b4        	j	0x1d954 <_svfiprintf_r+0x1b0>
   1de90: 90 18        	addi	a2, sp, 112
   1de92: d2 85        	mv	a1, s4
   1de94: 5a 85        	mv	a0, s6
   1de96: 36 ec        	sd	a3, 24(sp)
   1de98: ef f0 6f fb  	jal	0x1d64e <__ssprint_r>
   1de9c: e3 1a 05 b6  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1dea0: 8a 64        	ld	s1, 128(sp)
   1dea2: e2 66        	ld	a3, 24(sp)
   1dea4: c1 bc        	j	0x1d974 <_svfiprintf_r+0x1d0>
   1dea6: be 8c        	mv	s9, a5
   1dea8: 89 47        	addi	a5, zero, 2
   1deaa: 63 89 07 1a  	beqz	a5, 0x1e05c <_svfiprintf_r+0x8b8>
   1deae: 42 66        	ld	a2, 16(sp)
   1deb0: 13 03 c1 0e  	addi	t1, sp, 236
   1deb4: 1a 89        	mv	s2, t1
   1deb6: 93 77 f7 00  	andi	a5, a4, 15
   1deba: b2 97        	add	a5, a5, a2
   1debc: 83 c7 07 00  	lbu	a5, 0(a5)
   1dec0: 11 83        	srli	a4, a4, 4
   1dec2: 7d 19        	addi	s2, s2, -1
   1dec4: 23 00 f9 00  	sb	a5, 0(s2)
   1dec8: 7d f7        	bnez	a4, 0x1deb6 <_svfiprintf_r+0x712>
   1deca: 3b 03 23 41  	subw	t1, t1, s2
   1dece: 1b 0e 0c 00  	sext.w	t3, s8
   1ded2: e3 59 6c be  	bge	s8, t1, 0x1dac4 <_svfiprintf_r+0x320>
   1ded6: 1b 0e 03 00  	sext.w	t3, t1
   1deda: ed b6        	j	0x1dac4 <_svfiprintf_r+0x320>
   1dedc: b7 3e 02 00  	lui	t4, 35
   1dee0: 41 46        	addi	a2, zero, 16
   1dee2: 93 8e 0e 9b  	addi	t4, t4, -1616
   1dee6: 63 55 e6 06  	bge	a2, a4, 0x1df50 <_svfiprintf_r+0x7ac>
   1deea: 26 86        	mv	a2, s1
   1deec: b6 ec        	sd	a3, 88(sp)
   1deee: 41 4f        	addi	t5, zero, 16
   1def0: a2 86        	mv	a3, s0
   1def2: 9d 43        	addi	t2, zero, 7
   1def4: 16 ec        	sd	t0, 24(sp)
   1def6: 7e f0        	sd	t6, 32(sp)
   1def8: 46 f4        	sd	a7, 40(sp)
   1defa: f2 e4        	sd	t3, 72(sp)
   1defc: 9a e8        	sd	t1, 80(sp)
   1defe: 3a 84        	mv	s0, a4
   1df00: f6 84        	mv	s1, t4
   1df02: 21 a0        	j	0x1df0a <_svfiprintf_r+0x766>
   1df04: 41 34        	addiw	s0, s0, -16
   1df06: 63 5b 8f 02  	bge	t5, s0, 0x1df3c <_svfiprintf_r+0x798>
   1df0a: 85 27        	addiw	a5, a5, 1
   1df0c: 41 06        	addi	a2, a2, 16
   1df0e: 84 e2        	sd	s1, 0(a3)
   1df10: 23 b4 e6 01  	sd	t5, 8(a3)
   1df14: 32 e1        	sd	a2, 128(sp)
   1df16: be dc        	sw	a5, 120(sp)
   1df18: c1 06        	addi	a3, a3, 16
   1df1a: e3 d5 f3 fe  	bge	t2, a5, 0x1df04 <_svfiprintf_r+0x760>
   1df1e: 90 18        	addi	a2, sp, 112
   1df20: d2 85        	mv	a1, s4
   1df22: 5a 85        	mv	a0, s6
   1df24: ef f0 af f2  	jal	0x1d64e <__ssprint_r>
   1df28: e3 14 05 ae  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1df2c: 41 4f        	addi	t5, zero, 16
   1df2e: 41 34        	addiw	s0, s0, -16
   1df30: 0a 66        	ld	a2, 128(sp)
   1df32: e6 57        	lw	a5, 120(sp)
   1df34: d6 86        	mv	a3, s5
   1df36: 9d 43        	addi	t2, zero, 7
   1df38: e3 49 8f fc  	blt	t5, s0, 0x1df0a <_svfiprintf_r+0x766>
   1df3c: 22 87        	mv	a4, s0
   1df3e: e2 62        	ld	t0, 24(sp)
   1df40: 36 84        	mv	s0, a3
   1df42: 82 7f        	ld	t6, 32(sp)
   1df44: a2 78        	ld	a7, 40(sp)
   1df46: 26 6e        	ld	t3, 72(sp)
   1df48: 46 63        	ld	t1, 80(sp)
   1df4a: e6 66        	ld	a3, 88(sp)
   1df4c: a6 8e        	mv	t4, s1
   1df4e: b2 84        	mv	s1, a2
   1df50: ba 94        	add	s1, s1, a4
   1df52: 85 27        	addiw	a5, a5, 1
   1df54: 18 e4        	sd	a4, 8(s0)
   1df56: 23 30 d4 01  	sd	t4, 0(s0)
   1df5a: 26 e1        	sd	s1, 128(sp)
   1df5c: be dc        	sw	a5, 120(sp)
   1df5e: 1d 47        	addi	a4, zero, 7
   1df60: 41 04        	addi	s0, s0, 16
   1df62: e3 56 f7 98  	bge	a4, a5, 0x1d8ee <_svfiprintf_r+0x14a>
   1df66: 90 18        	addi	a2, sp, 112
   1df68: d2 85        	mv	a1, s4
   1df6a: 5a 85        	mv	a0, s6
   1df6c: b6 ec        	sd	a3, 88(sp)
   1df6e: 9a e8        	sd	t1, 80(sp)
   1df70: f2 e4        	sd	t3, 72(sp)
   1df72: 46 f4        	sd	a7, 40(sp)
   1df74: 7e f0        	sd	t6, 32(sp)
   1df76: 16 ec        	sd	t0, 24(sp)
   1df78: ef f0 6f ed  	jal	0x1d64e <__ssprint_r>
   1df7c: e3 1a 05 a8  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1df80: 8a 64        	ld	s1, 128(sp)
   1df82: e6 57        	lw	a5, 120(sp)
   1df84: e6 66        	ld	a3, 88(sp)
   1df86: 46 63        	ld	t1, 80(sp)
   1df88: 26 6e        	ld	t3, 72(sp)
   1df8a: a2 78        	ld	a7, 40(sp)
   1df8c: 82 7f        	ld	t6, 32(sp)
   1df8e: e2 62        	ld	t0, 24(sp)
   1df90: 56 84        	mv	s0, s5
   1df92: b1 ba        	j	0x1d8ee <_svfiprintf_r+0x14a>
   1df94: 90 18        	addi	a2, sp, 112
   1df96: d2 85        	mv	a1, s4
   1df98: 5a 85        	mv	a0, s6
   1df9a: b6 e8        	sd	a3, 80(sp)
   1df9c: 9a e4        	sd	t1, 72(sp)
   1df9e: 72 f4        	sd	t3, 40(sp)
   1dfa0: 46 f0        	sd	a7, 32(sp)
   1dfa2: 7e ec        	sd	t6, 24(sp)
   1dfa4: ef f0 af ea  	jal	0x1d64e <__ssprint_r>
   1dfa8: e3 14 05 a6  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1dfac: 8a 64        	ld	s1, 128(sp)
   1dfae: e6 57        	lw	a5, 120(sp)
   1dfb0: c6 66        	ld	a3, 80(sp)
   1dfb2: 26 63        	ld	t1, 72(sp)
   1dfb4: 22 7e        	ld	t3, 40(sp)
   1dfb6: 82 78        	ld	a7, 32(sp)
   1dfb8: e2 6f        	ld	t6, 24(sp)
   1dfba: 56 84        	mv	s0, s5
   1dfbc: bd b2        	j	0x1d92a <_svfiprintf_r+0x186>
   1dfbe: b7 3e 02 00  	lui	t4, 35
   1dfc2: 41 47        	addi	a4, zero, 16
   1dfc4: e6 57        	lw	a5, 120(sp)
   1dfc6: 93 8e 0e 9b  	addi	t4, t4, -1616
   1dfca: 63 5d 27 05  	bge	a4, s2, 0x1e024 <_svfiprintf_r+0x880>
   1dfce: 22 87        	mv	a4, s0
   1dfd0: 41 4c        	addi	s8, zero, 16
   1dfd2: 4a 84        	mv	s0, s2
   1dfd4: 9d 4c        	addi	s9, zero, 7
   1dfd6: 36 89        	mv	s2, a3
   1dfd8: 46 ec        	sd	a7, 24(sp)
   1dfda: a6 86        	mv	a3, s1
   1dfdc: 72 f0        	sd	t3, 32(sp)
   1dfde: f6 84        	mv	s1, t4
   1dfe0: 21 a0        	j	0x1dfe8 <_svfiprintf_r+0x844>
   1dfe2: 41 34        	addiw	s0, s0, -16
   1dfe4: 63 59 8c 02  	bge	s8, s0, 0x1e016 <_svfiprintf_r+0x872>
   1dfe8: 85 27        	addiw	a5, a5, 1
   1dfea: c1 06        	addi	a3, a3, 16
   1dfec: 04 e3        	sd	s1, 0(a4)
   1dfee: 23 34 87 01  	sd	s8, 8(a4)
   1dff2: 36 e1        	sd	a3, 128(sp)
   1dff4: be dc        	sw	a5, 120(sp)
   1dff6: 41 07        	addi	a4, a4, 16
   1dff8: e3 d5 fc fe  	bge	s9, a5, 0x1dfe2 <_svfiprintf_r+0x83e>
   1dffc: 90 18        	addi	a2, sp, 112
   1dffe: d2 85        	mv	a1, s4
   1e000: 5a 85        	mv	a0, s6
   1e002: ef f0 cf e4  	jal	0x1d64e <__ssprint_r>
   1e006: e3 15 05 a0  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1e00a: 41 34        	addiw	s0, s0, -16
   1e00c: 8a 66        	ld	a3, 128(sp)
   1e00e: e6 57        	lw	a5, 120(sp)
   1e010: 56 87        	mv	a4, s5
   1e012: e3 4b 8c fc  	blt	s8, s0, 0x1dfe8 <_svfiprintf_r+0x844>
   1e016: e2 68        	ld	a7, 24(sp)
   1e018: 02 7e        	ld	t3, 32(sp)
   1e01a: a6 8e        	mv	t4, s1
   1e01c: b6 84        	mv	s1, a3
   1e01e: ca 86        	mv	a3, s2
   1e020: 22 89        	mv	s2, s0
   1e022: 3a 84        	mv	s0, a4
   1e024: 1b 87 17 00  	addiw	a4, a5, 1
   1e028: ca 94        	add	s1, s1, s2
   1e02a: 23 30 d4 01  	sd	t4, 0(s0)
   1e02e: 23 34 24 01  	sd	s2, 8(s0)
   1e032: 26 e1        	sd	s1, 128(sp)
   1e034: ba dc        	sw	a4, 120(sp)
   1e036: 9d 47        	addi	a5, zero, 7
   1e038: e3 d6 e7 92  	bge	a5, a4, 0x1d964 <_svfiprintf_r+0x1c0>
   1e03c: 90 18        	addi	a2, sp, 112
   1e03e: d2 85        	mv	a1, s4
   1e040: 5a 85        	mv	a0, s6
   1e042: 36 f4        	sd	a3, 40(sp)
   1e044: 72 f0        	sd	t3, 32(sp)
   1e046: 46 ec        	sd	a7, 24(sp)
   1e048: ef f0 6f e0  	jal	0x1d64e <__ssprint_r>
   1e04c: e3 12 05 9c  	bnez	a0, 0x1da10 <_svfiprintf_r+0x26c>
   1e050: 8a 64        	ld	s1, 128(sp)
   1e052: a2 76        	ld	a3, 40(sp)
   1e054: 02 7e        	ld	t3, 32(sp)
   1e056: e2 68        	ld	a7, 24(sp)
   1e058: 31 b2        	j	0x1d964 <_svfiprintf_r+0x1c0>
   1e05a: b2 86        	mv	a3, a2
   1e05c: 13 03 c1 0e  	addi	t1, sp, 236
   1e060: 1a 89        	mv	s2, t1
   1e062: 93 77 77 00  	andi	a5, a4, 7
   1e066: 93 87 07 03  	addi	a5, a5, 48
   1e06a: a3 0f f9 fe  	sb	a5, -1(s2)
   1e06e: 0d 83        	srli	a4, a4, 3
   1e070: 4a 86        	mv	a2, s2
   1e072: 7d 19        	addi	s2, s2, -1
   1e074: 7d f7        	bnez	a4, 0x1e062 <_svfiprintf_r+0x8be>
   1e076: 13 f7 1c 00  	andi	a4, s9, 1
   1e07a: 05 c3        	beqz	a4, 0x1e09a <_svfiprintf_r+0x8f6>
   1e07c: 13 07 00 03  	addi	a4, zero, 48
   1e080: 63 8d e7 00  	beq	a5, a4, 0x1e09a <_svfiprintf_r+0x8f6>
   1e084: 79 16        	addi	a2, a2, -2
   1e086: 3b 03 c3 40  	subw	t1, t1, a2
   1e08a: a3 0f e9 fe  	sb	a4, -1(s2)
   1e08e: 1b 0e 03 00  	sext.w	t3, t1
   1e092: 63 41 83 0d  	blt	t1, s8, 0x1e154 <_svfiprintf_r+0x9b0>
   1e096: 32 89        	mv	s2, a2
   1e098: 35 b4        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e09a: 3b 03 23 41  	subw	t1, t1, s2
   1e09e: 1b 0e 03 00  	sext.w	t3, t1
   1e0a2: e3 51 83 a3  	bge	t1, s8, 0x1dac4 <_svfiprintf_r+0x320>
   1e0a6: 1b 0e 0c 00  	sext.w	t3, s8
   1e0aa: 29 bc        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e0ac: 01 4e        	mv	t3, zero
   1e0ae: 01 43        	mv	t1, zero
   1e0b0: 13 09 c1 0e  	addi	s2, sp, 236
   1e0b4: 01 bc        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e0b6: 13 03 c1 0e  	addi	t1, sp, 236
   1e0ba: 62 78        	ld	a6, 56(sp)
   1e0bc: 13 ff 0c 40  	andi	t5, s9, 1024
   1e0c0: 81 45        	mv	a1, zero
   1e0c2: 1a 86        	mv	a2, t1
   1e0c4: a9 46        	addi	a3, zero, 10
   1e0c6: 25 4e        	addi	t3, zero, 9
   1e0c8: 93 0f f0 0f  	addi	t6, zero, 255
   1e0cc: 39 a0        	j	0x1e0da <_svfiprintf_r+0x936>
   1e0ce: 33 d7 d7 02  	divu	a4, a5, a3
   1e0d2: 63 77 fe 02  	bgeu	t3, a5, 0x1e100 <_svfiprintf_r+0x95c>
   1e0d6: ba 87        	mv	a5, a4
   1e0d8: 4a 86        	mv	a2, s2
   1e0da: 13 09 f6 ff  	addi	s2, a2, -1
   1e0de: 85 25        	addiw	a1, a1, 1
   1e0e0: 33 f7 d7 02  	remu	a4, a5, a3
   1e0e4: 1b 07 07 03  	addiw	a4, a4, 48
   1e0e8: a3 0f e6 fe  	sb	a4, -1(a2)
   1e0ec: e3 01 0f fe  	beqz	t5, 0x1e0ce <_svfiprintf_r+0x92a>
   1e0f0: 03 47 08 00  	lbu	a4, 0(a6)
   1e0f4: e3 1d b7 fc  	bne	a4, a1, 0x1e0ce <_svfiprintf_r+0x92a>
   1e0f8: e3 0b f7 fd  	beq	a4, t6, 0x1e0ce <_svfiprintf_r+0x92a>
   1e0fc: 63 69 fe 10  	bltu	t3, a5, 0x1e20e <_svfiprintf_r+0xa6a>
   1e100: 3b 03 23 41  	subw	t1, t1, s2
   1e104: 42 fc        	sd	a6, 56(sp)
   1e106: 1b 0e 03 00  	sext.w	t3, t1
   1e10a: 63 4e 83 0d  	blt	t1, s8, 0x1e1e6 <_svfiprintf_r+0xa42>
   1e10e: f6 86        	mv	a3, t4
   1e110: 55 ba        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e112: 01 4e        	mv	t3, zero
   1e114: 13 09 c1 0e  	addi	s2, sp, 236
   1e118: 75 b2        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e11a: 83 c7 0d 00  	lbu	a5, 0(s11)
   1e11e: 63 05 07 f4  	beqz	a4, 0x1d868 <_svfiprintf_r+0xc4>
   1e122: 03 47 07 00  	lbu	a4, 0(a4)
   1e126: 63 01 07 f4  	beqz	a4, 0x1d868 <_svfiprintf_r+0xc4>
   1e12a: 93 ec 0c 40  	ori	s9, s9, 1024
   1e12e: 81 2c        	sext.w	s9, s9
   1e130: 6f f0 8f f3  	j	0x1d868 <_svfiprintf_r+0xc4>
   1e134: 13 77 03 04  	andi	a4, t1, 64
   1e138: 9c 42        	lw	a5, 0(a3)
   1e13a: 63 07 07 16  	beqz	a4, 0x1e2a8 <_svfiprintf_r+0xb04>
   1e13e: c2 17        	slli	a5, a5, 48
   1e140: c1 93        	srli	a5, a5, 48
   1e142: 7d b4        	j	0x1dbf0 <_svfiprintf_r+0x44c>
   1e144: 93 f5 07 04  	andi	a1, a5, 64
   1e148: 98 42        	lw	a4, 0(a3)
   1e14a: 63 85 05 16  	beqz	a1, 0x1e2b4 <_svfiprintf_r+0xb10>
   1e14e: 42 17        	slli	a4, a4, 48
   1e150: 41 93        	srli	a4, a4, 48
   1e152: 81 bc        	j	0x1dba2 <_svfiprintf_r+0x3fe>
   1e154: 1b 0e 0c 00  	sext.w	t3, s8
   1e158: 32 89        	mv	s2, a2
   1e15a: ad b2        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e15c: 37 27 02 00  	lui	a4, 34
   1e160: 13 07 07 bf  	addi	a4, a4, -1040
   1e164: 3a e8        	sd	a4, 16(sp)
   1e166: 13 f7 0c 02  	andi	a4, s9, 32
   1e16a: 13 86 86 00  	addi	a2, a3, 8
   1e16e: 39 eb        	bnez	a4, 0x1e1c4 <_svfiprintf_r+0xa20>
   1e170: 13 f7 0c 01  	andi	a4, s9, 16
   1e174: 21 eb        	bnez	a4, 0x1e1c4 <_svfiprintf_r+0xa20>
   1e176: 93 f5 0c 04  	andi	a1, s9, 64
   1e17a: 98 42        	lw	a4, 0(a3)
   1e17c: 63 80 05 12  	beqz	a1, 0x1e29c <_svfiprintf_r+0xaf8>
   1e180: 42 17        	slli	a4, a4, 48
   1e182: 41 93        	srli	a4, a4, 48
   1e184: 93 f6 1c 00  	andi	a3, s9, 1
   1e188: 99 ca        	beqz	a3, 0x1e19e <_svfiprintf_r+0x9fa>
   1e18a: 11 cb        	beqz	a4, 0x1e19e <_svfiprintf_r+0x9fa>
   1e18c: 93 ec 2c 00  	ori	s9, s9, 2
   1e190: 93 06 00 03  	addi	a3, zero, 48
   1e194: 23 04 d1 06  	sb	a3, 104(sp)
   1e198: a3 04 f1 06  	sb	a5, 105(sp)
   1e19c: 81 2c        	sext.w	s9, s9
   1e19e: 93 fc fc bf  	andi	s9, s9, -1025
   1e1a2: 81 2c        	sext.w	s9, s9
   1e1a4: b2 86        	mv	a3, a2
   1e1a6: 65 b2        	j	0x1db4e <_svfiprintf_r+0x3aa>
   1e1a8: e6 87        	mv	a5, s9
   1e1aa: d5 b2        	j	0x1db8e <_svfiprintf_r+0x3ea>
   1e1ac: 66 83        	mv	t1, s9
   1e1ae: 3d b4        	j	0x1dbdc <_svfiprintf_r+0x438>
   1e1b0: 37 27 02 00  	lui	a4, 34
   1e1b4: 13 07 87 c0  	addi	a4, a4, -1016
   1e1b8: 3a e8        	sd	a4, 16(sp)
   1e1ba: 13 f7 0c 02  	andi	a4, s9, 32
   1e1be: 13 86 86 00  	addi	a2, a3, 8
   1e1c2: 5d d7        	beqz	a4, 0x1e170 <_svfiprintf_r+0x9cc>
   1e1c4: 98 62        	ld	a4, 0(a3)
   1e1c6: 7d bf        	j	0x1e184 <_svfiprintf_r+0x9e0>
   1e1c8: 4a 85        	mv	a0, s2
   1e1ca: ef b0 5f af  	jal	0x19cbe <strlen>
   1e1ce: 1b 03 05 00  	sext.w	t1, a0
   1e1d2: 13 4e f3 ff  	not	t3, t1
   1e1d6: 13 5e fe 43  	srai	t3, t3, 63
   1e1da: 82 76        	ld	a3, 32(sp)
   1e1dc: e2 68        	ld	a7, 24(sp)
   1e1de: 33 7e 6e 00  	and	t3, t3, t1
   1e1e2: 01 4c        	mv	s8, zero
   1e1e4: c5 b0        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e1e6: 1b 0e 0c 00  	sext.w	t3, s8
   1e1ea: f6 86        	mv	a3, t4
   1e1ec: e1 b8        	j	0x1dac4 <_svfiprintf_r+0x320>
   1e1ee: 93 05 00 04  	addi	a1, zero, 64
   1e1f2: ef a0 ef b3  	jal	0x18530 <_malloc_r>
   1e1f6: 23 30 aa 00  	sd	a0, 0(s4)
   1e1fa: 23 3c aa 00  	sd	a0, 24(s4)
   1e1fe: 63 0f 05 10  	beqz	a0, 0x1e31c <_svfiprintf_r+0xb78>
   1e202: 13 07 00 04  	addi	a4, zero, 64
   1e206: 23 20 ea 02  	sw	a4, 32(s4)
   1e20a: 6f f0 ef dc  	j	0x1d7d8 <_svfiprintf_r+0x34>
   1e20e: be e4        	sd	a5, 72(sp)
   1e210: c2 77        	ld	a5, 48(sp)
   1e212: 86 65        	ld	a1, 64(sp)
   1e214: 9a e8        	sd	t1, 80(sp)
   1e216: 33 09 f9 40  	sub	s2, s2, a5
   1e21a: 3e 86        	mv	a2, a5
   1e21c: 4a 85        	mv	a0, s2
   1e21e: 76 fc        	sd	t4, 56(sp)
   1e220: 7a f4        	sd	t5, 40(sp)
   1e222: 46 f0        	sd	a7, 32(sp)
   1e224: 42 ec        	sd	a6, 24(sp)
   1e226: ef b0 9f b2  	jal	0x19d4e <strncpy>
   1e22a: 62 68        	ld	a6, 24(sp)
   1e22c: a6 67        	ld	a5, 72(sp)
   1e22e: 29 47        	addi	a4, zero, 10
   1e230: 03 46 18 00  	lbu	a2, 1(a6)
   1e234: 33 d7 e7 02  	divu	a4, a5, a4
   1e238: 82 78        	ld	a7, 32(sp)
   1e23a: b3 37 c0 00  	snez	a5, a2
   1e23e: 22 7f        	ld	t5, 40(sp)
   1e240: e2 7e        	ld	t4, 56(sp)
   1e242: 46 63        	ld	t1, 80(sp)
   1e244: 3e 98        	add	a6, a6, a5
   1e246: 81 45        	mv	a1, zero
   1e248: a9 46        	addi	a3, zero, 10
   1e24a: 25 4e        	addi	t3, zero, 9
   1e24c: 93 0f f0 0f  	addi	t6, zero, 255
   1e250: 59 b5        	j	0x1e0d6 <_svfiprintf_r+0x932>
   1e252: be 8c        	mv	s9, a5
   1e254: 81 47        	mv	a5, zero
   1e256: b2 86        	mv	a3, a2
   1e258: e3 82 07 e0  	beqz	a5, 0x1e05c <_svfiprintf_r+0x8b8>
   1e25c: 89 b9        	j	0x1deae <_svfiprintf_r+0x70a>
   1e25e: 93 ec 0c 20  	ori	s9, s9, 512
   1e262: 83 c7 1d 00  	lbu	a5, 1(s11)
   1e266: 81 2c        	sext.w	s9, s9
   1e268: 85 0d        	addi	s11, s11, 1
   1e26a: 6f f0 ef df  	j	0x1d868 <_svfiprintf_r+0xc4>
   1e26e: 93 ec 0c 02  	ori	s9, s9, 32
   1e272: 83 c7 1d 00  	lbu	a5, 1(s11)
   1e276: 81 2c        	sext.w	s9, s9
   1e278: 85 0d        	addi	s11, s11, 1
   1e27a: 6f f0 ef de  	j	0x1d868 <_svfiprintf_r+0xc4>
   1e27e: 99 47        	addi	a5, zero, 6
   1e280: 1b 03 0c 00  	sext.w	t1, s8
   1e284: 63 f3 87 01  	bgeu	a5, s8, 0x1e28a <_svfiprintf_r+0xae6>
   1e288: 19 43        	addi	t1, zero, 6
   1e28a: 37 29 02 00  	lui	s2, 34
   1e28e: 1a 8e        	mv	t3, t1
   1e290: 13 09 09 c2  	addi	s2, s2, -992
   1e294: 6f f0 cf e3  	j	0x1d8d0 <_svfiprintf_r+0x12c>
   1e298: 01 47        	mv	a4, zero
   1e29a: 11 b9        	j	0x1deae <_svfiprintf_r+0x70a>
   1e29c: 93 f6 0c 20  	andi	a3, s9, 512
   1e2a0: a1 ce        	beqz	a3, 0x1e2f8 <_svfiprintf_r+0xb54>
   1e2a2: 13 77 f7 0f  	andi	a4, a4, 255
   1e2a6: f9 bd        	j	0x1e184 <_svfiprintf_r+0x9e0>
   1e2a8: 13 77 03 20  	andi	a4, t1, 512
   1e2ac: 39 c3        	beqz	a4, 0x1e2f2 <_svfiprintf_r+0xb4e>
   1e2ae: 93 f7 f7 0f  	andi	a5, a5, 255
   1e2b2: 3d ba        	j	0x1dbf0 <_svfiprintf_r+0x44c>
   1e2b4: 93 f6 07 20  	andi	a3, a5, 512
   1e2b8: 95 ca        	beqz	a3, 0x1e2ec <_svfiprintf_r+0xb48>
   1e2ba: 13 77 f7 0f  	andi	a4, a4, 255
   1e2be: d5 b0        	j	0x1dba2 <_svfiprintf_r+0x3fe>
   1e2c0: 13 f7 0c 20  	andi	a4, s9, 512
   1e2c4: 19 cf        	beqz	a4, 0x1e2e2 <_svfiprintf_r+0xb3e>
   1e2c6: 9b 97 87 01  	slliw	a5, a5, 24
   1e2ca: 9b d7 87 41  	sraiw	a5, a5, 24
   1e2ce: 3e 87        	mv	a4, a5
   1e2d0: 63 52 07 ee  	bgez	a4, 0x1d9b4 <_svfiprintf_r+0x210>
   1e2d4: 6f f0 4f fb  	j	0x1da88 <_svfiprintf_r+0x2e4>
   1e2d8: 23 10 a7 01  	sh	s10, 0(a4)
   1e2dc: a1 06        	addi	a3, a3, 8
   1e2de: 6f f0 af d2  	j	0x1d808 <_svfiprintf_r+0x64>
   1e2e2: 3e 87        	mv	a4, a5
   1e2e4: 63 58 07 ec  	bgez	a4, 0x1d9b4 <_svfiprintf_r+0x210>
   1e2e8: 6f f0 0f fa  	j	0x1da88 <_svfiprintf_r+0x2e4>
   1e2ec: 02 17        	slli	a4, a4, 32
   1e2ee: 01 93        	srli	a4, a4, 32
   1e2f0: 4d b8        	j	0x1dba2 <_svfiprintf_r+0x3fe>
   1e2f2: 82 17        	slli	a5, a5, 32
   1e2f4: 81 93        	srli	a5, a5, 32
   1e2f6: ed b8        	j	0x1dbf0 <_svfiprintf_r+0x44c>
   1e2f8: 02 17        	slli	a4, a4, 32
   1e2fa: 01 93        	srli	a4, a4, 32
   1e2fc: 61 b5        	j	0x1e184 <_svfiprintf_r+0x9e0>
   1e2fe: 62 8e        	mv	t3, s8
   1e300: 62 83        	mv	t1, s8
   1e302: 01 4c        	mv	s8, zero
   1e304: 6f f0 0f fc  	j	0x1dac4 <_svfiprintf_r+0x320>
   1e308: f6 86        	mv	a3, t4
   1e30a: 93 07 00 03  	addi	a5, zero, 48
   1e30e: 6f f0 2f fa  	j	0x1dab0 <_svfiprintf_r+0x30c>
   1e312: 23 20 a7 01  	sw	s10, 0(a4)
   1e316: a1 06        	addi	a3, a3, 8
   1e318: 6f f0 0f cf  	j	0x1d808 <_svfiprintf_r+0x64>
   1e31c: b1 47        	addi	a5, zero, 12
   1e31e: 23 20 fb 00  	sw	a5, 0(s6)
   1e322: fd 5c        	addi	s9, zero, -1
   1e324: 6f f0 af ef  	j	0x1da1e <_svfiprintf_r+0x27a>
   1e328: 9c 42        	lw	a5, 0(a3)
   1e32a: a1 06        	addi	a3, a3, 8
   1e32c: 1b 8c 07 00  	sext.w	s8, a5
   1e330: 63 d3 07 00  	bgez	a5, 0x1e336 <_svfiprintf_r+0xb92>
   1e334: 7d 5c        	addi	s8, zero, -1
   1e336: 83 c7 1d 00  	lbu	a5, 1(s11)
   1e33a: ba 8d        	mv	s11, a4
   1e33c: 6f f0 cf d2  	j	0x1d868 <_svfiprintf_r+0xc4>

000000000001e340 <__swbuf_r>:
   1e340: 79 71        	addi	sp, sp, -48
   1e342: 22 f0        	sd	s0, 32(sp)
   1e344: 26 ec        	sd	s1, 24(sp)
   1e346: 4a e8        	sd	s2, 16(sp)
   1e348: 06 f4        	sd	ra, 40(sp)
   1e34a: 4e e4        	sd	s3, 8(sp)
   1e34c: 2a 89        	mv	s2, a0
   1e34e: ae 84        	mv	s1, a1
   1e350: 32 84        	mv	s0, a2
   1e352: 19 c1        	beqz	a0, 0x1e358 <__swbuf_r+0x18>
   1e354: 3c 49        	lw	a5, 80(a0)
   1e356: f9 cb        	beqz	a5, 0x1e42c <__swbuf_r+0xec>
   1e358: 1c 54        	lw	a5, 40(s0)
   1e35a: 03 17 04 01  	lh	a4, 16(s0)
   1e35e: 5c c4        	sw	a5, 12(s0)
   1e360: 93 77 87 00  	andi	a5, a4, 8
   1e364: b9 cb        	beqz	a5, 0x1e3ba <__swbuf_r+0x7a>
   1e366: 1c 6c        	ld	a5, 24(s0)
   1e368: a9 cb        	beqz	a5, 0x1e3ba <__swbuf_r+0x7a>
   1e36a: 93 f9 f4 0f  	andi	s3, s1, 255
   1e36e: 93 16 27 03  	slli	a3, a4, 50
   1e372: ce 84        	mv	s1, s3
   1e374: 63 d2 06 06  	bgez	a3, 0x1e3d8 <__swbuf_r+0x98>
   1e378: 18 60        	ld	a4, 0(s0)
   1e37a: 14 50        	lw	a3, 32(s0)
   1e37c: bb 07 f7 40  	subw	a5, a4, a5
   1e380: 63 dd d7 06  	bge	a5, a3, 0x1e3fa <__swbuf_r+0xba>
   1e384: 54 44        	lw	a3, 12(s0)
   1e386: 13 06 17 00  	addi	a2, a4, 1
   1e38a: 10 e0        	sd	a2, 0(s0)
   1e38c: fd 36        	addiw	a3, a3, -1
   1e38e: 54 c4        	sw	a3, 12(s0)
   1e390: 23 00 37 01  	sb	s3, 0(a4)
   1e394: 18 50        	lw	a4, 32(s0)
   1e396: 85 27        	addiw	a5, a5, 1
   1e398: 63 03 f7 08  	beq	a4, a5, 0x1e41e <__swbuf_r+0xde>
   1e39c: 83 57 04 01  	lhu	a5, 16(s0)
   1e3a0: 85 8b        	andi	a5, a5, 1
   1e3a2: 81 c7        	beqz	a5, 0x1e3aa <__swbuf_r+0x6a>
   1e3a4: a9 47        	addi	a5, zero, 10
   1e3a6: 63 8c f4 06  	beq	s1, a5, 0x1e41e <__swbuf_r+0xde>
   1e3aa: a2 70        	ld	ra, 40(sp)
   1e3ac: 02 74        	ld	s0, 32(sp)
   1e3ae: 42 69        	ld	s2, 16(sp)
   1e3b0: a2 69        	ld	s3, 8(sp)
   1e3b2: 26 85        	mv	a0, s1
   1e3b4: e2 64        	ld	s1, 24(sp)
   1e3b6: 45 61        	addi	sp, sp, 48
   1e3b8: 82 80        	ret
   1e3ba: a2 85        	mv	a1, s0
   1e3bc: 4a 85        	mv	a0, s2
   1e3be: ef 60 3f d8  	jal	0x15140 <__swsetup_r>
   1e3c2: 3d e1        	bnez	a0, 0x1e428 <__swbuf_r+0xe8>
   1e3c4: 03 17 04 01  	lh	a4, 16(s0)
   1e3c8: 93 f9 f4 0f  	andi	s3, s1, 255
   1e3cc: 1c 6c        	ld	a5, 24(s0)
   1e3ce: 93 16 27 03  	slli	a3, a4, 50
   1e3d2: ce 84        	mv	s1, s3
   1e3d4: e3 c2 06 fa  	bltz	a3, 0x1e378 <__swbuf_r+0x38>
   1e3d8: 83 26 c4 0a  	lw	a3, 172(s0)
   1e3dc: 09 66        	lui	a2, 2
   1e3de: 51 8f        	or	a4, a4, a2
   1e3e0: 79 76        	lui	a2, 1048574
   1e3e2: 7d 16        	addi	a2, a2, -1
   1e3e4: f1 8e        	and	a3, a3, a2
   1e3e6: 23 18 e4 00  	sh	a4, 16(s0)
   1e3ea: 18 60        	ld	a4, 0(s0)
   1e3ec: 23 26 d4 0a  	sw	a3, 172(s0)
   1e3f0: 14 50        	lw	a3, 32(s0)
   1e3f2: bb 07 f7 40  	subw	a5, a4, a5
   1e3f6: e3 c7 d7 f8  	blt	a5, a3, 0x1e384 <__swbuf_r+0x44>
   1e3fa: a2 85        	mv	a1, s0
   1e3fc: 4a 85        	mv	a0, s2
   1e3fe: ef 70 2f 86  	jal	0x15460 <_fflush_r>
   1e402: 1d e1        	bnez	a0, 0x1e428 <__swbuf_r+0xe8>
   1e404: 18 60        	ld	a4, 0(s0)
   1e406: 54 44        	lw	a3, 12(s0)
   1e408: 85 47        	addi	a5, zero, 1
   1e40a: 13 06 17 00  	addi	a2, a4, 1
   1e40e: fd 36        	addiw	a3, a3, -1
   1e410: 10 e0        	sd	a2, 0(s0)
   1e412: 54 c4        	sw	a3, 12(s0)
   1e414: 23 00 37 01  	sb	s3, 0(a4)
   1e418: 18 50        	lw	a4, 32(s0)
   1e41a: e3 11 f7 f8  	bne	a4, a5, 0x1e39c <__swbuf_r+0x5c>
   1e41e: a2 85        	mv	a1, s0
   1e420: 4a 85        	mv	a0, s2
   1e422: ef 70 ef 83  	jal	0x15460 <_fflush_r>
   1e426: 51 d1        	beqz	a0, 0x1e3aa <__swbuf_r+0x6a>
   1e428: fd 54        	addi	s1, zero, -1
   1e42a: 41 b7        	j	0x1e3aa <__swbuf_r+0x6a>
   1e42c: ef 70 6f b0  	jal	0x15732 <__sinit>
   1e430: 25 b7        	j	0x1e358 <__swbuf_r+0x18>

000000000001e432 <__swbuf>:
   1e432: aa 87        	mv	a5, a0
   1e434: 03 b5 81 93  	ld	a0, -1736(gp)
   1e438: 2e 86        	mv	a2, a1
   1e43a: be 85        	mv	a1, a5
   1e43c: 6f f0 5f f0  	j	0x1e340 <__swbuf_r>

000000000001e440 <_wcrtomb_r>:
   1e440: 79 71        	addi	sp, sp, -48
   1e442: 22 f0        	sd	s0, 32(sp)
   1e444: 26 ec        	sd	s1, 24(sp)
   1e446: b7 47 02 00  	lui	a5, 36
   1e44a: 06 f4        	sd	ra, 40(sp)
   1e44c: 83 b7 87 84  	ld	a5, -1976(a5)
   1e450: 2a 84        	mv	s0, a0
   1e452: b6 84        	mv	s1, a3
   1e454: 91 c9        	beqz	a1, 0x1e468 <_wcrtomb_r+0x28>
   1e456: 82 97        	jalr	a5
   1e458: fd 57        	addi	a5, zero, -1
   1e45a: 63 0d f5 00  	beq	a0, a5, 0x1e474 <_wcrtomb_r+0x34>
   1e45e: a2 70        	ld	ra, 40(sp)
   1e460: 02 74        	ld	s0, 32(sp)
   1e462: e2 64        	ld	s1, 24(sp)
   1e464: 45 61        	addi	sp, sp, 48
   1e466: 82 80        	ret
   1e468: 01 46        	mv	a2, zero
   1e46a: 8a 85        	mv	a1, sp
   1e46c: 82 97        	jalr	a5
   1e46e: fd 57        	addi	a5, zero, -1
   1e470: e3 17 f5 fe  	bne	a0, a5, 0x1e45e <_wcrtomb_r+0x1e>
   1e474: 23 a0 04 00  	sw	zero, 0(s1)
   1e478: 93 07 a0 08  	addi	a5, zero, 138
   1e47c: a2 70        	ld	ra, 40(sp)
   1e47e: 1c c0        	sw	a5, 0(s0)
   1e480: 02 74        	ld	s0, 32(sp)
   1e482: e2 64        	ld	s1, 24(sp)
   1e484: 45 61        	addi	sp, sp, 48
   1e486: 82 80        	ret

000000000001e488 <wcrtomb>:
   1e488: 79 71        	addi	sp, sp, -48
   1e48a: 22 f0        	sd	s0, 32(sp)
   1e48c: 26 ec        	sd	s1, 24(sp)
   1e48e: 06 f4        	sd	ra, 40(sp)
   1e490: 83 b4 81 93  	ld	s1, -1736(gp)
   1e494: b7 47 02 00  	lui	a5, 36
   1e498: 83 b7 87 84  	ld	a5, -1976(a5)
   1e49c: 32 84        	mv	s0, a2
   1e49e: 11 cd        	beqz	a0, 0x1e4ba <wcrtomb+0x32>
   1e4a0: 2e 86        	mv	a2, a1
   1e4a2: a2 86        	mv	a3, s0
   1e4a4: aa 85        	mv	a1, a0
   1e4a6: 26 85        	mv	a0, s1
   1e4a8: 82 97        	jalr	a5
   1e4aa: fd 57        	addi	a5, zero, -1
   1e4ac: 63 0f f5 00  	beq	a0, a5, 0x1e4ca <wcrtomb+0x42>
   1e4b0: a2 70        	ld	ra, 40(sp)
   1e4b2: 02 74        	ld	s0, 32(sp)
   1e4b4: e2 64        	ld	s1, 24(sp)
   1e4b6: 45 61        	addi	sp, sp, 48
   1e4b8: 82 80        	ret
   1e4ba: b2 86        	mv	a3, a2
   1e4bc: 8a 85        	mv	a1, sp
   1e4be: 01 46        	mv	a2, zero
   1e4c0: 26 85        	mv	a0, s1
   1e4c2: 82 97        	jalr	a5
   1e4c4: fd 57        	addi	a5, zero, -1
   1e4c6: e3 15 f5 fe  	bne	a0, a5, 0x1e4b0 <wcrtomb+0x28>
   1e4ca: 23 20 04 00  	sw	zero, 0(s0)
   1e4ce: a2 70        	ld	ra, 40(sp)
   1e4d0: 02 74        	ld	s0, 32(sp)
   1e4d2: 93 07 a0 08  	addi	a5, zero, 138
   1e4d6: 9c c0        	sw	a5, 0(s1)
   1e4d8: e2 64        	ld	s1, 24(sp)
   1e4da: 45 61        	addi	sp, sp, 48
   1e4dc: 82 80        	ret

000000000001e4de <abort>:
   1e4de: 41 11        	addi	sp, sp, -16
   1e4e0: 19 45        	addi	a0, zero, 6
   1e4e2: 06 e4        	sd	ra, 8(sp)
   1e4e4: ef 00 c0 23  	jal	0x1e720 <raise>
   1e4e8: 05 45        	addi	a0, zero, 1
   1e4ea: ef 00 40 46  	jal	0x1e94e <_exit>

000000000001e4ee <__register_exitproc>:
   1e4ee: 03 b7 81 83  	ld	a4, -1992(gp)
   1e4f2: 83 37 87 1f  	ld	a5, 504(a4)
   1e4f6: b1 c3        	beqz	a5, 0x1e53a <__register_exitproc+0x4c>
   1e4f8: 98 47        	lw	a4, 8(a5)
   1e4fa: 7d 48        	addi	a6, zero, 31
   1e4fc: 63 42 e8 06  	blt	a6, a4, 0x1e560 <__register_exitproc+0x72>
   1e500: 05 c5        	beqz	a0, 0x1e528 <__register_exitproc+0x3a>
   1e502: 13 18 37 00  	slli	a6, a4, 3
   1e506: 3e 98        	add	a6, a6, a5
   1e508: 23 38 c8 10  	sd	a2, 272(a6)
   1e50c: 83 a8 07 31  	lw	a7, 784(a5)
   1e510: 05 46        	addi	a2, zero, 1
   1e512: 3b 16 e6 00  	sllw	a2, a2, a4
   1e516: b3 e8 c8 00  	or	a7, a7, a2
   1e51a: 23 a8 17 31  	sw	a7, 784(a5)
   1e51e: 23 38 d8 20  	sd	a3, 528(a6)
   1e522: 89 46        	addi	a3, zero, 2
   1e524: 63 00 d5 02  	beq	a0, a3, 0x1e544 <__register_exitproc+0x56>
   1e528: 93 06 27 00  	addi	a3, a4, 2
   1e52c: 8e 06        	slli	a3, a3, 3
   1e52e: 05 27        	addiw	a4, a4, 1
   1e530: 98 c7        	sw	a4, 8(a5)
   1e532: b6 97        	add	a5, a5, a3
   1e534: 8c e3        	sd	a1, 0(a5)
   1e536: 01 45        	mv	a0, zero
   1e538: 82 80        	ret
   1e53a: 93 07 07 20  	addi	a5, a4, 512
   1e53e: 23 3c f7 1e  	sd	a5, 504(a4)
   1e542: 5d bf        	j	0x1e4f8 <__register_exitproc+0xa>
   1e544: 83 a6 47 31  	lw	a3, 788(a5)
   1e548: 01 45        	mv	a0, zero
   1e54a: d1 8e        	or	a3, a3, a2
   1e54c: 23 aa d7 30  	sw	a3, 788(a5)
   1e550: 93 06 27 00  	addi	a3, a4, 2
   1e554: 8e 06        	slli	a3, a3, 3
   1e556: 05 27        	addiw	a4, a4, 1
   1e558: 98 c7        	sw	a4, 8(a5)
   1e55a: b6 97        	add	a5, a5, a3
   1e55c: 8c e3        	sd	a1, 0(a5)
   1e55e: 82 80        	ret
   1e560: 7d 55        	addi	a0, zero, -1
   1e562: 82 80        	ret

000000000001e564 <_init_signal_r>:
   1e564: 83 37 85 51  	ld	a5, 1304(a0)
   1e568: 99 c3        	beqz	a5, 0x1e56e <_init_signal_r+0xa>
   1e56a: 01 45        	mv	a0, zero
   1e56c: 82 80        	ret
   1e56e: 41 11        	addi	sp, sp, -16
   1e570: 93 05 00 10  	addi	a1, zero, 256
   1e574: 22 e0        	sd	s0, 0(sp)
   1e576: 06 e4        	sd	ra, 8(sp)
   1e578: 2a 84        	mv	s0, a0
   1e57a: ef 90 7f fb  	jal	0x18530 <_malloc_r>
   1e57e: 23 3c a4 50  	sd	a0, 1304(s0)
   1e582: 0d c5        	beqz	a0, 0x1e5ac <_init_signal_r+0x48>
   1e584: 93 07 00 02  	addi	a5, zero, 32
   1e588: f3 26 20 c2  	csrr	a3, vlenb
   1e58c: 57 77 80 05  	<unknown>
   1e590: 57 3c 00 5e  	<unknown>
   1e594: 57 f7 87 05  	<unknown>
   1e598: 27 7c 05 02  	<unknown>
   1e59c: 99 8f        	sub	a5, a5, a4
   1e59e: 36 95        	add	a0, a0, a3
   1e5a0: f5 fb        	bnez	a5, 0x1e594 <_init_signal_r+0x30>
   1e5a2: 01 45        	mv	a0, zero
   1e5a4: a2 60        	ld	ra, 8(sp)
   1e5a6: 02 64        	ld	s0, 0(sp)
   1e5a8: 41 01        	addi	sp, sp, 16
   1e5aa: 82 80        	ret
   1e5ac: 7d 55        	addi	a0, zero, -1
   1e5ae: dd bf        	j	0x1e5a4 <_init_signal_r+0x40>

000000000001e5b0 <_signal_r>:
   1e5b0: 79 71        	addi	sp, sp, -48
   1e5b2: 26 ec        	sd	s1, 24(sp)
   1e5b4: 06 f4        	sd	ra, 40(sp)
   1e5b6: 22 f0        	sd	s0, 32(sp)
   1e5b8: fd 47        	addi	a5, zero, 31
   1e5ba: aa 84        	mv	s1, a0
   1e5bc: 63 ef b7 00  	bltu	a5, a1, 0x1e5da <_signal_r+0x2a>
   1e5c0: 83 37 85 51  	ld	a5, 1304(a0)
   1e5c4: 2e 84        	mv	s0, a1
   1e5c6: 91 cf        	beqz	a5, 0x1e5e2 <_signal_r+0x32>
   1e5c8: 0e 04        	slli	s0, s0, 3
   1e5ca: a2 97        	add	a5, a5, s0
   1e5cc: 88 63        	ld	a0, 0(a5)
   1e5ce: 90 e3        	sd	a2, 0(a5)
   1e5d0: a2 70        	ld	ra, 40(sp)
   1e5d2: 02 74        	ld	s0, 32(sp)
   1e5d4: e2 64        	ld	s1, 24(sp)
   1e5d6: 45 61        	addi	sp, sp, 48
   1e5d8: 82 80        	ret
   1e5da: d9 47        	addi	a5, zero, 22
   1e5dc: 1c c1        	sw	a5, 0(a0)
   1e5de: 7d 55        	addi	a0, zero, -1
   1e5e0: c5 bf        	j	0x1e5d0 <_signal_r+0x20>
   1e5e2: 93 05 00 10  	addi	a1, zero, 256
   1e5e6: 32 e4        	sd	a2, 8(sp)
   1e5e8: ef 90 9f f4  	jal	0x18530 <_malloc_r>
   1e5ec: 57 77 80 05  	<unknown>
   1e5f0: aa 87        	mv	a5, a0
   1e5f2: 23 bc a4 50  	sd	a0, 1304(s1)
   1e5f6: aa 86        	mv	a3, a0
   1e5f8: 22 66        	ld	a2, 8(sp)
   1e5fa: 13 07 00 02  	addi	a4, zero, 32
   1e5fe: 73 25 20 c2  	csrr	a0, vlenb
   1e602: 57 3c 00 5e  	<unknown>
   1e606: e1 df        	beqz	a5, 0x1e5de <_signal_r+0x2e>
   1e608: d7 75 87 05  	<unknown>
   1e60c: 27 fc 06 02  	<unknown>
   1e610: 0d 8f        	sub	a4, a4, a1
   1e612: aa 96        	add	a3, a3, a0
   1e614: 75 fb        	bnez	a4, 0x1e608 <_signal_r+0x58>
   1e616: 4d bf        	j	0x1e5c8 <_signal_r+0x18>

000000000001e618 <_raise_r>:
   1e618: 01 11        	addi	sp, sp, -32
   1e61a: 26 e4        	sd	s1, 8(sp)
   1e61c: 06 ec        	sd	ra, 24(sp)
   1e61e: 22 e8        	sd	s0, 16(sp)
   1e620: fd 47        	addi	a5, zero, 31
   1e622: aa 84        	mv	s1, a0
   1e624: 63 ee b7 04  	bltu	a5, a1, 0x1e680 <_raise_r+0x68>
   1e628: 83 37 85 51  	ld	a5, 1304(a0)
   1e62c: 2e 84        	mv	s0, a1
   1e62e: 95 c7        	beqz	a5, 0x1e65a <_raise_r+0x42>
   1e630: 13 97 35 00  	slli	a4, a1, 3
   1e634: ba 97        	add	a5, a5, a4
   1e636: 98 63        	ld	a4, 0(a5)
   1e638: 0d c3        	beqz	a4, 0x1e65a <_raise_r+0x42>
   1e63a: 85 46        	addi	a3, zero, 1
   1e63c: 63 09 d7 00  	beq	a4, a3, 0x1e64e <_raise_r+0x36>
   1e640: fd 56        	addi	a3, zero, -1
   1e642: 63 07 d7 02  	beq	a4, a3, 0x1e670 <_raise_r+0x58>
   1e646: 23 b0 07 00  	sd	zero, 0(a5)
   1e64a: 2e 85        	mv	a0, a1
   1e64c: 02 97        	jalr	a4
   1e64e: 01 45        	mv	a0, zero
   1e650: e2 60        	ld	ra, 24(sp)
   1e652: 42 64        	ld	s0, 16(sp)
   1e654: a2 64        	ld	s1, 8(sp)
   1e656: 05 61        	addi	sp, sp, 32
   1e658: 82 80        	ret
   1e65a: 26 85        	mv	a0, s1
   1e65c: ef 00 40 2c  	jal	0x1e920 <_getpid_r>
   1e660: 22 86        	mv	a2, s0
   1e662: 42 64        	ld	s0, 16(sp)
   1e664: e2 60        	ld	ra, 24(sp)
   1e666: aa 85        	mv	a1, a0
   1e668: 26 85        	mv	a0, s1
   1e66a: a2 64        	ld	s1, 8(sp)
   1e66c: 05 61        	addi	sp, sp, 32
   1e66e: a5 ac        	j	0x1e8e6 <_kill_r>
   1e670: e2 60        	ld	ra, 24(sp)
   1e672: 42 64        	ld	s0, 16(sp)
   1e674: d9 47        	addi	a5, zero, 22
   1e676: 1c c1        	sw	a5, 0(a0)
   1e678: a2 64        	ld	s1, 8(sp)
   1e67a: 05 45        	addi	a0, zero, 1
   1e67c: 05 61        	addi	sp, sp, 32
   1e67e: 82 80        	ret
   1e680: d9 47        	addi	a5, zero, 22
   1e682: 1c c1        	sw	a5, 0(a0)
   1e684: 7d 55        	addi	a0, zero, -1
   1e686: e9 b7        	j	0x1e650 <_raise_r+0x38>

000000000001e688 <__sigtramp_r>:
   1e688: fd 47        	addi	a5, zero, 31
   1e68a: 63 e9 b7 08  	bltu	a5, a1, 0x1e71c <__sigtramp_r+0x94>
   1e68e: 83 37 85 51  	ld	a5, 1304(a0)
   1e692: 01 11        	addi	sp, sp, -32
   1e694: 22 e8        	sd	s0, 16(sp)
   1e696: 26 e4        	sd	s1, 8(sp)
   1e698: 06 ec        	sd	ra, 24(sp)
   1e69a: 2e 84        	mv	s0, a1
   1e69c: aa 84        	mv	s1, a0
   1e69e: a1 c7        	beqz	a5, 0x1e6e6 <__sigtramp_r+0x5e>
   1e6a0: 13 17 34 00  	slli	a4, s0, 3
   1e6a4: ba 97        	add	a5, a5, a4
   1e6a6: 98 63        	ld	a4, 0(a5)
   1e6a8: 0d c3        	beqz	a4, 0x1e6ca <__sigtramp_r+0x42>
   1e6aa: fd 56        	addi	a3, zero, -1
   1e6ac: 63 0b d7 02  	beq	a4, a3, 0x1e6e2 <__sigtramp_r+0x5a>
   1e6b0: 85 46        	addi	a3, zero, 1
   1e6b2: 63 02 d7 02  	beq	a4, a3, 0x1e6d6 <__sigtramp_r+0x4e>
   1e6b6: 22 85        	mv	a0, s0
   1e6b8: 23 b0 07 00  	sd	zero, 0(a5)
   1e6bc: 02 97        	jalr	a4
   1e6be: 01 45        	mv	a0, zero
   1e6c0: e2 60        	ld	ra, 24(sp)
   1e6c2: 42 64        	ld	s0, 16(sp)
   1e6c4: a2 64        	ld	s1, 8(sp)
   1e6c6: 05 61        	addi	sp, sp, 32
   1e6c8: 82 80        	ret
   1e6ca: e2 60        	ld	ra, 24(sp)
   1e6cc: 42 64        	ld	s0, 16(sp)
   1e6ce: a2 64        	ld	s1, 8(sp)
   1e6d0: 05 45        	addi	a0, zero, 1
   1e6d2: 05 61        	addi	sp, sp, 32
   1e6d4: 82 80        	ret
   1e6d6: e2 60        	ld	ra, 24(sp)
   1e6d8: 42 64        	ld	s0, 16(sp)
   1e6da: a2 64        	ld	s1, 8(sp)
   1e6dc: 0d 45        	addi	a0, zero, 3
   1e6de: 05 61        	addi	sp, sp, 32
   1e6e0: 82 80        	ret
   1e6e2: 09 45        	addi	a0, zero, 2
   1e6e4: f1 bf        	j	0x1e6c0 <__sigtramp_r+0x38>
   1e6e6: 93 05 00 10  	addi	a1, zero, 256
   1e6ea: ef 90 7f e4  	jal	0x18530 <_malloc_r>
   1e6ee: 23 bc a4 50  	sd	a0, 1304(s1)
   1e6f2: aa 87        	mv	a5, a0
   1e6f4: 15 c1        	beqz	a0, 0x1e718 <__sigtramp_r+0x90>
   1e6f6: aa 86        	mv	a3, a0
   1e6f8: 13 07 00 02  	addi	a4, zero, 32
   1e6fc: f3 25 20 c2  	csrr	a1, vlenb
   1e700: 57 76 80 05  	<unknown>
   1e704: 57 3c 00 5e  	<unknown>
   1e708: 57 76 87 05  	<unknown>
   1e70c: 27 fc 06 02  	<unknown>
   1e710: 11 8f        	sub	a4, a4, a2
   1e712: ae 96        	add	a3, a3, a1
   1e714: 75 fb        	bnez	a4, 0x1e708 <__sigtramp_r+0x80>
   1e716: 69 b7        	j	0x1e6a0 <__sigtramp_r+0x18>
   1e718: 7d 55        	addi	a0, zero, -1
   1e71a: 5d b7        	j	0x1e6c0 <__sigtramp_r+0x38>
   1e71c: 7d 55        	addi	a0, zero, -1
   1e71e: 82 80        	ret

000000000001e720 <raise>:
   1e720: 01 11        	addi	sp, sp, -32
   1e722: 26 e4        	sd	s1, 8(sp)
   1e724: 06 ec        	sd	ra, 24(sp)
   1e726: 22 e8        	sd	s0, 16(sp)
   1e728: fd 47        	addi	a5, zero, 31
   1e72a: 83 b4 81 93  	ld	s1, -1736(gp)
   1e72e: 63 ed a7 04  	bltu	a5, a0, 0x1e788 <raise+0x68>
   1e732: 83 b7 84 51  	ld	a5, 1304(s1)
   1e736: 2a 84        	mv	s0, a0
   1e738: 8d c7        	beqz	a5, 0x1e762 <raise+0x42>
   1e73a: 13 17 35 00  	slli	a4, a0, 3
   1e73e: ba 97        	add	a5, a5, a4
   1e740: 98 63        	ld	a4, 0(a5)
   1e742: 05 c3        	beqz	a4, 0x1e762 <raise+0x42>
   1e744: 85 46        	addi	a3, zero, 1
   1e746: 63 08 d7 00  	beq	a4, a3, 0x1e756 <raise+0x36>
   1e74a: fd 56        	addi	a3, zero, -1
   1e74c: 63 06 d7 02  	beq	a4, a3, 0x1e778 <raise+0x58>
   1e750: 23 b0 07 00  	sd	zero, 0(a5)
   1e754: 02 97        	jalr	a4
   1e756: 01 45        	mv	a0, zero
   1e758: e2 60        	ld	ra, 24(sp)
   1e75a: 42 64        	ld	s0, 16(sp)
   1e75c: a2 64        	ld	s1, 8(sp)
   1e75e: 05 61        	addi	sp, sp, 32
   1e760: 82 80        	ret
   1e762: 26 85        	mv	a0, s1
   1e764: ef 00 c0 1b  	jal	0x1e920 <_getpid_r>
   1e768: 22 86        	mv	a2, s0
   1e76a: 42 64        	ld	s0, 16(sp)
   1e76c: e2 60        	ld	ra, 24(sp)
   1e76e: aa 85        	mv	a1, a0
   1e770: 26 85        	mv	a0, s1
   1e772: a2 64        	ld	s1, 8(sp)
   1e774: 05 61        	addi	sp, sp, 32
   1e776: 85 aa        	j	0x1e8e6 <_kill_r>
   1e778: e2 60        	ld	ra, 24(sp)
   1e77a: 42 64        	ld	s0, 16(sp)
   1e77c: d9 47        	addi	a5, zero, 22
   1e77e: 9c c0        	sw	a5, 0(s1)
   1e780: 05 45        	addi	a0, zero, 1
   1e782: a2 64        	ld	s1, 8(sp)
   1e784: 05 61        	addi	sp, sp, 32
   1e786: 82 80        	ret
   1e788: d9 47        	addi	a5, zero, 22
   1e78a: 9c c0        	sw	a5, 0(s1)
   1e78c: 7d 55        	addi	a0, zero, -1
   1e78e: e9 b7        	j	0x1e758 <raise+0x38>

000000000001e790 <signal>:
   1e790: 01 11        	addi	sp, sp, -32
   1e792: 4a e0        	sd	s2, 0(sp)
   1e794: 06 ec        	sd	ra, 24(sp)
   1e796: 22 e8        	sd	s0, 16(sp)
   1e798: 26 e4        	sd	s1, 8(sp)
   1e79a: fd 47        	addi	a5, zero, 31
   1e79c: 03 b9 81 93  	ld	s2, -1736(gp)
   1e7a0: 63 e2 a7 02  	bltu	a5, a0, 0x1e7c4 <signal+0x34>
   1e7a4: 2a 84        	mv	s0, a0
   1e7a6: 03 35 89 51  	ld	a0, 1304(s2)
   1e7aa: ae 84        	mv	s1, a1
   1e7ac: 0d c1        	beqz	a0, 0x1e7ce <signal+0x3e>
   1e7ae: 0e 04        	slli	s0, s0, 3
   1e7b0: b3 07 85 00  	add	a5, a0, s0
   1e7b4: 88 63        	ld	a0, 0(a5)
   1e7b6: 84 e3        	sd	s1, 0(a5)
   1e7b8: e2 60        	ld	ra, 24(sp)
   1e7ba: 42 64        	ld	s0, 16(sp)
   1e7bc: a2 64        	ld	s1, 8(sp)
   1e7be: 02 69        	ld	s2, 0(sp)
   1e7c0: 05 61        	addi	sp, sp, 32
   1e7c2: 82 80        	ret
   1e7c4: d9 47        	addi	a5, zero, 22
   1e7c6: 23 20 f9 00  	sw	a5, 0(s2)
   1e7ca: 7d 55        	addi	a0, zero, -1
   1e7cc: f5 b7        	j	0x1e7b8 <signal+0x28>
   1e7ce: 93 05 00 10  	addi	a1, zero, 256
   1e7d2: 4a 85        	mv	a0, s2
   1e7d4: ef 90 df d5  	jal	0x18530 <_malloc_r>
   1e7d8: d7 77 80 05  	<unknown>
   1e7dc: 23 3c a9 50  	sd	a0, 1304(s2)
   1e7e0: 2a 87        	mv	a4, a0
   1e7e2: 93 07 00 02  	addi	a5, zero, 32
   1e7e6: 73 26 20 c2  	csrr	a2, vlenb
   1e7ea: 57 3c 00 5e  	<unknown>
   1e7ee: 71 dd        	beqz	a0, 0x1e7ca <signal+0x3a>
   1e7f0: d7 f6 87 05  	<unknown>
   1e7f4: 27 7c 07 02  	<unknown>
   1e7f8: 95 8f        	sub	a5, a5, a3
   1e7fa: 32 97        	add	a4, a4, a2
   1e7fc: f5 fb        	bnez	a5, 0x1e7f0 <signal+0x60>
   1e7fe: 45 bf        	j	0x1e7ae <signal+0x1e>

000000000001e800 <_init_signal>:
   1e800: 41 11        	addi	sp, sp, -16
   1e802: 22 e0        	sd	s0, 0(sp)
   1e804: 03 b4 81 93  	ld	s0, -1736(gp)
   1e808: 06 e4        	sd	ra, 8(sp)
   1e80a: 83 37 84 51  	ld	a5, 1304(s0)
   1e80e: 91 c7        	beqz	a5, 0x1e81a <_init_signal+0x1a>
   1e810: 01 45        	mv	a0, zero
   1e812: a2 60        	ld	ra, 8(sp)
   1e814: 02 64        	ld	s0, 0(sp)
   1e816: 41 01        	addi	sp, sp, 16
   1e818: 82 80        	ret
   1e81a: 93 05 00 10  	addi	a1, zero, 256
   1e81e: 22 85        	mv	a0, s0
   1e820: ef 90 1f d1  	jal	0x18530 <_malloc_r>
   1e824: 23 3c a4 50  	sd	a0, 1304(s0)
   1e828: 0d c1        	beqz	a0, 0x1e84a <_init_signal+0x4a>
   1e82a: 93 07 00 02  	addi	a5, zero, 32
   1e82e: f3 26 20 c2  	csrr	a3, vlenb
   1e832: 57 77 80 05  	<unknown>
   1e836: 57 3c 00 5e  	<unknown>
   1e83a: 57 f7 87 05  	<unknown>
   1e83e: 27 7c 05 02  	<unknown>
   1e842: 99 8f        	sub	a5, a5, a4
   1e844: 36 95        	add	a0, a0, a3
   1e846: f5 fb        	bnez	a5, 0x1e83a <_init_signal+0x3a>
   1e848: e1 b7        	j	0x1e810 <_init_signal+0x10>
   1e84a: 7d 55        	addi	a0, zero, -1
   1e84c: d9 b7        	j	0x1e812 <_init_signal+0x12>

000000000001e84e <__sigtramp>:
   1e84e: 01 11        	addi	sp, sp, -32
   1e850: 26 e4        	sd	s1, 8(sp)
   1e852: 06 ec        	sd	ra, 24(sp)
   1e854: 22 e8        	sd	s0, 16(sp)
   1e856: fd 47        	addi	a5, zero, 31
   1e858: 83 b4 81 93  	ld	s1, -1736(gp)
   1e85c: 63 e3 a7 08  	bltu	a5, a0, 0x1e8e2 <__sigtramp+0x94>
   1e860: 83 b7 84 51  	ld	a5, 1304(s1)
   1e864: 2a 84        	mv	s0, a0
   1e866: a1 c7        	beqz	a5, 0x1e8ae <__sigtramp+0x60>
   1e868: 13 17 34 00  	slli	a4, s0, 3
   1e86c: ba 97        	add	a5, a5, a4
   1e86e: 98 63        	ld	a4, 0(a5)
   1e870: 0d c3        	beqz	a4, 0x1e892 <__sigtramp+0x44>
   1e872: fd 56        	addi	a3, zero, -1
   1e874: 63 0b d7 02  	beq	a4, a3, 0x1e8aa <__sigtramp+0x5c>
   1e878: 85 46        	addi	a3, zero, 1
   1e87a: 63 02 d7 02  	beq	a4, a3, 0x1e89e <__sigtramp+0x50>
   1e87e: 22 85        	mv	a0, s0
   1e880: 23 b0 07 00  	sd	zero, 0(a5)
   1e884: 02 97        	jalr	a4
   1e886: 01 45        	mv	a0, zero
   1e888: e2 60        	ld	ra, 24(sp)
   1e88a: 42 64        	ld	s0, 16(sp)
   1e88c: a2 64        	ld	s1, 8(sp)
   1e88e: 05 61        	addi	sp, sp, 32
   1e890: 82 80        	ret
   1e892: e2 60        	ld	ra, 24(sp)
   1e894: 42 64        	ld	s0, 16(sp)
   1e896: a2 64        	ld	s1, 8(sp)
   1e898: 05 45        	addi	a0, zero, 1
   1e89a: 05 61        	addi	sp, sp, 32
   1e89c: 82 80        	ret
   1e89e: e2 60        	ld	ra, 24(sp)
   1e8a0: 42 64        	ld	s0, 16(sp)
   1e8a2: a2 64        	ld	s1, 8(sp)
   1e8a4: 0d 45        	addi	a0, zero, 3
   1e8a6: 05 61        	addi	sp, sp, 32
   1e8a8: 82 80        	ret
   1e8aa: 09 45        	addi	a0, zero, 2
   1e8ac: f1 bf        	j	0x1e888 <__sigtramp+0x3a>
   1e8ae: 93 05 00 10  	addi	a1, zero, 256
   1e8b2: 26 85        	mv	a0, s1
   1e8b4: ef 90 df c7  	jal	0x18530 <_malloc_r>
   1e8b8: 23 bc a4 50  	sd	a0, 1304(s1)
   1e8bc: aa 87        	mv	a5, a0
   1e8be: 15 c1        	beqz	a0, 0x1e8e2 <__sigtramp+0x94>
   1e8c0: aa 86        	mv	a3, a0
   1e8c2: 13 07 00 02  	addi	a4, zero, 32
   1e8c6: f3 25 20 c2  	csrr	a1, vlenb
   1e8ca: 57 76 80 05  	<unknown>
   1e8ce: 57 3c 00 5e  	<unknown>
   1e8d2: 57 76 87 05  	<unknown>
   1e8d6: 27 fc 06 02  	<unknown>
   1e8da: 11 8f        	sub	a4, a4, a2
   1e8dc: ae 96        	add	a3, a3, a1
   1e8de: 75 fb        	bnez	a4, 0x1e8d2 <__sigtramp+0x84>
   1e8e0: 61 b7        	j	0x1e868 <__sigtramp+0x1a>
   1e8e2: 7d 55        	addi	a0, zero, -1
   1e8e4: 55 b7        	j	0x1e888 <__sigtramp+0x3a>

000000000001e8e6 <_kill_r>:
   1e8e6: 01 11        	addi	sp, sp, -32
   1e8e8: 2e 87        	mv	a4, a1
   1e8ea: 22 e8        	sd	s0, 16(sp)
   1e8ec: 26 e4        	sd	s1, 8(sp)
   1e8ee: 2a 84        	mv	s0, a0
   1e8f0: b2 85        	mv	a1, a2
   1e8f2: 3a 85        	mv	a0, a4
   1e8f4: 06 ec        	sd	ra, 24(sp)
   1e8f6: 23 ac 01 96  	sw	zero, -1672(gp)
   1e8fa: ef 00 00 0e  	jal	0x1e9da <_kill>
   1e8fe: fd 57        	addi	a5, zero, -1
   1e900: 63 07 f5 00  	beq	a0, a5, 0x1e90e <_kill_r+0x28>
   1e904: e2 60        	ld	ra, 24(sp)
   1e906: 42 64        	ld	s0, 16(sp)
   1e908: a2 64        	ld	s1, 8(sp)
   1e90a: 05 61        	addi	sp, sp, 32
   1e90c: 82 80        	ret
   1e90e: 83 a7 81 97  	lw	a5, -1672(gp)
   1e912: ed db        	beqz	a5, 0x1e904 <_kill_r+0x1e>
   1e914: e2 60        	ld	ra, 24(sp)
   1e916: 1c c0        	sw	a5, 0(s0)
   1e918: 42 64        	ld	s0, 16(sp)
   1e91a: a2 64        	ld	s1, 8(sp)
   1e91c: 05 61        	addi	sp, sp, 32
   1e91e: 82 80        	ret

000000000001e920 <_getpid_r>:
   1e920: 41 a8        	j	0x1e9b0 <_getpid>

000000000001e922 <_close>:
   1e922: 41 11        	addi	sp, sp, -16
   1e924: 06 e4        	sd	ra, 8(sp)
   1e926: 22 e0        	sd	s0, 0(sp)
   1e928: 93 08 90 03  	addi	a7, zero, 57
   1e92c: 73 00 00 00  	ecall	
   1e930: 2a 84        	mv	s0, a0
   1e932: 63 47 05 00  	bltz	a0, 0x1e940 <_close+0x1e>
   1e936: 01 25        	sext.w	a0, a0
   1e938: a2 60        	ld	ra, 8(sp)
   1e93a: 02 64        	ld	s0, 0(sp)
   1e93c: 41 01        	addi	sp, sp, 16
   1e93e: 82 80        	ret
   1e940: ef d0 7f ff  	jal	0x1c936 <__errno>
   1e944: 3b 04 80 40  	negw	s0, s0
   1e948: 00 c1        	sw	s0, 0(a0)
   1e94a: 7d 55        	addi	a0, zero, -1
   1e94c: f5 b7        	j	0x1e938 <_close+0x16>

000000000001e94e <_exit>:
   1e94e: 93 08 d0 05  	addi	a7, zero, 93
   1e952: 73 00 00 00  	ecall	
   1e956: 63 43 05 00  	bltz	a0, 0x1e95c <_exit+0xe>
   1e95a: 01 a0        	j	0x1e95a <_exit+0xc>
   1e95c: 41 11        	addi	sp, sp, -16
   1e95e: 22 e0        	sd	s0, 0(sp)
   1e960: 2a 84        	mv	s0, a0
   1e962: 06 e4        	sd	ra, 8(sp)
   1e964: 3b 04 80 40  	negw	s0, s0
   1e968: ef d0 ff fc  	jal	0x1c936 <__errno>
   1e96c: 00 c1        	sw	s0, 0(a0)
   1e96e: 01 a0        	j	0x1e96e <_exit+0x20>

000000000001e970 <_fstat>:
   1e970: 35 71        	addi	sp, sp, -160
   1e972: 26 e5        	sd	s1, 136(sp)
   1e974: 06 ed        	sd	ra, 152(sp)
   1e976: ae 84        	mv	s1, a1
   1e978: 22 e9        	sd	s0, 144(sp)
   1e97a: 93 08 00 05  	addi	a7, zero, 80
   1e97e: 8a 85        	mv	a1, sp
   1e980: 73 00 00 00  	ecall	
   1e984: 2a 84        	mv	s0, a0
   1e986: 63 4e 05 00  	bltz	a0, 0x1e9a2 <_fstat+0x32>
   1e98a: 1b 04 05 00  	sext.w	s0, a0
   1e98e: 26 85        	mv	a0, s1
   1e990: 8a 85        	mv	a1, sp
   1e992: ef 00 40 13  	jal	0x1eac6 <_conv_stat>
   1e996: ea 60        	ld	ra, 152(sp)
   1e998: 22 85        	mv	a0, s0
   1e99a: 4a 64        	ld	s0, 144(sp)
   1e99c: aa 64        	ld	s1, 136(sp)
   1e99e: 0d 61        	addi	sp, sp, 160
   1e9a0: 82 80        	ret
   1e9a2: 3b 04 80 40  	negw	s0, s0
   1e9a6: ef d0 1f f9  	jal	0x1c936 <__errno>
   1e9aa: 00 c1        	sw	s0, 0(a0)
   1e9ac: 7d 54        	addi	s0, zero, -1
   1e9ae: c5 b7        	j	0x1e98e <_fstat+0x1e>

000000000001e9b0 <_getpid>:
   1e9b0: 05 45        	addi	a0, zero, 1
   1e9b2: 82 80        	ret

000000000001e9b4 <_isatty>:
   1e9b4: 19 71        	addi	sp, sp, -128
   1e9b6: 2c 00        	addi	a1, sp, 8
   1e9b8: 86 fc        	sd	ra, 120(sp)
   1e9ba: ef f0 7f fb  	jal	0x1e970 <_fstat>
   1e9be: fd 57        	addi	a5, zero, -1
   1e9c0: 63 09 f5 00  	beq	a0, a5, 0x1e9d2 <_isatty+0x1e>
   1e9c4: 32 45        	lw	a0, 12(sp)
   1e9c6: e6 70        	ld	ra, 120(sp)
   1e9c8: 1b 55 d5 40  	sraiw	a0, a0, 13
   1e9cc: 05 89        	andi	a0, a0, 1
   1e9ce: 09 61        	addi	sp, sp, 128
   1e9d0: 82 80        	ret
   1e9d2: e6 70        	ld	ra, 120(sp)
   1e9d4: 01 45        	mv	a0, zero
   1e9d6: 09 61        	addi	sp, sp, 128
   1e9d8: 82 80        	ret

000000000001e9da <_kill>:
   1e9da: 41 11        	addi	sp, sp, -16
   1e9dc: 06 e4        	sd	ra, 8(sp)
   1e9de: ef d0 9f f5  	jal	0x1c936 <__errno>
   1e9e2: a2 60        	ld	ra, 8(sp)
   1e9e4: d9 47        	addi	a5, zero, 22
   1e9e6: 1c c1        	sw	a5, 0(a0)
   1e9e8: 7d 55        	addi	a0, zero, -1
   1e9ea: 41 01        	addi	sp, sp, 16
   1e9ec: 82 80        	ret

000000000001e9ee <_lseek>:
   1e9ee: 41 11        	addi	sp, sp, -16
   1e9f0: 06 e4        	sd	ra, 8(sp)
   1e9f2: 22 e0        	sd	s0, 0(sp)
   1e9f4: 93 08 e0 03  	addi	a7, zero, 62
   1e9f8: 73 00 00 00  	ecall	
   1e9fc: 2a 84        	mv	s0, a0
   1e9fe: 63 47 05 00  	bltz	a0, 0x1ea0c <_lseek+0x1e>
   1ea02: a2 60        	ld	ra, 8(sp)
   1ea04: 22 85        	mv	a0, s0
   1ea06: 02 64        	ld	s0, 0(sp)
   1ea08: 41 01        	addi	sp, sp, 16
   1ea0a: 82 80        	ret
   1ea0c: 3b 04 80 40  	negw	s0, s0
   1ea10: ef d0 7f f2  	jal	0x1c936 <__errno>
   1ea14: 00 c1        	sw	s0, 0(a0)
   1ea16: 7d 54        	addi	s0, zero, -1
   1ea18: ed b7        	j	0x1ea02 <_lseek+0x14>

000000000001ea1a <_read>:
   1ea1a: 41 11        	addi	sp, sp, -16
   1ea1c: 06 e4        	sd	ra, 8(sp)
   1ea1e: 22 e0        	sd	s0, 0(sp)
   1ea20: 93 08 f0 03  	addi	a7, zero, 63
   1ea24: 73 00 00 00  	ecall	
   1ea28: 2a 84        	mv	s0, a0
   1ea2a: 63 47 05 00  	bltz	a0, 0x1ea38 <_read+0x1e>
   1ea2e: a2 60        	ld	ra, 8(sp)
   1ea30: 22 85        	mv	a0, s0
   1ea32: 02 64        	ld	s0, 0(sp)
   1ea34: 41 01        	addi	sp, sp, 16
   1ea36: 82 80        	ret
   1ea38: 3b 04 80 40  	negw	s0, s0
   1ea3c: ef d0 bf ef  	jal	0x1c936 <__errno>
   1ea40: 00 c1        	sw	s0, 0(a0)
   1ea42: 7d 54        	addi	s0, zero, -1
   1ea44: ed b7        	j	0x1ea2e <_read+0x14>

000000000001ea46 <_sbrk>:
   1ea46: 03 b7 01 98  	ld	a4, -1664(gp)
   1ea4a: 41 11        	addi	sp, sp, -16
   1ea4c: 06 e4        	sd	ra, 8(sp)
   1ea4e: aa 87        	mv	a5, a0
   1ea50: 01 ef        	bnez	a4, 0x1ea68 <_sbrk+0x22>
   1ea52: 93 08 60 0d  	addi	a7, zero, 214
   1ea56: 01 45        	mv	a0, zero
   1ea58: 73 00 00 00  	ecall	
   1ea5c: 7d 56        	addi	a2, zero, -1
   1ea5e: 2a 87        	mv	a4, a0
   1ea60: 63 05 c5 02  	beq	a0, a2, 0x1ea8a <_sbrk+0x44>
   1ea64: 23 b0 a1 98  	sd	a0, -1664(gp)
   1ea68: 93 08 60 0d  	addi	a7, zero, 214
   1ea6c: 33 85 e7 00  	add	a0, a5, a4
   1ea70: 73 00 00 00  	ecall	
   1ea74: 03 b7 01 98  	ld	a4, -1664(gp)
   1ea78: ba 97        	add	a5, a5, a4
   1ea7a: 63 18 f5 00  	bne	a0, a5, 0x1ea8a <_sbrk+0x44>
   1ea7e: a2 60        	ld	ra, 8(sp)
   1ea80: 23 b0 a1 98  	sd	a0, -1664(gp)
   1ea84: 3a 85        	mv	a0, a4
   1ea86: 41 01        	addi	sp, sp, 16
   1ea88: 82 80        	ret
   1ea8a: ef d0 df ea  	jal	0x1c936 <__errno>
   1ea8e: a2 60        	ld	ra, 8(sp)
   1ea90: b1 47        	addi	a5, zero, 12
   1ea92: 1c c1        	sw	a5, 0(a0)
   1ea94: 7d 55        	addi	a0, zero, -1
   1ea96: 41 01        	addi	sp, sp, 16
   1ea98: 82 80        	ret

000000000001ea9a <_write>:
   1ea9a: 41 11        	addi	sp, sp, -16
   1ea9c: 06 e4        	sd	ra, 8(sp)
   1ea9e: 22 e0        	sd	s0, 0(sp)
   1eaa0: 93 08 00 04  	addi	a7, zero, 64
   1eaa4: 73 00 00 00  	ecall	
   1eaa8: 2a 84        	mv	s0, a0
   1eaaa: 63 47 05 00  	bltz	a0, 0x1eab8 <_write+0x1e>
   1eaae: a2 60        	ld	ra, 8(sp)
   1eab0: 22 85        	mv	a0, s0
   1eab2: 02 64        	ld	s0, 0(sp)
   1eab4: 41 01        	addi	sp, sp, 16
   1eab6: 82 80        	ret
   1eab8: 3b 04 80 40  	negw	s0, s0
   1eabc: ef d0 bf e7  	jal	0x1c936 <__errno>
   1eac0: 00 c1        	sw	s0, 0(a0)
   1eac2: 7d 54        	addi	s0, zero, -1
   1eac4: ed b7        	j	0x1eaae <_write+0x14>

000000000001eac6 <_conv_stat>:
   1eac6: 83 b3 05 00  	ld	t2, 0(a1)
   1eaca: 83 b2 85 00  	ld	t0, 8(a1)
   1eace: 83 af 05 01  	lw	t6, 16(a1)
   1ead2: 03 af 45 01  	lw	t5, 20(a1)
   1ead6: 83 ae 85 01  	lw	t4, 24(a1)
   1eada: 03 ae c5 01  	lw	t3, 28(a1)
   1eade: 03 b3 05 02  	ld	t1, 32(a1)
   1eae2: 83 b8 05 03  	ld	a7, 48(a1)
   1eae6: 03 b8 05 04  	ld	a6, 64(a1)
   1eaea: 90 5d        	lw	a2, 56(a1)
   1eaec: b4 65        	ld	a3, 72(a1)
   1eaee: b8 6d        	ld	a4, 88(a1)
   1eaf0: bc 75        	ld	a5, 104(a1)
   1eaf2: 23 10 75 00  	sh	t2, 0(a0)
   1eaf6: 23 11 55 00  	sh	t0, 2(a0)
   1eafa: 23 22 f5 01  	sw	t6, 4(a0)
   1eafe: 23 14 e5 01  	sh	t5, 8(a0)
   1eb02: 23 15 d5 01  	sh	t4, 10(a0)
   1eb06: 23 16 c5 01  	sh	t3, 12(a0)
   1eb0a: 23 17 65 00  	sh	t1, 14(a0)
   1eb0e: 23 38 15 01  	sd	a7, 16(a0)
   1eb12: 23 38 05 05  	sd	a6, 80(a0)
   1eb16: 30 e5        	sd	a2, 72(a0)
   1eb18: 14 ed        	sd	a3, 24(a0)
   1eb1a: 18 f5        	sd	a4, 40(a0)
   1eb1c: 1c fd        	sd	a5, 56(a0)
   1eb1e: 82 80        	ret

000000000001eb20 <__netf2>:
   1eb20: f3 27 20 00  	frrm	a5
   1eb24: a1 67        	lui	a5, 8
   1eb26: 13 d8 05 03  	srli	a6, a1, 48
   1eb2a: fd 17        	addi	a5, a5, -1
   1eb2c: 7d 5e        	addi	t3, zero, -1
   1eb2e: 13 57 0e 01  	srli	a4, t3, 16
   1eb32: 13 d3 06 03  	srli	t1, a3, 48
   1eb36: 33 78 f8 00  	and	a6, a6, a5
   1eb3a: b3 f8 e5 00  	and	a7, a1, a4
   1eb3e: 33 73 f3 00  	and	t1, t1, a5
   1eb42: 75 8f        	and	a4, a4, a3
   1eb44: fd 91        	srli	a1, a1, 63
   1eb46: fd 92        	srli	a3, a3, 63
   1eb48: 63 1f f8 00  	bne	a6, a5, 0x1eb66 <__netf2+0x46>
   1eb4c: 33 e5 a8 00  	or	a0, a7, a0
   1eb50: 0d e9        	bnez	a0, 0x1eb82 <__netf2+0x62>
   1eb52: 85 47        	addi	a5, zero, 1
   1eb54: 63 17 03 01  	bne	t1, a6, 0x1eb62 <__netf2+0x42>
   1eb58: 59 8e        	or	a2, a2, a4
   1eb5a: 19 ea        	bnez	a2, 0x1eb70 <__netf2+0x50>
   1eb5c: 95 8d        	sub	a1, a1, a3
   1eb5e: b3 37 b0 00  	snez	a5, a1
   1eb62: 3e 85        	mv	a0, a5
   1eb64: 05 a8        	j	0x1eb94 <__netf2+0x74>
   1eb66: 63 18 f3 02  	bne	t1, a5, 0x1eb96 <__netf2+0x76>
   1eb6a: 59 8e        	or	a2, a2, a4
   1eb6c: 85 47        	addi	a5, zero, 1
   1eb6e: 75 da        	beqz	a2, 0x1eb62 <__netf2+0x42>
   1eb70: fd 57        	addi	a5, zero, -1
   1eb72: c5 83        	srli	a5, a5, 17
   1eb74: 05 45        	addi	a0, zero, 1
   1eb76: 63 ef e7 00  	bltu	a5, a4, 0x1eb94 <__netf2+0x74>
   1eb7a: 73 60 18 00  	csrsi	fflags, 16
   1eb7e: 05 45        	addi	a0, zero, 1
   1eb80: 82 80        	ret
   1eb82: 13 5e 1e 01  	srli	t3, t3, 17
   1eb86: e3 7a 1e ff  	bgeu	t3, a7, 0x1eb7a <__netf2+0x5a>
   1eb8a: 05 45        	addi	a0, zero, 1
   1eb8c: 63 14 03 01  	bne	t1, a6, 0x1eb94 <__netf2+0x74>
   1eb90: 59 8e        	or	a2, a2, a4
   1eb92: 79 fe        	bnez	a2, 0x1eb70 <__netf2+0x50>
   1eb94: 82 80        	ret
   1eb96: 85 47        	addi	a5, zero, 1
   1eb98: e3 15 68 fc  	bne	a6, t1, 0x1eb62 <__netf2+0x42>
   1eb9c: e3 93 e8 fc  	bne	a7, a4, 0x1eb62 <__netf2+0x42>
   1eba0: e3 11 c5 fc  	bne	a0, a2, 0x1eb62 <__netf2+0x42>
   1eba4: 63 89 d5 00  	beq	a1, a3, 0x1ebb6 <__netf2+0x96>
   1eba8: e3 1d 08 fa  	bnez	a6, 0x1eb62 <__netf2+0x42>
   1ebac: b3 e8 a8 00  	or	a7, a7, a0
   1ebb0: b3 37 10 01  	snez	a5, a7
   1ebb4: 7d b7        	j	0x1eb62 <__netf2+0x42>
   1ebb6: 81 47        	mv	a5, zero
   1ebb8: 6d b7        	j	0x1eb62 <__netf2+0x42>

000000000001ebba <__gttf2>:
   1ebba: 2a 83        	mv	t1, a0
   1ebbc: f3 27 20 00  	frrm	a5
   1ebc0: 21 6e        	lui	t3, 8
   1ebc2: 93 d8 05 03  	srli	a7, a1, 48
   1ebc6: 7d 1e        	addi	t3, t3, -1
   1ebc8: 7d 57        	addi	a4, zero, -1
   1ebca: 41 83        	srli	a4, a4, 16
   1ebcc: 93 d7 06 03  	srli	a5, a3, 48
   1ebd0: b3 f8 c8 01  	and	a7, a7, t3
   1ebd4: 33 f8 e5 00  	and	a6, a1, a4
   1ebd8: 33 f5 c7 01  	and	a0, a5, t3
   1ebdc: 75 8f        	and	a4, a4, a3
   1ebde: fd 91        	srli	a1, a1, 63
   1ebe0: fd 92        	srli	a3, a3, 63
   1ebe2: 63 92 c8 03  	bne	a7, t3, 0x1ec06 <__gttf2+0x4c>
   1ebe6: 33 68 68 00  	or	a6, a6, t1
   1ebea: 63 14 08 08  	bnez	a6, 0x1ec72 <__gttf2+0xb8>
   1ebee: 63 06 15 01  	beq	a0, a7, 0x1ebfa <__gttf2+0x40>
   1ebf2: 05 45        	addi	a0, zero, 1
   1ebf4: b5 cd        	beqz	a1, 0x1ec70 <__gttf2+0xb6>
   1ebf6: 7d 55        	addi	a0, zero, -1
   1ebf8: 82 80        	ret
   1ebfa: 51 8f        	or	a4, a4, a2
   1ebfc: 3d eb        	bnez	a4, 0x1ec72 <__gttf2+0xb8>
   1ebfe: e3 9a d5 fe  	bne	a1, a3, 0x1ebf2 <__gttf2+0x38>
   1ec02: 01 45        	mv	a0, zero
   1ec04: 82 80        	ret
   1ec06: 63 14 c5 03  	bne	a0, t3, 0x1ec2e <__gttf2+0x74>
   1ec0a: 51 8f        	or	a4, a4, a2
   1ec0c: 3d e3        	bnez	a4, 0x1ec72 <__gttf2+0xb8>
   1ec0e: 63 88 08 00  	beqz	a7, 0x1ec1e <__gttf2+0x64>
   1ec12: e3 90 d5 fe  	bne	a1, a3, 0x1ebf2 <__gttf2+0x38>
   1ec16: 7d 55        	addi	a0, zero, -1
   1ec18: a1 cd        	beqz	a1, 0x1ec70 <__gttf2+0xb6>
   1ec1a: 2e 85        	mv	a0, a1
   1ec1c: 82 80        	ret
   1ec1e: 33 68 68 00  	or	a6, a6, t1
   1ec22: e3 18 08 fe  	bnez	a6, 0x1ec12 <__gttf2+0x58>
   1ec26: 7d 55        	addi	a0, zero, -1
   1ec28: a1 c6        	beqz	a3, 0x1ec70 <__gttf2+0xb6>
   1ec2a: 36 85        	mv	a0, a3
   1ec2c: 82 80        	ret
   1ec2e: 63 83 08 02  	beqz	a7, 0x1ec54 <__gttf2+0x9a>
   1ec32: 61 d1        	beqz	a0, 0x1ebf2 <__gttf2+0x38>
   1ec34: e3 9f d5 fa  	bne	a1, a3, 0x1ebf2 <__gttf2+0x38>
   1ec38: e3 4d 15 fb  	blt	a0, a7, 0x1ebf2 <__gttf2+0x38>
   1ec3c: e3 cd a8 fc  	blt	a7, a0, 0x1ec16 <__gttf2+0x5c>
   1ec40: e3 69 07 fb  	bltu	a4, a6, 0x1ebf2 <__gttf2+0x38>
   1ec44: e3 19 e8 fc  	bne	a6, a4, 0x1ec16 <__gttf2+0x5c>
   1ec48: e3 65 66 fa  	bltu	a2, t1, 0x1ebf2 <__gttf2+0x38>
   1ec4c: 01 45        	mv	a0, zero
   1ec4e: 63 71 c3 02  	bgeu	t1, a2, 0x1ec70 <__gttf2+0xb6>
   1ec52: d1 b7        	j	0x1ec16 <__gttf2+0x5c>
   1ec54: b3 67 68 00  	or	a5, a6, t1
   1ec58: 19 c1        	beqz	a0, 0x1ec5e <__gttf2+0xa4>
   1ec5a: f1 d7        	beqz	a5, 0x1ec26 <__gttf2+0x6c>
   1ec5c: 5d bf        	j	0x1ec12 <__gttf2+0x58>
   1ec5e: b3 68 c7 00  	or	a7, a4, a2
   1ec62: 63 86 08 00  	beqz	a7, 0x1ec6e <__gttf2+0xb4>
   1ec66: e1 d3        	beqz	a5, 0x1ec26 <__gttf2+0x6c>
   1ec68: e3 8c d5 fc  	beq	a1, a3, 0x1ec40 <__gttf2+0x86>
   1ec6c: 59 b7        	j	0x1ebf2 <__gttf2+0x38>
   1ec6e: d1 f3        	bnez	a5, 0x1ebf2 <__gttf2+0x38>
   1ec70: 82 80        	ret
   1ec72: 73 60 18 00  	csrsi	fflags, 16
   1ec76: 79 55        	addi	a0, zero, -2
   1ec78: 82 80        	ret

000000000001ec7a <__lttf2>:
   1ec7a: 2a 83        	mv	t1, a0
   1ec7c: f3 27 20 00  	frrm	a5
   1ec80: 21 6e        	lui	t3, 8
   1ec82: 93 d8 05 03  	srli	a7, a1, 48
   1ec86: 7d 1e        	addi	t3, t3, -1
   1ec88: 7d 57        	addi	a4, zero, -1
   1ec8a: 41 83        	srli	a4, a4, 16
   1ec8c: 93 d7 06 03  	srli	a5, a3, 48
   1ec90: b3 f8 c8 01  	and	a7, a7, t3
   1ec94: 33 f8 e5 00  	and	a6, a1, a4
   1ec98: 33 f5 c7 01  	and	a0, a5, t3
   1ec9c: 75 8f        	and	a4, a4, a3
   1ec9e: fd 91        	srli	a1, a1, 63
   1eca0: fd 92        	srli	a3, a3, 63
   1eca2: 63 92 c8 03  	bne	a7, t3, 0x1ecc6 <__lttf2+0x4c>
   1eca6: 33 68 68 00  	or	a6, a6, t1
   1ecaa: 63 14 08 08  	bnez	a6, 0x1ed32 <__lttf2+0xb8>
   1ecae: 63 06 15 01  	beq	a0, a7, 0x1ecba <__lttf2+0x40>
   1ecb2: 05 45        	addi	a0, zero, 1
   1ecb4: b5 cd        	beqz	a1, 0x1ed30 <__lttf2+0xb6>
   1ecb6: 7d 55        	addi	a0, zero, -1
   1ecb8: 82 80        	ret
   1ecba: 51 8f        	or	a4, a4, a2
   1ecbc: 3d eb        	bnez	a4, 0x1ed32 <__lttf2+0xb8>
   1ecbe: e3 9a d5 fe  	bne	a1, a3, 0x1ecb2 <__lttf2+0x38>
   1ecc2: 01 45        	mv	a0, zero
   1ecc4: 82 80        	ret
   1ecc6: 63 14 c5 03  	bne	a0, t3, 0x1ecee <__lttf2+0x74>
   1ecca: 51 8f        	or	a4, a4, a2
   1eccc: 3d e3        	bnez	a4, 0x1ed32 <__lttf2+0xb8>
   1ecce: 63 88 08 00  	beqz	a7, 0x1ecde <__lttf2+0x64>
   1ecd2: e3 90 d5 fe  	bne	a1, a3, 0x1ecb2 <__lttf2+0x38>
   1ecd6: 7d 55        	addi	a0, zero, -1
   1ecd8: a1 cd        	beqz	a1, 0x1ed30 <__lttf2+0xb6>
   1ecda: 2e 85        	mv	a0, a1
   1ecdc: 82 80        	ret
   1ecde: 33 68 68 00  	or	a6, a6, t1
   1ece2: e3 18 08 fe  	bnez	a6, 0x1ecd2 <__lttf2+0x58>
   1ece6: 7d 55        	addi	a0, zero, -1
   1ece8: a1 c6        	beqz	a3, 0x1ed30 <__lttf2+0xb6>
   1ecea: 36 85        	mv	a0, a3
   1ecec: 82 80        	ret
   1ecee: 63 83 08 02  	beqz	a7, 0x1ed14 <__lttf2+0x9a>
   1ecf2: 61 d1        	beqz	a0, 0x1ecb2 <__lttf2+0x38>
   1ecf4: e3 9f d5 fa  	bne	a1, a3, 0x1ecb2 <__lttf2+0x38>
   1ecf8: e3 4d 15 fb  	blt	a0, a7, 0x1ecb2 <__lttf2+0x38>
   1ecfc: e3 cd a8 fc  	blt	a7, a0, 0x1ecd6 <__lttf2+0x5c>
   1ed00: e3 69 07 fb  	bltu	a4, a6, 0x1ecb2 <__lttf2+0x38>
   1ed04: e3 19 e8 fc  	bne	a6, a4, 0x1ecd6 <__lttf2+0x5c>
   1ed08: e3 65 66 fa  	bltu	a2, t1, 0x1ecb2 <__lttf2+0x38>
   1ed0c: 01 45        	mv	a0, zero
   1ed0e: 63 71 c3 02  	bgeu	t1, a2, 0x1ed30 <__lttf2+0xb6>
   1ed12: d1 b7        	j	0x1ecd6 <__lttf2+0x5c>
   1ed14: b3 67 68 00  	or	a5, a6, t1
   1ed18: 19 c1        	beqz	a0, 0x1ed1e <__lttf2+0xa4>
   1ed1a: f1 d7        	beqz	a5, 0x1ece6 <__lttf2+0x6c>
   1ed1c: 5d bf        	j	0x1ecd2 <__lttf2+0x58>
   1ed1e: b3 68 c7 00  	or	a7, a4, a2
   1ed22: 63 86 08 00  	beqz	a7, 0x1ed2e <__lttf2+0xb4>
   1ed26: e1 d3        	beqz	a5, 0x1ece6 <__lttf2+0x6c>
   1ed28: e3 8c d5 fc  	beq	a1, a3, 0x1ed00 <__lttf2+0x86>
   1ed2c: 59 b7        	j	0x1ecb2 <__lttf2+0x38>
   1ed2e: d1 f3        	bnez	a5, 0x1ecb2 <__lttf2+0x38>
   1ed30: 82 80        	ret
   1ed32: 73 60 18 00  	csrsi	fflags, 16
   1ed36: 09 45        	addi	a0, zero, 2
   1ed38: 82 80        	ret

000000000001ed3a <__multf3>:
   1ed3a: 1d 71        	addi	sp, sp, -96
   1ed3c: a2 e8        	sd	s0, 80(sp)
   1ed3e: 52 f8        	sd	s4, 48(sp)
   1ed40: 62 e8        	sd	s8, 16(sp)
   1ed42: 86 ec        	sd	ra, 88(sp)
   1ed44: a6 e4        	sd	s1, 72(sp)
   1ed46: ca e0        	sd	s2, 64(sp)
   1ed48: 4e fc        	sd	s3, 56(sp)
   1ed4a: 56 f4        	sd	s5, 40(sp)
   1ed4c: 5a f0        	sd	s6, 32(sp)
   1ed4e: 5e ec        	sd	s7, 24(sp)
   1ed50: 66 e4        	sd	s9, 8(sp)
   1ed52: 2a 84        	mv	s0, a0
   1ed54: 32 8c        	mv	s8, a2
   1ed56: 36 8a        	mv	s4, a3
   1ed58: f3 29 20 00  	frrm	s3
   1ed5c: 21 67        	lui	a4, 8
   1ed5e: 93 d7 05 03  	srli	a5, a1, 48
   1ed62: 7d 17        	addi	a4, a4, -1
   1ed64: 13 99 05 01  	slli	s2, a1, 16
   1ed68: f9 8f        	and	a5, a5, a4
   1ed6a: 81 29        	sext.w	s3, s3
   1ed6c: 13 59 09 01  	srli	s2, s2, 16
   1ed70: 93 da f5 03  	srli	s5, a1, 63
   1ed74: 95 c7        	beqz	a5, 0x1eda0 <__multf3+0x66>
   1ed76: 9b 8c 07 00  	sext.w	s9, a5
   1ed7a: 63 84 e7 08  	beq	a5, a4, 0x1ee02 <__multf3+0xc8>
   1ed7e: 0e 09        	slli	s2, s2, 3
   1ed80: 13 57 d5 03  	srli	a4, a0, 61
   1ed84: 33 67 27 01  	or	a4, a4, s2
   1ed88: f1 7c        	lui	s9, 1048572
   1ed8a: 05 49        	addi	s2, zero, 1
   1ed8c: 4e 19        	slli	s2, s2, 51
   1ed8e: 85 0c        	addi	s9, s9, 1
   1ed90: 33 69 27 01  	or	s2, a4, s2
   1ed94: 13 1b 35 00  	slli	s6, a0, 3
   1ed98: be 9c        	add	s9, s9, a5
   1ed9a: 81 4b        	mv	s7, zero
   1ed9c: 81 44        	mv	s1, zero
   1ed9e: b5 a8        	j	0x1ee1a <__multf3+0xe0>
   1eda0: 33 6b a9 00  	or	s6, s2, a0
   1eda4: 63 0b 0b 0e  	beqz	s6, 0x1ee9a <__multf3+0x160>
   1eda8: 63 0e 09 02  	beqz	s2, 0x1ede4 <__multf3+0xaa>
   1edac: 4a 85        	mv	a0, s2
   1edae: ef 10 a0 30  	jal	0x200b8 <__clzdi2>
   1edb2: 1b 07 05 00  	sext.w	a4, a0
   1edb6: 93 07 17 ff  	addi	a5, a4, -15
   1edba: 93 06 d0 03  	addi	a3, zero, 61
   1edbe: 1b 86 37 00  	addiw	a2, a5, 3
   1edc2: bb 87 f6 40  	subw	a5, a3, a5
   1edc6: 33 19 c9 00  	sll	s2, s2, a2
   1edca: b3 57 f4 00  	srl	a5, s0, a5
   1edce: b3 e7 27 01  	or	a5, a5, s2
   1edd2: 33 19 c4 00  	sll	s2, s0, a2
   1edd6: f1 7c        	lui	s9, 1048572
   1edd8: c5 0c        	addi	s9, s9, 17
   1edda: 4a 8b        	mv	s6, s2
   1eddc: b3 8c ec 40  	sub	s9, s9, a4
   1ede0: 3e 89        	mv	s2, a5
   1ede2: 65 bf        	j	0x1ed9a <__multf3+0x60>
   1ede4: ef 10 40 2d  	jal	0x200b8 <__clzdi2>
   1ede8: 1b 07 05 04  	addiw	a4, a0, 64
   1edec: 93 07 17 ff  	addi	a5, a4, -15
   1edf0: 93 06 c0 03  	addi	a3, zero, 60
   1edf4: e3 d3 f6 fc  	bge	a3, a5, 0x1edba <__multf3+0x80>
   1edf8: 9b 87 37 fc  	addiw	a5, a5, -61
   1edfc: b3 17 f4 00  	sll	a5, s0, a5
   1ee00: d9 bf        	j	0x1edd6 <__multf3+0x9c>
   1ee02: 33 6b a9 00  	or	s6, s2, a0
   1ee06: 63 0e 0b 08  	beqz	s6, 0x1eea2 <__multf3+0x168>
   1ee0a: 85 44        	addi	s1, zero, 1
   1ee0c: be 14        	slli	s1, s1, 47
   1ee0e: b3 34 99 00  	sltu	s1, s2, s1
   1ee12: 9b 94 44 00  	slliw	s1, s1, 4
   1ee16: 2a 8b        	mv	s6, a0
   1ee18: 8d 4b        	addi	s7, zero, 3
   1ee1a: a1 65        	lui	a1, 8
   1ee1c: 13 56 0a 03  	srli	a2, s4, 48
   1ee20: fd 56        	addi	a3, zero, -1
   1ee22: fd 15        	addi	a1, a1, -1
   1ee24: 13 d4 06 01  	srli	s0, a3, 16
   1ee28: 6d 8e        	and	a2, a2, a1
   1ee2a: 33 74 8a 00  	and	s0, s4, s0
   1ee2e: e2 87        	mv	a5, s8
   1ee30: 13 5a fa 03  	srli	s4, s4, 63
   1ee34: 35 ca        	beqz	a2, 0x1eea8 <__multf3+0x16e>
   1ee36: 1b 07 06 00  	sext.w	a4, a2
   1ee3a: 63 05 b6 0c  	beq	a2, a1, 0x1ef04 <__multf3+0x1ca>
   1ee3e: 0e 04        	slli	s0, s0, 3
   1ee40: 13 57 dc 03  	srli	a4, s8, 61
   1ee44: 41 8f        	or	a4, a4, s0
   1ee46: 05 44        	addi	s0, zero, 1
   1ee48: 4e 14        	slli	s0, s0, 51
   1ee4a: 59 8c        	or	s0, s0, a4
   1ee4c: 71 77        	lui	a4, 1048572
   1ee4e: 05 07        	addi	a4, a4, 1
   1ee50: 93 17 3c 00  	slli	a5, s8, 3
   1ee54: 32 97        	add	a4, a4, a2
   1ee56: 81 48        	mv	a7, zero
   1ee58: 93 96 2b 00  	slli	a3, s7, 2
   1ee5c: 33 08 97 01  	add	a6, a4, s9
   1ee60: b3 e6 16 01  	or	a3, a3, a7
   1ee64: 29 47        	addi	a4, zero, 10
   1ee66: 33 c6 4a 01  	xor	a2, s5, s4
   1ee6a: 93 05 18 00  	addi	a1, a6, 1
   1ee6e: 63 41 d7 2c  	blt	a4, a3, 0x1f130 <__multf3+0x3f6>
   1ee72: 09 47        	addi	a4, zero, 2
   1ee74: 63 4a d7 0a  	blt	a4, a3, 0x1ef28 <__multf3+0x1ee>
   1ee78: fd 16        	addi	a3, a3, -1
   1ee7a: 05 47        	addi	a4, zero, 1
   1ee7c: 63 65 d7 0c  	bltu	a4, a3, 0x1ef46 <__multf3+0x20c>
   1ee80: c6 8b        	mv	s7, a7
   1ee82: 09 47        	addi	a4, zero, 2
   1ee84: 63 80 eb 4c  	beq	s7, a4, 0x1f344 <__multf3+0x60a>
   1ee88: 0d 47        	addi	a4, zero, 3
   1ee8a: 63 82 eb 4c  	beq	s7, a4, 0x1f34e <__multf3+0x614>
   1ee8e: 05 47        	addi	a4, zero, 1
   1ee90: 63 92 eb 26  	bne	s7, a4, 0x1f0f4 <__multf3+0x3ba>
   1ee94: 01 47        	mv	a4, zero
   1ee96: 01 45        	mv	a0, zero
   1ee98: 79 a1        	j	0x1f326 <__multf3+0x5ec>
   1ee9a: 01 49        	mv	s2, zero
   1ee9c: 81 4c        	mv	s9, zero
   1ee9e: 85 4b        	addi	s7, zero, 1
   1eea0: f5 bd        	j	0x1ed9c <__multf3+0x62>
   1eea2: 01 49        	mv	s2, zero
   1eea4: 89 4b        	addi	s7, zero, 2
   1eea6: dd bd        	j	0x1ed9c <__multf3+0x62>
   1eea8: b3 67 8c 00  	or	a5, s8, s0
   1eeac: ad c7        	beqz	a5, 0x1ef16 <__multf3+0x1dc>
   1eeae: 1d c8        	beqz	s0, 0x1eee4 <__multf3+0x1aa>
   1eeb0: 22 85        	mv	a0, s0
   1eeb2: ef 10 60 20  	jal	0x200b8 <__clzdi2>
   1eeb6: 01 25        	sext.w	a0, a0
   1eeb8: 93 07 15 ff  	addi	a5, a0, -15
   1eebc: 13 07 d0 03  	addi	a4, zero, 61
   1eec0: 1b 86 37 00  	addiw	a2, a5, 3
   1eec4: bb 07 f7 40  	subw	a5, a4, a5
   1eec8: 33 14 c4 00  	sll	s0, s0, a2
   1eecc: b3 57 fc 00  	srl	a5, s8, a5
   1eed0: b3 e6 87 00  	or	a3, a5, s0
   1eed4: 33 14 cc 00  	sll	s0, s8, a2
   1eed8: 71 77        	lui	a4, 1048572
   1eeda: 45 07        	addi	a4, a4, 17
   1eedc: a2 87        	mv	a5, s0
   1eede: 09 8f        	sub	a4, a4, a0
   1eee0: 36 84        	mv	s0, a3
   1eee2: 95 bf        	j	0x1ee56 <__multf3+0x11c>
   1eee4: 62 85        	mv	a0, s8
   1eee6: ef 10 20 1d  	jal	0x200b8 <__clzdi2>
   1eeea: 1b 05 05 04  	addiw	a0, a0, 64
   1eeee: 93 07 15 ff  	addi	a5, a0, -15
   1eef2: 13 07 c0 03  	addi	a4, zero, 60
   1eef6: e3 53 f7 fc  	bge	a4, a5, 0x1eebc <__multf3+0x182>
   1eefa: 9b 87 37 fc  	addiw	a5, a5, -61
   1eefe: b3 16 fc 00  	sll	a3, s8, a5
   1ef02: d9 bf        	j	0x1eed8 <__multf3+0x19e>
   1ef04: 33 66 8c 00  	or	a2, s8, s0
   1ef08: 01 ce        	beqz	a2, 0x1ef20 <__multf3+0x1e6>
   1ef0a: c5 82        	srli	a3, a3, 17
   1ef0c: 8d 48        	addi	a7, zero, 3
   1ef0e: e3 e5 86 f4  	bltu	a3, s0, 0x1ee58 <__multf3+0x11e>
   1ef12: c1 44        	addi	s1, zero, 16
   1ef14: 91 b7        	j	0x1ee58 <__multf3+0x11e>
   1ef16: 01 44        	mv	s0, zero
   1ef18: 81 47        	mv	a5, zero
   1ef1a: 01 47        	mv	a4, zero
   1ef1c: 85 48        	addi	a7, zero, 1
   1ef1e: 2d bf        	j	0x1ee58 <__multf3+0x11e>
   1ef20: 01 44        	mv	s0, zero
   1ef22: 81 47        	mv	a5, zero
   1ef24: 89 48        	addi	a7, zero, 2
   1ef26: 0d bf        	j	0x1ee58 <__multf3+0x11e>
   1ef28: 05 47        	addi	a4, zero, 1
   1ef2a: b3 16 d7 00  	sll	a3, a4, a3
   1ef2e: 13 f5 06 53  	andi	a0, a3, 1328
   1ef32: 63 18 05 20  	bnez	a0, 0x1f142 <__multf3+0x408>
   1ef36: 13 f3 06 24  	andi	t1, a3, 576
   1ef3a: 63 1f 03 3e  	bnez	t1, 0x1f338 <__multf3+0x5fe>
   1ef3e: 93 f6 86 08  	andi	a3, a3, 136
   1ef42: 63 9e 06 1e  	bnez	a3, 0x1f13e <__multf3+0x404>
   1ef46: fd 56        	addi	a3, zero, -1
   1ef48: 81 92        	srli	a3, a3, 32
   1ef4a: 33 8f 87 00  	add	t5, a5, s0
   1ef4e: 13 55 0b 02  	srli	a0, s6, 32
   1ef52: 13 d3 07 02  	srli	t1, a5, 32
   1ef56: b3 0f 69 01  	add	t6, s2, s6
   1ef5a: b3 3e ff 00  	sltu	t4, t5, a5
   1ef5e: 33 7b db 00  	and	s6, s6, a3
   1ef62: f5 8f        	and	a5, a5, a3
   1ef64: b3 82 67 03  	mul	t0, a5, s6
   1ef68: 33 be 2f 01  	sltu	t3, t6, s2
   1ef6c: f2 88        	mv	a7, t3
   1ef6e: 76 87        	mv	a4, t4
   1ef70: b3 07 f5 02  	mul	a5, a0, a5
   1ef74: b3 06 65 02  	mul	a3, a0, t1
   1ef78: 13 d5 02 02  	srli	a0, t0, 32
   1ef7c: 33 03 63 03  	mul	t1, t1, s6
   1ef80: 3e 93        	add	t1, t1, a5
   1ef82: 1a 95        	add	a0, a0, t1
   1ef84: 63 75 f5 00  	bgeu	a0, a5, 0x1ef8e <__multf3+0x254>
   1ef88: 85 47        	addi	a5, zero, 1
   1ef8a: 82 17        	slli	a5, a5, 32
   1ef8c: be 96        	add	a3, a3, a5
   1ef8e: fd 57        	addi	a5, zero, -1
   1ef90: 81 93        	srli	a5, a5, 32
   1ef92: 13 53 05 02  	srli	t1, a0, 32
   1ef96: 7d 8d        	and	a0, a0, a5
   1ef98: b3 f2 f2 00  	and	t0, t0, a5
   1ef9c: 02 15        	slli	a0, a0, 32
   1ef9e: b3 f3 ff 00  	and	t2, t6, a5
   1efa2: 36 93        	add	t1, t1, a3
   1efa4: b3 77 ff 00  	and	a5, t5, a5
   1efa8: 93 56 0f 02  	srli	a3, t5, 32
   1efac: 16 95        	add	a0, a0, t0
   1efae: 93 d2 0f 02  	srli	t0, t6, 32
   1efb2: b3 8a 77 02  	mul	s5, a5, t2
   1efb6: b3 87 f2 02  	mul	a5, t0, a5
   1efba: 13 da 0a 02  	srli	s4, s5, 32
   1efbe: b3 82 d2 02  	mul	t0, t0, a3
   1efc2: b3 86 76 02  	mul	a3, a3, t2
   1efc6: be 96        	add	a3, a3, a5
   1efc8: 36 9a        	add	s4, s4, a3
   1efca: 63 75 fa 00  	bgeu	s4, a5, 0x1efd4 <__multf3+0x29a>
   1efce: 85 47        	addi	a5, zero, 1
   1efd0: 82 17        	slli	a5, a5, 32
   1efd2: be 92        	add	t0, t0, a5
   1efd4: fd 57        	addi	a5, zero, -1
   1efd6: 81 93        	srli	a5, a5, 32
   1efd8: 93 53 0a 02  	srli	t2, s4, 32
   1efdc: 33 7a fa 00  	and	s4, s4, a5
   1efe0: b3 fa fa 00  	and	s5, s5, a5
   1efe4: 02 1a        	slli	s4, s4, 32
   1efe6: 93 56 04 02  	srli	a3, s0, 32
   1efea: 56 9a        	add	s4, s4, s5
   1efec: 93 5a 09 02  	srli	s5, s2, 32
   1eff0: 33 79 f9 00  	and	s2, s2, a5
   1eff4: e1 8f        	and	a5, a5, s0
   1eff6: 33 0b f9 02  	mul	s6, s2, a5
   1effa: b3 87 fa 02  	mul	a5, s5, a5
   1effe: 13 54 0b 02  	srli	s0, s6, 32
   1f002: b3 8a da 02  	mul	s5, s5, a3
   1f006: b3 86 26 03  	mul	a3, a3, s2
   1f00a: be 96        	add	a3, a3, a5
   1f00c: 36 94        	add	s0, s0, a3
   1f00e: 63 75 f4 00  	bgeu	s0, a5, 0x1f018 <__multf3+0x2de>
   1f012: 85 47        	addi	a5, zero, 1
   1f014: 82 17        	slli	a5, a5, 32
   1f016: be 9a        	add	s5, s5, a5
   1f018: 7d 59        	addi	s2, zero, -1
   1f01a: 13 59 09 02  	srli	s2, s2, 32
   1f01e: 93 56 04 02  	srli	a3, s0, 32
   1f022: b3 77 24 01  	and	a5, s0, s2
   1f026: 33 07 e0 40  	neg	a4, a4
   1f02a: 33 04 6a 00  	add	s0, s4, t1
   1f02e: 33 77 f7 01  	and	a4, a4, t6
   1f032: b3 08 10 41  	neg	a7, a7
   1f036: b3 3f 44 01  	sltu	t6, s0, s4
   1f03a: b3 f8 e8 01  	and	a7, a7, t5
   1f03e: 33 8f f3 01  	add	t5, t2, t6
   1f042: 16 9f        	add	t5, t5, t0
   1f044: 33 7e de 01  	and	t3, t3, t4
   1f048: b3 3f ff 01  	sltu	t6, t5, t6
   1f04c: 7a 97        	add	a4, a4, t5
   1f04e: f2 9f        	add	t6, t6, t3
   1f050: 33 3f e7 01  	sltu	t5, a4, t5
   1f054: 46 97        	add	a4, a4, a7
   1f056: 7e 9f        	add	t5, t5, t6
   1f058: b3 38 17 01  	sltu	a7, a4, a7
   1f05c: b3 0e a4 40  	sub	t4, s0, a0
   1f060: 33 0e 67 40  	sub	t3, a4, t1
   1f064: 82 17        	slli	a5, a5, 32
   1f066: 33 7b 2b 01  	and	s6, s6, s2
   1f06a: fa 98        	add	a7, a7, t5
   1f06c: 33 3f d4 01  	sltu	t5, s0, t4
   1f070: b3 3f c7 01  	sltu	t6, a4, t3
   1f074: d6 96        	add	a3, a3, s5
   1f076: 33 0e ee 41  	sub	t3, t3, t5
   1f07a: da 97        	add	a5, a5, s6
   1f07c: 01 4f        	mv	t5, zero
   1f07e: 63 76 d4 01  	bgeu	s0, t4, 0x1f08a <__multf3+0x350>
   1f082: 33 03 e3 40  	sub	t1, t1, a4
   1f086: 13 3f 13 00  	seqz	t5, t1
   1f08a: 33 6f ff 01  	or	t5, t5, t6
   1f08e: b3 8f fe 40  	sub	t6, t4, a5
   1f092: b6 98        	add	a7, a7, a3
   1f094: 33 b7 fe 01  	sltu	a4, t4, t6
   1f098: b3 06 de 40  	sub	a3, t3, a3
   1f09c: 33 3e de 00  	sltu	t3, t3, a3
   1f0a0: 33 87 e7 40  	sub	a4, a5, a4
   1f0a4: 01 43        	mv	t1, zero
   1f0a6: 63 f4 fe 01  	bgeu	t4, t6, 0x1f0ae <__multf3+0x374>
   1f0aa: 13 b3 16 00  	seqz	t1, a3
   1f0ae: ba 96        	add	a3, a3, a4
   1f0b0: 33 b7 f6 00  	sltu	a4, a3, a5
   1f0b4: 46 97        	add	a4, a4, a7
   1f0b6: 33 07 e7 41  	sub	a4, a4, t5
   1f0ba: 33 63 c3 01  	or	t1, t1, t3
   1f0be: 33 07 67 40  	sub	a4, a4, t1
   1f0c2: 93 97 df 00  	slli	a5, t6, 13
   1f0c6: 36 07        	slli	a4, a4, 13
   1f0c8: 13 d4 36 03  	srli	s0, a3, 51
   1f0cc: c9 8f        	or	a5, a5, a0
   1f0ce: 59 8c        	or	s0, s0, a4
   1f0d0: b3 37 f0 00  	snez	a5, a5
   1f0d4: 13 d5 3f 03  	srli	a0, t6, 51
   1f0d8: 51 93        	srli	a4, a4, 52
   1f0da: c9 8f        	or	a5, a5, a0
   1f0dc: b6 06        	slli	a3, a3, 13
   1f0de: 05 8b        	andi	a4, a4, 1
   1f0e0: d5 8f        	or	a5, a5, a3
   1f0e2: 3d c3        	beqz	a4, 0x1f148 <__multf3+0x40e>
   1f0e4: 13 d7 17 00  	srli	a4, a5, 1
   1f0e8: 85 8b        	andi	a5, a5, 1
   1f0ea: 5d 8f        	or	a4, a4, a5
   1f0ec: 93 17 f4 03  	slli	a5, s0, 63
   1f0f0: d9 8f        	or	a5, a5, a4
   1f0f2: 05 80        	srli	s0, s0, 1
   1f0f4: 91 66        	lui	a3, 4
   1f0f6: fd 16        	addi	a3, a3, -1
   1f0f8: ae 96        	add	a3, a3, a1
   1f0fa: 63 50 d0 0e  	blez	a3, 0x1f1da <__multf3+0x4a0>
   1f0fe: 13 f7 77 00  	andi	a4, a5, 7
   1f102: 31 cb        	beqz	a4, 0x1f156 <__multf3+0x41c>
   1f104: 09 47        	addi	a4, zero, 2
   1f106: 93 e4 14 00  	ori	s1, s1, 1
   1f10a: 63 85 e9 04  	beq	s3, a4, 0x1f154 <__multf3+0x41a>
   1f10e: 0d 47        	addi	a4, zero, 3
   1f110: 63 8e e9 02  	beq	s3, a4, 0x1f14c <__multf3+0x412>
   1f114: 63 91 09 04  	bnez	s3, 0x1f156 <__multf3+0x41c>
   1f118: 13 f7 f7 00  	andi	a4, a5, 15
   1f11c: 11 45        	addi	a0, zero, 4
   1f11e: 63 0c a7 02  	beq	a4, a0, 0x1f156 <__multf3+0x41c>
   1f122: 13 87 47 00  	addi	a4, a5, 4
   1f126: 33 35 f7 00  	sltu	a0, a4, a5
   1f12a: 2a 94        	add	s0, s0, a0
   1f12c: ba 87        	mv	a5, a4
   1f12e: 25 a0        	j	0x1f156 <__multf3+0x41c>
   1f130: 3d 47        	addi	a4, zero, 15
   1f132: 63 8e e6 20  	beq	a3, a4, 0x1f34e <__multf3+0x614>
   1f136: 2d 47        	addi	a4, zero, 11
   1f138: 56 86        	mv	a2, s5
   1f13a: 63 94 e6 00  	bne	a3, a4, 0x1f142 <__multf3+0x408>
   1f13e: 52 86        	mv	a2, s4
   1f140: 81 b3        	j	0x1ee80 <__multf3+0x146>
   1f142: 4a 84        	mv	s0, s2
   1f144: da 87        	mv	a5, s6
   1f146: 35 bb        	j	0x1ee82 <__multf3+0x148>
   1f148: c2 85        	mv	a1, a6
   1f14a: 6d b7        	j	0x1f0f4 <__multf3+0x3ba>
   1f14c: 09 e6        	bnez	a2, 0x1f156 <__multf3+0x41c>
   1f14e: 13 87 87 00  	addi	a4, a5, 8
   1f152: d1 bf        	j	0x1f126 <__multf3+0x3ec>
   1f154: 6d fe        	bnez	a2, 0x1f14e <__multf3+0x414>
   1f156: 13 57 44 03  	srli	a4, s0, 52
   1f15a: 05 8b        	andi	a4, a4, 1
   1f15c: 19 c7        	beqz	a4, 0x1f16a <__multf3+0x430>
   1f15e: 7d 57        	addi	a4, zero, -1
   1f160: 52 17        	slli	a4, a4, 52
   1f162: 7d 17        	addi	a4, a4, -1
   1f164: 91 66        	lui	a3, 4
   1f166: 79 8c        	and	s0, s0, a4
   1f168: ae 96        	add	a3, a3, a1
   1f16a: a1 65        	lui	a1, 8
   1f16c: 13 87 e5 ff  	addi	a4, a1, -2
   1f170: 63 40 d7 04  	blt	a4, a3, 0x1f1b0 <__multf3+0x476>
   1f174: 8d 83        	srli	a5, a5, 3
   1f176: 13 15 d4 03  	slli	a0, s0, 61
   1f17a: 5d 8d        	or	a0, a0, a5
   1f17c: 13 57 34 00  	srli	a4, s0, 3
   1f180: 1b 16 f6 00  	slliw	a2, a2, 15
   1f184: 93 15 07 01  	slli	a1, a4, 16
   1f188: d1 8e        	or	a3, a3, a2
   1f18a: c2 16        	slli	a3, a3, 48
   1f18c: c1 81        	srli	a1, a1, 16
   1f18e: d5 8d        	or	a1, a1, a3
   1f190: 99 c0        	beqz	s1, 0x1f196 <__multf3+0x45c>
   1f192: 73 a0 14 00  	csrs	fflags, s1
   1f196: e6 60        	ld	ra, 88(sp)
   1f198: 46 64        	ld	s0, 80(sp)
   1f19a: a6 64        	ld	s1, 72(sp)
   1f19c: 06 69        	ld	s2, 64(sp)
   1f19e: e2 79        	ld	s3, 56(sp)
   1f1a0: 42 7a        	ld	s4, 48(sp)
   1f1a2: a2 7a        	ld	s5, 40(sp)
   1f1a4: 02 7b        	ld	s6, 32(sp)
   1f1a6: e2 6b        	ld	s7, 24(sp)
   1f1a8: 42 6c        	ld	s8, 16(sp)
   1f1aa: a2 6c        	ld	s9, 8(sp)
   1f1ac: 25 61        	addi	sp, sp, 96
   1f1ae: 82 80        	ret
   1f1b0: 89 47        	addi	a5, zero, 2
   1f1b2: 63 82 f9 02  	beq	s3, a5, 0x1f1d6 <__multf3+0x49c>
   1f1b6: 8d 47        	addi	a5, zero, 3
   1f1b8: 63 87 f9 00  	beq	s3, a5, 0x1f1c6 <__multf3+0x48c>
   1f1bc: 63 86 09 00  	beqz	s3, 0x1f1c8 <__multf3+0x48e>
   1f1c0: 7d 55        	addi	a0, zero, -1
   1f1c2: ba 86        	mv	a3, a4
   1f1c4: 29 a0        	j	0x1f1ce <__multf3+0x494>
   1f1c6: 6d fe        	bnez	a2, 0x1f1c0 <__multf3+0x486>
   1f1c8: 01 45        	mv	a0, zero
   1f1ca: 93 86 f5 ff  	addi	a3, a1, -1
   1f1ce: 93 e4 54 00  	ori	s1, s1, 5
   1f1d2: 2a 87        	mv	a4, a0
   1f1d4: 75 b7        	j	0x1f180 <__multf3+0x446>
   1f1d6: 6d d6        	beqz	a2, 0x1f1c0 <__multf3+0x486>
   1f1d8: c5 bf        	j	0x1f1c8 <__multf3+0x48e>
   1f1da: c5 e2        	bnez	a3, 0x1f27a <__multf3+0x540>
   1f1dc: 93 f6 77 00  	andi	a3, a5, 7
   1f1e0: 22 87        	mv	a4, s0
   1f1e2: 8d ce        	beqz	a3, 0x1f21c <__multf3+0x4e2>
   1f1e4: 89 46        	addi	a3, zero, 2
   1f1e6: 93 e4 14 00  	ori	s1, s1, 1
   1f1ea: 63 88 d9 02  	beq	s3, a3, 0x1f21a <__multf3+0x4e0>
   1f1ee: 8d 46        	addi	a3, zero, 3
   1f1f0: 63 81 d9 02  	beq	s3, a3, 0x1f212 <__multf3+0x4d8>
   1f1f4: 63 94 09 02  	bnez	s3, 0x1f21c <__multf3+0x4e2>
   1f1f8: 93 f6 f7 00  	andi	a3, a5, 15
   1f1fc: 91 45        	addi	a1, zero, 4
   1f1fe: 63 8f b6 00  	beq	a3, a1, 0x1f21c <__multf3+0x4e2>
   1f202: 13 b7 c7 ff  	sltiu	a4, a5, -4
   1f206: 13 47 17 00  	xori	a4, a4, 1
   1f20a: 02 17        	slli	a4, a4, 32
   1f20c: 01 93        	srli	a4, a4, 32
   1f20e: 22 97        	add	a4, a4, s0
   1f210: 31 a0        	j	0x1f21c <__multf3+0x4e2>
   1f212: 09 e6        	bnez	a2, 0x1f21c <__multf3+0x4e2>
   1f214: 13 b7 87 ff  	sltiu	a4, a5, -8
   1f218: fd b7        	j	0x1f206 <__multf3+0x4cc>
   1f21a: 6d fe        	bnez	a2, 0x1f214 <__multf3+0x4da>
   1f21c: 51 93        	srli	a4, a4, 52
   1f21e: 13 47 17 00  	xori	a4, a4, 1
   1f222: 93 75 17 00  	andi	a1, a4, 1
   1f226: 05 47        	addi	a4, zero, 1
   1f228: 93 06 00 04  	addi	a3, zero, 64
   1f22c: 99 9e        	subw	a3, a3, a4
   1f22e: 33 15 d4 00  	sll	a0, s0, a3
   1f232: 33 d8 e7 00  	srl	a6, a5, a4
   1f236: b3 96 d7 00  	sll	a3, a5, a3
   1f23a: 33 65 05 01  	or	a0, a0, a6
   1f23e: b3 36 d0 00  	snez	a3, a3
   1f242: 55 8d        	or	a0, a0, a3
   1f244: 33 57 e4 00  	srl	a4, s0, a4
   1f248: 93 77 75 00  	andi	a5, a0, 7
   1f24c: bd cf        	beqz	a5, 0x1f2ca <__multf3+0x590>
   1f24e: 89 47        	addi	a5, zero, 2
   1f250: 93 e4 14 00  	ori	s1, s1, 1
   1f254: 63 8a f9 06  	beq	s3, a5, 0x1f2c8 <__multf3+0x58e>
   1f258: 8d 47        	addi	a5, zero, 3
   1f25a: 63 83 f9 06  	beq	s3, a5, 0x1f2c0 <__multf3+0x586>
   1f25e: 63 96 09 06  	bnez	s3, 0x1f2ca <__multf3+0x590>
   1f262: 93 77 f5 00  	andi	a5, a0, 15
   1f266: 91 46        	addi	a3, zero, 4
   1f268: 63 81 d7 06  	beq	a5, a3, 0x1f2ca <__multf3+0x590>
   1f26c: 93 07 45 00  	addi	a5, a0, 4
   1f270: b3 b6 a7 00  	sltu	a3, a5, a0
   1f274: 36 97        	add	a4, a4, a3
   1f276: 3e 85        	mv	a0, a5
   1f278: 89 a8        	j	0x1f2ca <__multf3+0x590>
   1f27a: 05 47        	addi	a4, zero, 1
   1f27c: 15 8f        	sub	a4, a4, a3
   1f27e: 93 06 40 07  	addi	a3, zero, 116
   1f282: 63 cf e6 06  	blt	a3, a4, 0x1f300 <__multf3+0x5c6>
   1f286: 93 06 f0 03  	addi	a3, zero, 63
   1f28a: 9b 05 07 00  	sext.w	a1, a4
   1f28e: 63 c4 e6 00  	blt	a3, a4, 0x1f296 <__multf3+0x55c>
   1f292: 85 45        	addi	a1, zero, 1
   1f294: 51 bf        	j	0x1f228 <__multf3+0x4ee>
   1f296: 1b 85 05 fc  	addiw	a0, a1, -64
   1f29a: 13 08 00 04  	addi	a6, zero, 64
   1f29e: 33 55 a4 00  	srl	a0, s0, a0
   1f2a2: 81 46        	mv	a3, zero
   1f2a4: 63 07 07 01  	beq	a4, a6, 0x1f2b2 <__multf3+0x578>
   1f2a8: 93 06 00 08  	addi	a3, zero, 128
   1f2ac: 8d 9e        	subw	a3, a3, a1
   1f2ae: b3 16 d4 00  	sll	a3, s0, a3
   1f2b2: dd 8e        	or	a3, a3, a5
   1f2b4: b3 36 d0 00  	snez	a3, a3
   1f2b8: 55 8d        	or	a0, a0, a3
   1f2ba: 85 45        	addi	a1, zero, 1
   1f2bc: 01 47        	mv	a4, zero
   1f2be: 69 b7        	j	0x1f248 <__multf3+0x50e>
   1f2c0: 09 e6        	bnez	a2, 0x1f2ca <__multf3+0x590>
   1f2c2: 93 07 85 00  	addi	a5, a0, 8
   1f2c6: 6d b7        	j	0x1f270 <__multf3+0x536>
   1f2c8: 6d fe        	bnez	a2, 0x1f2c2 <__multf3+0x588>
   1f2ca: 93 57 37 03  	srli	a5, a4, 51
   1f2ce: 85 8b        	andi	a5, a5, 1
   1f2d0: 99 cb        	beqz	a5, 0x1f2e6 <__multf3+0x5ac>
   1f2d2: 93 e4 14 00  	ori	s1, s1, 1
   1f2d6: 01 47        	mv	a4, zero
   1f2d8: 01 45        	mv	a0, zero
   1f2da: 85 46        	addi	a3, zero, 1
   1f2dc: e3 82 05 ea  	beqz	a1, 0x1f180 <__multf3+0x446>
   1f2e0: 93 e4 24 00  	ori	s1, s1, 2
   1f2e4: 71 bd        	j	0x1f180 <__multf3+0x446>
   1f2e6: 93 17 d7 03  	slli	a5, a4, 61
   1f2ea: 0d 81        	srli	a0, a0, 3
   1f2ec: 5d 8d        	or	a0, a0, a5
   1f2ee: 0d 83        	srli	a4, a4, 3
   1f2f0: 81 46        	mv	a3, zero
   1f2f2: e3 87 05 e8  	beqz	a1, 0x1f180 <__multf3+0x446>
   1f2f6: 93 f7 14 00  	andi	a5, s1, 1
   1f2fa: e3 83 07 e8  	beqz	a5, 0x1f180 <__multf3+0x446>
   1f2fe: cd b7        	j	0x1f2e0 <__multf3+0x5a6>
   1f300: 33 e5 87 00  	or	a0, a5, s0
   1f304: 11 cd        	beqz	a0, 0x1f320 <__multf3+0x5e6>
   1f306: 89 47        	addi	a5, zero, 2
   1f308: 93 e4 14 00  	ori	s1, s1, 1
   1f30c: 63 83 f9 02  	beq	s3, a5, 0x1f332 <__multf3+0x5f8>
   1f310: 8d 47        	addi	a5, zero, 3
   1f312: 63 8c f9 00  	beq	s3, a5, 0x1f32a <__multf3+0x5f0>
   1f316: 15 45        	addi	a0, zero, 5
   1f318: 63 83 09 00  	beqz	s3, 0x1f31e <__multf3+0x5e4>
   1f31c: 05 45        	addi	a0, zero, 1
   1f31e: 0d 81        	srli	a0, a0, 3
   1f320: 93 e4 24 00  	ori	s1, s1, 2
   1f324: 01 47        	mv	a4, zero
   1f326: 81 46        	mv	a3, zero
   1f328: a1 bd        	j	0x1f180 <__multf3+0x446>
   1f32a: 05 45        	addi	a0, zero, 1
   1f32c: 6d fa        	bnez	a2, 0x1f31e <__multf3+0x5e4>
   1f32e: 25 45        	addi	a0, zero, 9
   1f330: fd b7        	j	0x1f31e <__multf3+0x5e4>
   1f332: 05 45        	addi	a0, zero, 1
   1f334: 6d d6        	beqz	a2, 0x1f31e <__multf3+0x5e4>
   1f336: e5 bf        	j	0x1f32e <__multf3+0x5f4>
   1f338: a1 66        	lui	a3, 8
   1f33a: 3e 17        	slli	a4, a4, 47
   1f33c: fd 16        	addi	a3, a3, -1
   1f33e: 01 46        	mv	a2, zero
   1f340: c1 44        	addi	s1, zero, 16
   1f342: 3d bd        	j	0x1f180 <__multf3+0x446>
   1f344: a1 66        	lui	a3, 8
   1f346: 01 47        	mv	a4, zero
   1f348: 01 45        	mv	a0, zero
   1f34a: fd 16        	addi	a3, a3, -1
   1f34c: 15 bd        	j	0x1f180 <__multf3+0x446>
   1f34e: 05 47        	addi	a4, zero, 1
   1f350: a1 66        	lui	a3, 8
   1f352: 3e 17        	slli	a4, a4, 47
   1f354: 01 45        	mv	a0, zero
   1f356: fd 16        	addi	a3, a3, -1
   1f358: 01 46        	mv	a2, zero
   1f35a: 1d b5        	j	0x1f180 <__multf3+0x446>

000000000001f35c <__subtf3>:
   1f35c: 79 71        	addi	sp, sp, -48
   1f35e: 06 f4        	sd	ra, 40(sp)
   1f360: 22 f0        	sd	s0, 32(sp)
   1f362: 26 ec        	sd	s1, 24(sp)
   1f364: 4a e8        	sd	s2, 16(sp)
   1f366: 4e e4        	sd	s3, 8(sp)
   1f368: 52 e0        	sd	s4, 0(sp)
   1f36a: 73 29 20 00  	frrm	s2
   1f36e: 7d 53        	addi	t1, zero, -1
   1f370: 13 53 03 01  	srli	t1, t1, 16
   1f374: 13 d4 05 03  	srli	s0, a1, 48
   1f378: a1 68        	lui	a7, 8
   1f37a: 93 d4 f5 03  	srli	s1, a1, 63
   1f37e: b3 f5 65 00  	and	a1, a1, t1
   1f382: 93 97 35 00  	slli	a5, a1, 3
   1f386: fd 18        	addi	a7, a7, -1
   1f388: 93 55 d5 03  	srli	a1, a0, 61
   1f38c: 13 d7 06 03  	srli	a4, a3, 48
   1f390: 13 d8 f6 03  	srli	a6, a3, 63
   1f394: b3 f6 66 00  	and	a3, a3, t1
   1f398: dd 8d        	or	a1, a1, a5
   1f39a: 33 77 17 01  	and	a4, a4, a7
   1f39e: 93 97 36 00  	slli	a5, a3, 3
   1f3a2: 93 56 d6 03  	srli	a3, a2, 61
   1f3a6: 01 29        	sext.w	s2, s2
   1f3a8: 33 74 14 01  	and	s0, s0, a7
   1f3ac: 0e 05        	slli	a0, a0, 3
   1f3ae: dd 8e        	or	a3, a3, a5
   1f3b0: 0e 06        	slli	a2, a2, 3
   1f3b2: 63 18 17 05  	bne	a4, a7, 0x1f402 <__subtf3+0xa6>
   1f3b6: e1 77        	lui	a5, 1048568
   1f3b8: 85 27        	addiw	a5, a5, 1
   1f3ba: b3 e8 c6 00  	or	a7, a3, a2
   1f3be: a1 9f        	addw	a5, a5, s0
   1f3c0: 63 87 08 02  	beqz	a7, 0x1f3ee <__subtf3+0x92>
   1f3c4: 42 83        	mv	t1, a6
   1f3c6: 63 93 04 4b  	bne	s1, a6, 0x1f86c <__subtf3+0x510>
   1f3ca: 63 8d 07 24  	beqz	a5, 0x1f624 <__subtf3+0x2c8>
   1f3ce: 63 12 04 4a  	bnez	s0, 0x1f872 <__subtf3+0x516>
   1f3d2: 33 e8 a5 00  	or	a6, a1, a0
   1f3d6: 63 17 08 1e  	bnez	a6, 0x1f5c4 <__subtf3+0x268>
   1f3da: a1 65        	lui	a1, 8
   1f3dc: fd 15        	addi	a1, a1, -1
   1f3de: 81 47        	mv	a5, zero
   1f3e0: 63 14 b7 0a  	bne	a4, a1, 0x1f488 <__subtf3+0x12c>
   1f3e4: 33 e8 c6 00  	or	a6, a3, a2
   1f3e8: 63 05 08 2c  	beqz	a6, 0x1f6b2 <__subtf3+0x356>
   1f3ec: 59 a1        	j	0x1f872 <__subtf3+0x516>
   1f3ee: 13 43 18 00  	xori	t1, a6, 1
   1f3f2: 63 97 64 48  	bne	s1, t1, 0x1f880 <__subtf3+0x524>
   1f3f6: 63 87 07 22  	beqz	a5, 0x1f624 <__subtf3+0x2c8>
   1f3fa: 61 dc        	beqz	s0, 0x1f3d2 <__subtf3+0x76>
   1f3fc: 01 48        	mv	a6, zero
   1f3fe: 81 45        	mv	a1, zero
   1f400: b5 ab        	j	0x1f97c <__subtf3+0x620>
   1f402: 13 43 18 00  	xori	t1, a6, 1
   1f406: bb 07 e4 40  	subw	a5, s0, a4
   1f40a: 3e 88        	mv	a6, a5
   1f40c: 63 99 64 38  	bne	s1, t1, 0x1f79e <__subtf3+0x442>
   1f410: 63 50 f0 16  	blez	a5, 0x1f570 <__subtf3+0x214>
   1f414: 69 eb        	bnez	a4, 0x1f4e6 <__subtf3+0x18a>
   1f416: b3 e7 c6 00  	or	a5, a3, a2
   1f41a: 91 e7        	bnez	a5, 0x1f426 <__subtf3+0xca>
   1f41c: 63 09 14 05  	beq	s0, a7, 0x1f46e <__subtf3+0x112>
   1f420: ae 86        	mv	a3, a1
   1f422: 2a 86        	mv	a2, a0
   1f424: a1 ac        	j	0x1f67c <__subtf3+0x320>
   1f426: 9b 07 f8 ff  	addiw	a5, a6, -1
   1f42a: a1 e3        	bnez	a5, 0x1f46a <__subtf3+0x10e>
   1f42c: 2a 96        	add	a2, a2, a0
   1f42e: ae 96        	add	a3, a3, a1
   1f430: 33 35 a6 00  	sltu	a0, a2, a0
   1f434: aa 96        	add	a3, a3, a0
   1f436: 22 87        	mv	a4, s0
   1f438: 93 d7 36 03  	srli	a5, a3, 51
   1f43c: 85 8b        	andi	a5, a5, 1
   1f43e: 63 87 07 22  	beqz	a5, 0x1f66c <__subtf3+0x310>
   1f442: a1 67        	lui	a5, 8
   1f444: 05 07        	addi	a4, a4, 1
   1f446: 93 85 f7 ff  	addi	a1, a5, -1
   1f44a: 63 03 b7 32  	beq	a4, a1, 0x1f770 <__subtf3+0x414>
   1f44e: fd 57        	addi	a5, zero, -1
   1f450: ce 17        	slli	a5, a5, 51
   1f452: fd 17        	addi	a5, a5, -1
   1f454: f5 8f        	and	a5, a5, a3
   1f456: 93 56 16 00  	srli	a3, a2, 1
   1f45a: 05 8a        	andi	a2, a2, 1
   1f45c: d1 8e        	or	a3, a3, a2
   1f45e: 13 96 f7 03  	slli	a2, a5, 63
   1f462: 55 8e        	or	a2, a2, a3
   1f464: 93 d6 17 00  	srli	a3, a5, 1
   1f468: 11 a4        	j	0x1f66c <__subtf3+0x310>
   1f46a: 63 15 14 0d  	bne	s0, a7, 0x1f534 <__subtf3+0x1d8>
   1f46e: 33 e8 a5 00  	or	a6, a1, a0
   1f472: 63 02 08 7e  	beqz	a6, 0x1fc56 <__subtf3+0x8fa>
   1f476: 93 d7 25 03  	srli	a5, a1, 50
   1f47a: 93 c7 17 00  	xori	a5, a5, 1
   1f47e: 85 8b        	andi	a5, a5, 1
   1f480: 92 07        	slli	a5, a5, 4
   1f482: ae 86        	mv	a3, a1
   1f484: 2a 86        	mv	a2, a0
   1f486: 22 87        	mv	a4, s0
   1f488: 93 75 76 00  	andi	a1, a2, 7
   1f48c: 63 90 05 78  	bnez	a1, 0x1fc0c <__subtf3+0x8b0>
   1f490: b6 85        	mv	a1, a3
   1f492: 32 85        	mv	a0, a2
   1f494: 93 d6 35 03  	srli	a3, a1, 51
   1f498: 85 8a        	andi	a3, a3, 1
   1f49a: 63 81 06 7c  	beqz	a3, 0x1fc5c <__subtf3+0x900>
   1f49e: a1 66        	lui	a3, 8
   1f4a0: 05 07        	addi	a4, a4, 1
   1f4a2: 13 86 f6 ff  	addi	a2, a3, -1
   1f4a6: 63 00 c7 78  	beq	a4, a2, 0x1fc26 <__subtf3+0x8ca>
   1f4aa: fd 56        	addi	a3, zero, -1
   1f4ac: ce 16        	slli	a3, a3, 51
   1f4ae: fd 16        	addi	a3, a3, -1
   1f4b0: f5 8d        	and	a1, a1, a3
   1f4b2: 93 56 35 00  	srli	a3, a0, 3
   1f4b6: 13 95 d5 03  	slli	a0, a1, 61
   1f4ba: 55 8d        	or	a0, a0, a3
   1f4bc: 8d 81        	srli	a1, a1, 3
   1f4be: 46 17        	slli	a4, a4, 49
   1f4c0: 9b 94 f4 00  	slliw	s1, s1, 15
   1f4c4: 45 93        	srli	a4, a4, 49
   1f4c6: c2 05        	slli	a1, a1, 16
   1f4c8: 45 8f        	or	a4, a4, s1
   1f4ca: 42 17        	slli	a4, a4, 48
   1f4cc: c1 81        	srli	a1, a1, 16
   1f4ce: d9 8d        	or	a1, a1, a4
   1f4d0: 99 c3        	beqz	a5, 0x1f4d6 <__subtf3+0x17a>
   1f4d2: 73 a0 17 00  	csrs	fflags, a5
   1f4d6: a2 70        	ld	ra, 40(sp)
   1f4d8: 02 74        	ld	s0, 32(sp)
   1f4da: e2 64        	ld	s1, 24(sp)
   1f4dc: 42 69        	ld	s2, 16(sp)
   1f4de: a2 69        	ld	s3, 8(sp)
   1f4e0: 02 6a        	ld	s4, 0(sp)
   1f4e2: 45 61        	addi	sp, sp, 48
   1f4e4: 82 80        	ret
   1f4e6: e3 04 14 f9  	beq	s0, a7, 0x1f46e <__subtf3+0x112>
   1f4ea: 13 07 40 07  	addi	a4, zero, 116
   1f4ee: 63 47 f7 04  	blt	a4, a5, 0x1f53c <__subtf3+0x1e0>
   1f4f2: 05 47        	addi	a4, zero, 1
   1f4f4: 4e 17        	slli	a4, a4, 51
   1f4f6: d9 8e        	or	a3, a3, a4
   1f4f8: 13 07 f0 03  	addi	a4, zero, 63
   1f4fc: 63 43 f7 04  	blt	a4, a5, 0x1f542 <__subtf3+0x1e6>
   1f500: 13 08 00 04  	addi	a6, zero, 64
   1f504: 3b 08 f8 40  	subw	a6, a6, a5
   1f508: 33 97 06 01  	sll	a4, a3, a6
   1f50c: b3 58 f6 00  	srl	a7, a2, a5
   1f510: 33 18 06 01  	sll	a6, a2, a6
   1f514: 33 67 17 01  	or	a4, a4, a7
   1f518: 33 38 00 01  	snez	a6, a6
   1f51c: 33 67 07 01  	or	a4, a4, a6
   1f520: b3 d7 f6 00  	srl	a5, a3, a5
   1f524: 33 06 a7 00  	add	a2, a4, a0
   1f528: ae 97        	add	a5, a5, a1
   1f52a: 33 37 e6 00  	sltu	a4, a2, a4
   1f52e: b3 86 e7 00  	add	a3, a5, a4
   1f532: 11 b7        	j	0x1f436 <__subtf3+0xda>
   1f534: 13 07 40 07  	addi	a4, zero, 116
   1f538: e3 50 f7 fc  	bge	a4, a5, 0x1f4f8 <__subtf3+0x19c>
   1f53c: 81 47        	mv	a5, zero
   1f53e: 05 47        	addi	a4, zero, 1
   1f540: d5 b7        	j	0x1f524 <__subtf3+0x1c8>
   1f542: 1b 87 07 fc  	addiw	a4, a5, -64
   1f546: 93 08 00 04  	addi	a7, zero, 64
   1f54a: 33 d7 e6 00  	srl	a4, a3, a4
   1f54e: 01 48        	mv	a6, zero
   1f550: 63 88 17 01  	beq	a5, a7, 0x1f560 <__subtf3+0x204>
   1f554: 13 08 00 08  	addi	a6, zero, 128
   1f558: 3b 08 f8 40  	subw	a6, a6, a5
   1f55c: 33 98 06 01  	sll	a6, a3, a6
   1f560: 33 68 c8 00  	or	a6, a6, a2
   1f564: 33 38 00 01  	snez	a6, a6
   1f568: 33 67 07 01  	or	a4, a4, a6
   1f56c: 81 47        	mv	a5, zero
   1f56e: 5d bf        	j	0x1f524 <__subtf3+0x1c8>
   1f570: d5 cb        	beqz	a5, 0x1f624 <__subtf3+0x2c8>
   1f572: e3 00 04 e6  	beqz	s0, 0x1f3d2 <__subtf3+0x76>
   1f576: 93 08 c0 f8  	addi	a7, zero, -116
   1f57a: 63 cb 17 07  	blt	a5, a7, 0x1f5f0 <__subtf3+0x294>
   1f57e: 05 48        	addi	a6, zero, 1
   1f580: 4e 18        	slli	a6, a6, 51
   1f582: bb 07 f0 40  	negw	a5, a5
   1f586: b3 e5 05 01  	or	a1, a1, a6
   1f58a: 13 08 f0 03  	addi	a6, zero, 63
   1f58e: 63 44 f8 06  	blt	a6, a5, 0x1f5f6 <__subtf3+0x29a>
   1f592: 93 08 00 04  	addi	a7, zero, 64
   1f596: bb 88 f8 40  	subw	a7, a7, a5
   1f59a: 33 53 f5 00  	srl	t1, a0, a5
   1f59e: 33 98 15 01  	sll	a6, a1, a7
   1f5a2: b3 18 15 01  	sll	a7, a0, a7
   1f5a6: 33 68 68 00  	or	a6, a6, t1
   1f5aa: b3 38 10 01  	snez	a7, a7
   1f5ae: 33 65 18 01  	or	a0, a6, a7
   1f5b2: b3 d7 f5 00  	srl	a5, a1, a5
   1f5b6: 2a 96        	add	a2, a2, a0
   1f5b8: b6 97        	add	a5, a5, a3
   1f5ba: 33 35 a6 00  	sltu	a0, a2, a0
   1f5be: b3 86 a7 00  	add	a3, a5, a0
   1f5c2: 9d bd        	j	0x1f438 <__subtf3+0xdc>
   1f5c4: 93 c7 f7 ff  	not	a5, a5
   1f5c8: 99 e7        	bnez	a5, 0x1f5d6 <__subtf3+0x27a>
   1f5ca: 2a 96        	add	a2, a2, a0
   1f5cc: ae 96        	add	a3, a3, a1
   1f5ce: 33 35 a6 00  	sltu	a0, a2, a0
   1f5d2: aa 96        	add	a3, a3, a0
   1f5d4: 95 b5        	j	0x1f438 <__subtf3+0xdc>
   1f5d6: 21 68        	lui	a6, 8
   1f5d8: 7d 18        	addi	a6, a6, -1
   1f5da: 63 17 07 01  	bne	a4, a6, 0x1f5e8 <__subtf3+0x28c>
   1f5de: 33 e8 c6 00  	or	a6, a3, a2
   1f5e2: e3 0e 08 e0  	beqz	a6, 0x1f3fe <__subtf3+0xa2>
   1f5e6: 71 a4        	j	0x1f872 <__subtf3+0x516>
   1f5e8: 13 08 40 07  	addi	a6, zero, 116
   1f5ec: e3 5f f8 f8  	bge	a6, a5, 0x1f58a <__subtf3+0x22e>
   1f5f0: 81 47        	mv	a5, zero
   1f5f2: 05 45        	addi	a0, zero, 1
   1f5f4: c9 b7        	j	0x1f5b6 <__subtf3+0x25a>
   1f5f6: 1b 88 07 fc  	addiw	a6, a5, -64
   1f5fa: 13 03 00 04  	addi	t1, zero, 64
   1f5fe: 33 d8 05 01  	srl	a6, a1, a6
   1f602: 81 48        	mv	a7, zero
   1f604: 63 88 67 00  	beq	a5, t1, 0x1f614 <__subtf3+0x2b8>
   1f608: 93 08 00 08  	addi	a7, zero, 128
   1f60c: bb 88 f8 40  	subw	a7, a7, a5
   1f610: b3 98 15 01  	sll	a7, a1, a7
   1f614: b3 e8 a8 00  	or	a7, a7, a0
   1f618: b3 38 10 01  	snez	a7, a7
   1f61c: 33 65 18 01  	or	a0, a6, a7
   1f620: 81 47        	mv	a5, zero
   1f622: 51 bf        	j	0x1f5b6 <__subtf3+0x25a>
   1f624: a1 68        	lui	a7, 8
   1f626: 13 88 e8 ff  	addi	a6, a7, -2
   1f62a: 93 07 14 00  	addi	a5, s0, 1
   1f62e: 33 f3 07 01  	and	t1, a5, a6
   1f632: fd 18        	addi	a7, a7, -1
   1f634: 63 19 03 0c  	bnez	t1, 0x1f706 <__subtf3+0x3aa>
   1f638: 33 e8 a5 00  	or	a6, a1, a0
   1f63c: 15 e8        	bnez	s0, 0x1f670 <__subtf3+0x314>
   1f63e: 63 00 08 54  	beqz	a6, 0x1fb7e <__subtf3+0x822>
   1f642: b3 e7 c6 00  	or	a5, a3, a2
   1f646: 63 8b 07 46  	beqz	a5, 0x1fabc <__subtf3+0x760>
   1f64a: 2a 96        	add	a2, a2, a0
   1f64c: b3 37 a6 00  	sltu	a5, a2, a0
   1f650: ae 96        	add	a3, a3, a1
   1f652: b3 85 f6 00  	add	a1, a3, a5
   1f656: 93 d7 35 03  	srli	a5, a1, 51
   1f65a: 85 8b        	andi	a5, a5, 1
   1f65c: 32 85        	mv	a0, a2
   1f65e: 63 8f 07 44  	beqz	a5, 0x1fabc <__subtf3+0x760>
   1f662: fd 56        	addi	a3, zero, -1
   1f664: ce 16        	slli	a3, a3, 51
   1f666: fd 16        	addi	a3, a3, -1
   1f668: ed 8e        	and	a3, a3, a1
   1f66a: 05 47        	addi	a4, zero, 1
   1f66c: 81 47        	mv	a5, zero
   1f66e: 29 bd        	j	0x1f488 <__subtf3+0x12c>
   1f670: 63 18 14 03  	bne	s0, a7, 0x1f6a0 <__subtf3+0x344>
   1f674: 63 16 08 00  	bnez	a6, 0x1f680 <__subtf3+0x324>
   1f678: e3 03 87 f6  	beq	a4, s0, 0x1f5de <__subtf3+0x282>
   1f67c: 22 87        	mv	a4, s0
   1f67e: fd b7        	j	0x1f66c <__subtf3+0x310>
   1f680: 85 48        	addi	a7, zero, 1
   1f682: ca 18        	slli	a7, a7, 50
   1f684: b3 f7 15 01  	and	a5, a1, a7
   1f688: 33 e3 c6 00  	or	t1, a3, a2
   1f68c: 63 8a 07 5c  	beqz	a5, 0x1fc60 <__subtf3+0x904>
   1f690: 81 47        	mv	a5, zero
   1f692: 63 13 87 06  	bne	a4, s0, 0x1f6f8 <__subtf3+0x39c>
   1f696: 63 10 03 02  	bnez	t1, 0x1f6b6 <__subtf3+0x35a>
   1f69a: ae 86        	mv	a3, a1
   1f69c: 2a 86        	mv	a2, a0
   1f69e: ed b3        	j	0x1f488 <__subtf3+0x12c>
   1f6a0: 81 47        	mv	a5, zero
   1f6a2: 63 19 17 05  	bne	a4, a7, 0x1f6f4 <__subtf3+0x398>
   1f6a6: b3 e8 c6 00  	or	a7, a3, a2
   1f6aa: 63 90 08 5c  	bnez	a7, 0x1fc6a <__subtf3+0x90e>
   1f6ae: e3 16 08 fe  	bnez	a6, 0x1f69a <__subtf3+0x33e>
   1f6b2: 81 45        	mv	a1, zero
   1f6b4: 19 a8        	j	0x1f6ca <__subtf3+0x36e>
   1f6b6: b3 f8 16 01  	and	a7, a3, a7
   1f6ba: 63 84 08 04  	beqz	a7, 0x1f702 <__subtf3+0x3a6>
   1f6be: 85 45        	addi	a1, zero, 1
   1f6c0: 21 67        	lui	a4, 8
   1f6c2: 81 44        	mv	s1, zero
   1f6c4: ca 15        	slli	a1, a1, 50
   1f6c6: 01 48        	mv	a6, zero
   1f6c8: 7d 17        	addi	a4, a4, -1
   1f6ca: a1 66        	lui	a3, 8
   1f6cc: 13 95 d5 03  	slli	a0, a1, 61
   1f6d0: 13 58 38 00  	srli	a6, a6, 3
   1f6d4: fd 16        	addi	a3, a3, -1
   1f6d6: 33 65 05 01  	or	a0, a0, a6
   1f6da: 8d 81        	srli	a1, a1, 3
   1f6dc: e3 11 d7 de  	bne	a4, a3, 0x1f4be <__subtf3+0x162>
   1f6e0: 4d 8d        	or	a0, a0, a1
   1f6e2: 81 45        	mv	a1, zero
   1f6e4: e3 0d 05 dc  	beqz	a0, 0x1f4be <__subtf3+0x162>
   1f6e8: 85 45        	addi	a1, zero, 1
   1f6ea: be 15        	slli	a1, a1, 47
   1f6ec: 01 45        	mv	a0, zero
   1f6ee: 81 44        	mv	s1, zero
   1f6f0: f9 b3        	j	0x1f4be <__subtf3+0x162>
   1f6f2: c1 47        	addi	a5, zero, 16
   1f6f4: 63 02 08 58  	beqz	a6, 0x1fc78 <__subtf3+0x91c>
   1f6f8: d1 8e        	or	a3, a3, a2
   1f6fa: f1 f2        	bnez	a3, 0x1f6be <__subtf3+0x362>
   1f6fc: ae 86        	mv	a3, a1
   1f6fe: 2a 86        	mv	a2, a0
   1f700: a5 ab        	j	0x1fc78 <__subtf3+0x91c>
   1f702: c1 47        	addi	a5, zero, 16
   1f704: d5 bf        	j	0x1f6f8 <__subtf3+0x39c>
   1f706: 63 8e 17 01  	beq	a5, a7, 0x1f722 <__subtf3+0x3c6>
   1f70a: 2a 96        	add	a2, a2, a0
   1f70c: 33 35 a6 00  	sltu	a0, a2, a0
   1f710: ae 96        	add	a3, a3, a1
   1f712: aa 96        	add	a3, a3, a0
   1f714: 13 97 f6 03  	slli	a4, a3, 63
   1f718: 05 82        	srli	a2, a2, 1
   1f71a: 59 8e        	or	a2, a2, a4
   1f71c: 85 82        	srli	a3, a3, 1
   1f71e: 3e 87        	mv	a4, a5
   1f720: b1 b7        	j	0x1f66c <__subtf3+0x310>
   1f722: 63 05 09 52  	beqz	s2, 0x1fc4c <__subtf3+0x8f0>
   1f726: 0d 47        	addi	a4, zero, 3
   1f728: 63 1a e9 02  	bne	s2, a4, 0x1f75c <__subtf3+0x400>
   1f72c: 63 80 04 52  	beqz	s1, 0x1fc4c <__subtf3+0x8f0>
   1f730: 81 46        	mv	a3, zero
   1f732: fd 55        	addi	a1, zero, -1
   1f734: 7d 55        	addi	a0, zero, -1
   1f736: 42 87        	mv	a4, a6
   1f738: 95 47        	addi	a5, zero, 5
   1f73a: 0d 46        	addi	a2, zero, 3
   1f73c: 63 0e c9 4c  	beq	s2, a2, 0x1fc18 <__subtf3+0x8bc>
   1f740: 63 1d 09 4c  	bnez	s2, 0x1fc1a <__subtf3+0x8be>
   1f744: 13 76 f5 00  	andi	a2, a0, 15
   1f748: 11 48        	addi	a6, zero, 4
   1f74a: 63 08 06 4d  	beq	a2, a6, 0x1fc1a <__subtf3+0x8be>
   1f74e: 13 06 45 00  	addi	a2, a0, 4
   1f752: 33 38 a6 00  	sltu	a6, a2, a0
   1f756: c2 95        	add	a1, a1, a6
   1f758: 32 85        	mv	a0, a2
   1f75a: c1 a1        	j	0x1fc1a <__subtf3+0x8be>
   1f75c: 09 47        	addi	a4, zero, 2
   1f75e: 63 14 e9 00  	bne	s2, a4, 0x1f766 <__subtf3+0x40a>
   1f762: 63 95 04 4e  	bnez	s1, 0x1fc4c <__subtf3+0x8f0>
   1f766: fd 55        	addi	a1, zero, -1
   1f768: 7d 55        	addi	a0, zero, -1
   1f76a: 42 87        	mv	a4, a6
   1f76c: 95 47        	addi	a5, zero, 5
   1f76e: 1d b3        	j	0x1f494 <__subtf3+0x138>
   1f770: 63 0f 09 4c  	beqz	s2, 0x1fc4e <__subtf3+0x8f2>
   1f774: 8d 46        	addi	a3, zero, 3
   1f776: 63 1a d9 00  	bne	s2, a3, 0x1f78a <__subtf3+0x42e>
   1f77a: 63 8a 04 4c  	beqz	s1, 0x1fc4e <__subtf3+0x8f2>
   1f77e: 81 46        	mv	a3, zero
   1f780: fd 55        	addi	a1, zero, -1
   1f782: 7d 55        	addi	a0, zero, -1
   1f784: 13 87 e7 ff  	addi	a4, a5, -2
   1f788: 45 bf        	j	0x1f738 <__subtf3+0x3dc>
   1f78a: 89 46        	addi	a3, zero, 2
   1f78c: 63 14 d9 00  	bne	s2, a3, 0x1f794 <__subtf3+0x438>
   1f790: 63 9f 04 4a  	bnez	s1, 0x1fc4e <__subtf3+0x8f2>
   1f794: fd 55        	addi	a1, zero, -1
   1f796: 7d 55        	addi	a0, zero, -1
   1f798: 13 87 e7 ff  	addi	a4, a5, -2
   1f79c: c1 bf        	j	0x1f76c <__subtf3+0x410>
   1f79e: 63 54 f0 10  	blez	a5, 0x1f8a6 <__subtf3+0x54a>
   1f7a2: 29 e7        	bnez	a4, 0x1f7ec <__subtf3+0x490>
   1f7a4: b3 e7 c6 00  	or	a5, a3, a2
   1f7a8: e3 8a 07 c6  	beqz	a5, 0x1f41c <__subtf3+0xc0>
   1f7ac: 9b 07 f8 ff  	addiw	a5, a6, -1
   1f7b0: 8d e7        	bnez	a5, 0x1f7da <__subtf3+0x47e>
   1f7b2: 33 06 c5 40  	sub	a2, a0, a2
   1f7b6: b3 86 d5 40  	sub	a3, a1, a3
   1f7ba: 33 35 c5 00  	sltu	a0, a0, a2
   1f7be: 89 8e        	sub	a3, a3, a0
   1f7c0: 22 87        	mv	a4, s0
   1f7c2: 93 d7 36 03  	srli	a5, a3, 51
   1f7c6: 85 8b        	andi	a5, a5, 1
   1f7c8: e3 82 07 ea  	beqz	a5, 0x1f66c <__subtf3+0x310>
   1f7cc: 93 99 d6 00  	slli	s3, a3, 13
   1f7d0: 93 d9 d9 00  	srli	s3, s3, 13
   1f7d4: 32 8a        	mv	s4, a2
   1f7d6: 3a 84        	mv	s0, a4
   1f7d8: 69 ac        	j	0x1fa72 <__subtf3+0x716>
   1f7da: e3 0a 14 c9  	beq	s0, a7, 0x1f46e <__subtf3+0x112>
   1f7de: 13 07 40 07  	addi	a4, zero, 116
   1f7e2: 63 5e f7 00  	bge	a4, a5, 0x1f7fe <__subtf3+0x4a2>
   1f7e6: 81 47        	mv	a5, zero
   1f7e8: 05 47        	addi	a4, zero, 1
   1f7ea: 81 a0        	j	0x1f82a <__subtf3+0x4ce>
   1f7ec: e3 01 14 c9  	beq	s0, a7, 0x1f46e <__subtf3+0x112>
   1f7f0: 13 07 40 07  	addi	a4, zero, 116
   1f7f4: e3 49 f7 fe  	blt	a4, a5, 0x1f7e6 <__subtf3+0x48a>
   1f7f8: 05 47        	addi	a4, zero, 1
   1f7fa: 4e 17        	slli	a4, a4, 51
   1f7fc: d9 8e        	or	a3, a3, a4
   1f7fe: 13 07 f0 03  	addi	a4, zero, 63
   1f802: 63 4e f7 02  	blt	a4, a5, 0x1f83e <__subtf3+0x4e2>
   1f806: 13 08 00 04  	addi	a6, zero, 64
   1f80a: 3b 08 f8 40  	subw	a6, a6, a5
   1f80e: 33 97 06 01  	sll	a4, a3, a6
   1f812: b3 58 f6 00  	srl	a7, a2, a5
   1f816: 33 18 06 01  	sll	a6, a2, a6
   1f81a: 33 67 17 01  	or	a4, a4, a7
   1f81e: 33 38 00 01  	snez	a6, a6
   1f822: 33 67 07 01  	or	a4, a4, a6
   1f826: b3 d7 f6 00  	srl	a5, a3, a5
   1f82a: 33 07 e5 40  	sub	a4, a0, a4
   1f82e: b3 87 f5 40  	sub	a5, a1, a5
   1f832: b3 36 e5 00  	sltu	a3, a0, a4
   1f836: 3a 86        	mv	a2, a4
   1f838: b3 86 d7 40  	sub	a3, a5, a3
   1f83c: 51 b7        	j	0x1f7c0 <__subtf3+0x464>
   1f83e: 1b 87 07 fc  	addiw	a4, a5, -64
   1f842: 93 08 00 04  	addi	a7, zero, 64
   1f846: 33 d7 e6 00  	srl	a4, a3, a4
   1f84a: 01 48        	mv	a6, zero
   1f84c: 63 88 17 01  	beq	a5, a7, 0x1f85c <__subtf3+0x500>
   1f850: 13 08 00 08  	addi	a6, zero, 128
   1f854: 3b 08 f8 40  	subw	a6, a6, a5
   1f858: 33 98 06 01  	sll	a6, a3, a6
   1f85c: 33 68 c8 00  	or	a6, a6, a2
   1f860: 33 38 00 01  	snez	a6, a6
   1f864: 33 67 07 01  	or	a4, a4, a6
   1f868: 81 47        	mv	a5, zero
   1f86a: c1 b7        	j	0x1f82a <__subtf3+0x4ce>
   1f86c: e5 c3        	beqz	a5, 0x1f94c <__subtf3+0x5f0>
   1f86e: c2 84        	mv	s1, a6
   1f870: 01 cc        	beqz	s0, 0x1f888 <__subtf3+0x52c>
   1f872: 93 d7 26 03  	srli	a5, a3, 50
   1f876: 93 c7 17 00  	xori	a5, a5, 1
   1f87a: 85 8b        	andi	a5, a5, 1
   1f87c: 92 07        	slli	a5, a5, 4
   1f87e: 29 b1        	j	0x1f488 <__subtf3+0x12c>
   1f880: f1 c7        	beqz	a5, 0x1f94c <__subtf3+0x5f0>
   1f882: 9a 84        	mv	s1, t1
   1f884: e3 1c 04 b6  	bnez	s0, 0x1f3fc <__subtf3+0xa0>
   1f888: 33 e8 a5 00  	or	a6, a1, a0
   1f88c: e3 07 08 b4  	beqz	a6, 0x1f3da <__subtf3+0x7e>
   1f890: 93 c7 f7 ff  	not	a5, a5
   1f894: 89 ef        	bnez	a5, 0x1f8ae <__subtf3+0x552>
   1f896: 33 05 a6 40  	sub	a0, a2, a0
   1f89a: b3 37 a6 00  	sltu	a5, a2, a0
   1f89e: 8d 8e        	sub	a3, a3, a1
   1f8a0: 2a 86        	mv	a2, a0
   1f8a2: 9d 8e        	sub	a3, a3, a5
   1f8a4: 39 bf        	j	0x1f7c2 <__subtf3+0x466>
   1f8a6: dd c3        	beqz	a5, 0x1f94c <__subtf3+0x5f0>
   1f8a8: 11 ec        	bnez	s0, 0x1f8c4 <__subtf3+0x568>
   1f8aa: 9a 84        	mv	s1, t1
   1f8ac: f1 bf        	j	0x1f888 <__subtf3+0x52c>
   1f8ae: 21 68        	lui	a6, 8
   1f8b0: 7d 18        	addi	a6, a6, -1
   1f8b2: e3 06 07 d3  	beq	a4, a6, 0x1f5de <__subtf3+0x282>
   1f8b6: 13 08 40 07  	addi	a6, zero, 116
   1f8ba: 63 50 f8 02  	bge	a6, a5, 0x1f8da <__subtf3+0x57e>
   1f8be: 81 47        	mv	a5, zero
   1f8c0: 05 45        	addi	a0, zero, 1
   1f8c2: 91 a0        	j	0x1f906 <__subtf3+0x5aa>
   1f8c4: 93 08 c0 f8  	addi	a7, zero, -116
   1f8c8: 63 c0 17 09  	blt	a5, a7, 0x1f948 <__subtf3+0x5ec>
   1f8cc: 05 48        	addi	a6, zero, 1
   1f8ce: 4e 18        	slli	a6, a6, 51
   1f8d0: bb 07 f0 40  	negw	a5, a5
   1f8d4: b3 e5 05 01  	or	a1, a1, a6
   1f8d8: 9a 84        	mv	s1, t1
   1f8da: 13 08 f0 03  	addi	a6, zero, 63
   1f8de: 63 4e f8 02  	blt	a6, a5, 0x1f91a <__subtf3+0x5be>
   1f8e2: 93 08 00 04  	addi	a7, zero, 64
   1f8e6: bb 88 f8 40  	subw	a7, a7, a5
   1f8ea: 33 53 f5 00  	srl	t1, a0, a5
   1f8ee: 33 98 15 01  	sll	a6, a1, a7
   1f8f2: b3 18 15 01  	sll	a7, a0, a7
   1f8f6: 33 68 68 00  	or	a6, a6, t1
   1f8fa: b3 38 10 01  	snez	a7, a7
   1f8fe: 33 65 18 01  	or	a0, a6, a7
   1f902: b3 d7 f5 00  	srl	a5, a1, a5
   1f906: 33 05 a6 40  	sub	a0, a2, a0
   1f90a: b3 35 a6 00  	sltu	a1, a2, a0
   1f90e: b3 87 f6 40  	sub	a5, a3, a5
   1f912: 2a 86        	mv	a2, a0
   1f914: b3 86 b7 40  	sub	a3, a5, a1
   1f918: 6d b5        	j	0x1f7c2 <__subtf3+0x466>
   1f91a: 1b 88 07 fc  	addiw	a6, a5, -64
   1f91e: 13 03 00 04  	addi	t1, zero, 64
   1f922: 33 d8 05 01  	srl	a6, a1, a6
   1f926: 81 48        	mv	a7, zero
   1f928: 63 88 67 00  	beq	a5, t1, 0x1f938 <__subtf3+0x5dc>
   1f92c: 93 08 00 08  	addi	a7, zero, 128
   1f930: bb 88 f8 40  	subw	a7, a7, a5
   1f934: b3 98 15 01  	sll	a7, a1, a7
   1f938: b3 e8 a8 00  	or	a7, a7, a0
   1f93c: b3 38 10 01  	snez	a7, a7
   1f940: 33 65 18 01  	or	a0, a6, a7
   1f944: 81 47        	mv	a5, zero
   1f946: c1 b7        	j	0x1f906 <__subtf3+0x5aa>
   1f948: 9a 84        	mv	s1, t1
   1f94a: 95 bf        	j	0x1f8be <__subtf3+0x562>
   1f94c: 21 6e        	lui	t3, 8
   1f94e: 93 07 14 00  	addi	a5, s0, 1
   1f952: 13 08 ee ff  	addi	a6, t3, -2
   1f956: b3 f7 07 01  	and	a5, a5, a6
   1f95a: f5 e7        	bnez	a5, 0x1fa46 <__subtf3+0x6ea>
   1f95c: 33 e8 c6 00  	or	a6, a3, a2
   1f960: b3 e8 a5 00  	or	a7, a1, a0
   1f964: 29 ec        	bnez	s0, 0x1f9be <__subtf3+0x662>
   1f966: 63 9d 08 00  	bnez	a7, 0x1f980 <__subtf3+0x624>
   1f96a: 63 1d 08 20  	bnez	a6, 0x1fb84 <__subtf3+0x828>
   1f96e: 93 04 e9 ff  	addi	s1, s2, -2
   1f972: 93 b4 14 00  	seqz	s1, s1
   1f976: 81 45        	mv	a1, zero
   1f978: 01 48        	mv	a6, zero
   1f97a: 01 47        	mv	a4, zero
   1f97c: 81 47        	mv	a5, zero
   1f97e: b1 b3        	j	0x1f6ca <__subtf3+0x36e>
   1f980: 63 0e 08 12  	beqz	a6, 0x1fabc <__subtf3+0x760>
   1f984: 33 07 c5 40  	sub	a4, a0, a2
   1f988: 33 38 e5 00  	sltu	a6, a0, a4
   1f98c: b3 87 d5 40  	sub	a5, a1, a3
   1f990: b3 87 07 41  	sub	a5, a5, a6
   1f994: 13 d8 37 03  	srli	a6, a5, 51
   1f998: 13 78 18 00  	andi	a6, a6, 1
   1f99c: 63 0b 08 00  	beqz	a6, 0x1f9b2 <__subtf3+0x656>
   1f9a0: 33 05 a6 40  	sub	a0, a2, a0
   1f9a4: 8d 8e        	sub	a3, a3, a1
   1f9a6: 33 36 a6 00  	sltu	a2, a2, a0
   1f9aa: b3 85 c6 40  	sub	a1, a3, a2
   1f9ae: 9a 84        	mv	s1, t1
   1f9b0: 31 a2        	j	0x1fabc <__subtf3+0x760>
   1f9b2: b3 66 f7 00  	or	a3, a4, a5
   1f9b6: c5 de        	beqz	a3, 0x1f96e <__subtf3+0x612>
   1f9b8: be 85        	mv	a1, a5
   1f9ba: 3a 85        	mv	a0, a4
   1f9bc: 01 a2        	j	0x1fabc <__subtf3+0x760>
   1f9be: 7d 1e        	addi	t3, t3, -1
   1f9c0: 63 10 c4 05  	bne	s0, t3, 0x1fa00 <__subtf3+0x6a4>
   1f9c4: 63 9e 08 00  	bnez	a7, 0x1f9e0 <__subtf3+0x684>
   1f9c8: 81 47        	mv	a5, zero
   1f9ca: 63 13 87 06  	bne	a4, s0, 0x1fa30 <__subtf3+0x6d4>
   1f9ce: 85 45        	addi	a1, zero, 1
   1f9d0: ca 15        	slli	a1, a1, 50
   1f9d2: 81 44        	mv	s1, zero
   1f9d4: 63 00 08 04  	beqz	a6, 0x1fa14 <__subtf3+0x6b8>
   1f9d8: f5 8d        	and	a1, a1, a3
   1f9da: a5 c5        	beqz	a1, 0x1fa42 <__subtf3+0x6e6>
   1f9dc: 9a 84        	mv	s1, t1
   1f9de: 6d b4        	j	0x1f488 <__subtf3+0x12c>
   1f9e0: 05 46        	addi	a2, zero, 1
   1f9e2: 4a 16        	slli	a2, a2, 50
   1f9e4: b3 f7 c5 00  	and	a5, a1, a2
   1f9e8: 63 8b 07 28  	beqz	a5, 0x1fc7e <__subtf3+0x922>
   1f9ec: 81 47        	mv	a5, zero
   1f9ee: 63 10 87 2a  	bne	a4, s0, 0x1fc8e <__subtf3+0x932>
   1f9f2: e3 05 08 d0  	beqz	a6, 0x1f6fc <__subtf3+0x3a0>
   1f9f6: 75 8e        	and	a2, a2, a3
   1f9f8: e3 13 06 cc  	bnez	a2, 0x1f6be <__subtf3+0x362>
   1f9fc: c1 47        	addi	a5, zero, 16
   1f9fe: 41 ac        	j	0x1fc8e <__subtf3+0x932>
   1fa00: 81 47        	mv	a5, zero
   1fa02: 63 15 c7 03  	bne	a4, t3, 0x1fa2c <__subtf3+0x6d0>
   1fa06: 63 19 08 00  	bnez	a6, 0x1fa18 <__subtf3+0x6bc>
   1fa0a: e3 99 08 ce  	bnez	a7, 0x1f6fc <__subtf3+0x3a0>
   1fa0e: 85 45        	addi	a1, zero, 1
   1fa10: 81 44        	mv	s1, zero
   1fa12: ca 15        	slli	a1, a1, 50
   1fa14: c1 47        	addi	a5, zero, 16
   1fa16: 55 b9        	j	0x1f6ca <__subtf3+0x36e>
   1fa18: 13 de 26 03  	srli	t3, a3, 50
   1fa1c: 13 7e 1e 00  	andi	t3, t3, 1
   1fa20: 63 05 0e 00  	beqz	t3, 0x1fa2a <__subtf3+0x6ce>
   1fa24: e3 9d 08 c8  	bnez	a7, 0x1f6be <__subtf3+0x362>
   1fa28: 55 bf        	j	0x1f9dc <__subtf3+0x680>
   1fa2a: c1 47        	addi	a5, zero, 16
   1fa2c: 63 91 08 26  	bnez	a7, 0x1fc8e <__subtf3+0x932>
   1fa30: 9a 84        	mv	s1, t1
   1fa32: 63 13 08 24  	bnez	a6, 0x1fc78 <__subtf3+0x91c>
   1fa36: 85 45        	addi	a1, zero, 1
   1fa38: 21 67        	lui	a4, 8
   1fa3a: 81 44        	mv	s1, zero
   1fa3c: ca 15        	slli	a1, a1, 50
   1fa3e: 7d 17        	addi	a4, a4, -1
   1fa40: d1 bf        	j	0x1fa14 <__subtf3+0x6b8>
   1fa42: c1 47        	addi	a5, zero, 16
   1fa44: f5 b7        	j	0x1fa30 <__subtf3+0x6d4>
   1fa46: b3 07 c5 40  	sub	a5, a0, a2
   1fa4a: 33 37 f5 00  	sltu	a4, a0, a5
   1fa4e: b3 89 d5 40  	sub	s3, a1, a3
   1fa52: b3 89 e9 40  	sub	s3, s3, a4
   1fa56: 13 d7 39 03  	srli	a4, s3, 51
   1fa5a: 05 8b        	andi	a4, a4, 1
   1fa5c: 3e 8a        	mv	s4, a5
   1fa5e: 45 cf        	beqz	a4, 0x1fb16 <__subtf3+0x7ba>
   1fa60: 33 0a a6 40  	sub	s4, a2, a0
   1fa64: b3 89 b6 40  	sub	s3, a3, a1
   1fa68: 33 36 46 01  	sltu	a2, a2, s4
   1fa6c: b3 89 c9 40  	sub	s3, s3, a2
   1fa70: 9a 84        	mv	s1, t1
   1fa72: 63 86 09 0a  	beqz	s3, 0x1fb1e <__subtf3+0x7c2>
   1fa76: 4e 85        	mv	a0, s3
   1fa78: ef 00 00 64  	jal	0x200b8 <__clzdi2>
   1fa7c: 1b 07 45 ff  	addiw	a4, a0, -12
   1fa80: 93 06 00 04  	addi	a3, zero, 64
   1fa84: 99 9e        	subw	a3, a3, a4
   1fa86: b3 99 e9 00  	sll	s3, s3, a4
   1fa8a: b3 56 da 00  	srl	a3, s4, a3
   1fa8e: b3 e6 36 01  	or	a3, a3, s3
   1fa92: b3 19 ea 00  	sll	s3, s4, a4
   1fa96: 63 4c 87 0c  	blt	a4, s0, 0x1fb6e <__subtf3+0x812>
   1fa9a: 01 9f        	subw	a4, a4, s0
   1fa9c: 05 27        	addiw	a4, a4, 1
   1fa9e: 93 07 00 04  	addi	a5, zero, 64
   1faa2: 99 9f        	subw	a5, a5, a4
   1faa4: 33 95 f6 00  	sll	a0, a3, a5
   1faa8: 33 d6 e9 00  	srl	a2, s3, a4
   1faac: b3 97 f9 00  	sll	a5, s3, a5
   1fab0: 51 8d        	or	a0, a0, a2
   1fab2: b3 37 f0 00  	snez	a5, a5
   1fab6: 5d 8d        	or	a0, a0, a5
   1fab8: b3 d5 e6 00  	srl	a1, a3, a4
   1fabc: b3 67 b5 00  	or	a5, a0, a1
   1fac0: e3 8b 07 ea  	beqz	a5, 0x1f976 <__subtf3+0x61a>
   1fac4: 93 57 f5 03  	srli	a5, a0, 63
   1fac8: 13 97 15 00  	slli	a4, a1, 1
   1facc: 3e 97        	add	a4, a4, a5
   1face: 93 17 15 00  	slli	a5, a0, 1
   1fad2: 13 f6 77 00  	andi	a2, a5, 7
   1fad6: 93 76 75 00  	andi	a3, a0, 7
   1fada: 63 0f 06 0e  	beqz	a2, 0x1fbd8 <__subtf3+0x87c>
   1fade: 09 46        	addi	a2, zero, 2
   1fae0: 63 07 c9 0c  	beq	s2, a2, 0x1fbae <__subtf3+0x852>
   1fae4: 05 46        	addi	a2, zero, 1
   1fae6: 0d 48        	addi	a6, zero, 3
   1fae8: 52 16        	slli	a2, a2, 52
   1faea: 63 00 09 0b  	beq	s2, a6, 0x1fb8a <__subtf3+0x82e>
   1faee: 63 1f 09 08  	bnez	s2, 0x1fb8c <__subtf3+0x830>
   1faf2: 13 f8 f7 00  	andi	a6, a5, 15
   1faf6: 91 48        	addi	a7, zero, 4
   1faf8: 63 0a 18 09  	beq	a6, a7, 0x1fb8c <__subtf3+0x830>
   1fafc: 93 b7 c7 ff  	sltiu	a5, a5, -4
   1fb00: 93 c7 17 00  	xori	a5, a5, 1
   1fb04: 82 17        	slli	a5, a5, 32
   1fb06: 81 93        	srli	a5, a5, 32
   1fb08: ba 97        	add	a5, a5, a4
   1fb0a: f1 8f        	and	a5, a5, a2
   1fb0c: e5 e3        	bnez	a5, 0x1fbec <__subtf3+0x890>
   1fb0e: f9 c2        	beqz	a3, 0x1fbd4 <__subtf3+0x878>
   1fb10: 85 46        	addi	a3, zero, 1
   1fb12: 01 47        	mv	a4, zero
   1fb14: 01 a2        	j	0x1fc14 <__subtf3+0x8b8>
   1fb16: b3 e7 37 01  	or	a5, a5, s3
   1fb1a: a1 ff        	bnez	a5, 0x1fa72 <__subtf3+0x716>
   1fb1c: 89 bd        	j	0x1f96e <__subtf3+0x612>
   1fb1e: 52 85        	mv	a0, s4
   1fb20: ef 00 80 59  	jal	0x200b8 <__clzdi2>
   1fb24: 1b 07 45 03  	addiw	a4, a0, 52
   1fb28: 93 05 f0 03  	addi	a1, zero, 63
   1fb2c: e3 da e5 f4  	bge	a1, a4, 0x1fa80 <__subtf3+0x724>
   1fb30: 9b 06 45 ff  	addiw	a3, a0, -12
   1fb34: b3 16 da 00  	sll	a3, s4, a3
   1fb38: 63 4b 87 02  	blt	a4, s0, 0x1fb6e <__subtf3+0x812>
   1fb3c: bb 07 87 40  	subw	a5, a4, s0
   1fb40: 1b 87 17 00  	addiw	a4, a5, 1
   1fb44: e3 dd e5 f4  	bge	a1, a4, 0x1fa9e <__subtf3+0x742>
   1fb48: 9b 87 17 fc  	addiw	a5, a5, -63
   1fb4c: 93 05 00 04  	addi	a1, zero, 64
   1fb50: b3 d7 f6 00  	srl	a5, a3, a5
   1fb54: 01 45        	mv	a0, zero
   1fb56: 63 07 b7 00  	beq	a4, a1, 0x1fb64 <__subtf3+0x808>
   1fb5a: 13 05 00 08  	addi	a0, zero, 128
   1fb5e: 19 9d        	subw	a0, a0, a4
   1fb60: 33 95 a6 00  	sll	a0, a3, a0
   1fb64: 33 35 a0 00  	snez	a0, a0
   1fb68: 5d 8d        	or	a0, a0, a5
   1fb6a: 81 45        	mv	a1, zero
   1fb6c: 81 bf        	j	0x1fabc <__subtf3+0x760>
   1fb6e: fd 57        	addi	a5, zero, -1
   1fb70: ce 17        	slli	a5, a5, 51
   1fb72: fd 17        	addi	a5, a5, -1
   1fb74: 33 07 e4 40  	sub	a4, s0, a4
   1fb78: fd 8e        	and	a3, a3, a5
   1fb7a: 4e 86        	mv	a2, s3
   1fb7c: c5 bc        	j	0x1f66c <__subtf3+0x310>
   1fb7e: b6 85        	mv	a1, a3
   1fb80: 32 85        	mv	a0, a2
   1fb82: 2d bf        	j	0x1fabc <__subtf3+0x760>
   1fb84: b6 85        	mv	a1, a3
   1fb86: 32 85        	mv	a0, a2
   1fb88: 1d b5        	j	0x1f9ae <__subtf3+0x652>
   1fb8a: 89 c4        	beqz	s1, 0x1fb94 <__subtf3+0x838>
   1fb8c: 71 8f        	and	a4, a4, a2
   1fb8e: 39 ef        	bnez	a4, 0x1fbec <__subtf3+0x890>
   1fb90: c1 f2        	bnez	a3, 0x1fb10 <__subtf3+0x7b4>
   1fb92: 21 a8        	j	0x1fbaa <__subtf3+0x84e>
   1fb94: 93 b7 87 ff  	sltiu	a5, a5, -8
   1fb98: 93 c7 17 00  	xori	a5, a5, 1
   1fb9c: 82 17        	slli	a5, a5, 32
   1fb9e: 81 93        	srli	a5, a5, 32
   1fba0: ba 97        	add	a5, a5, a4
   1fba2: f1 8f        	and	a5, a5, a2
   1fba4: a1 e7        	bnez	a5, 0x1fbec <__subtf3+0x890>
   1fba6: 01 47        	mv	a4, zero
   1fba8: ad e6        	bnez	a3, 0x1fc12 <__subtf3+0x8b6>
   1fbaa: 85 47        	addi	a5, zero, 1
   1fbac: 31 a0        	j	0x1fbb8 <__subtf3+0x85c>
   1fbae: 81 e8        	bnez	s1, 0x1fbbe <__subtf3+0x862>
   1fbb0: 51 93        	srli	a4, a4, 52
   1fbb2: 05 8b        	andi	a4, a4, 1
   1fbb4: 85 47        	addi	a5, zero, 1
   1fbb6: 05 ef        	bnez	a4, 0x1fbee <__subtf3+0x892>
   1fbb8: 93 e7 27 00  	ori	a5, a5, 2
   1fbbc: e1 b8        	j	0x1f494 <__subtf3+0x138>
   1fbbe: 93 b7 87 ff  	sltiu	a5, a5, -8
   1fbc2: 93 c7 17 00  	xori	a5, a5, 1
   1fbc6: 82 17        	slli	a5, a5, 32
   1fbc8: 81 93        	srli	a5, a5, 32
   1fbca: ba 97        	add	a5, a5, a4
   1fbcc: d1 93        	srli	a5, a5, 52
   1fbce: 85 8b        	andi	a5, a5, 1
   1fbd0: 91 ef        	bnez	a5, 0x1fbec <__subtf3+0x890>
   1fbd2: b1 e6        	bnez	a3, 0x1fc1e <__subtf3+0x8c2>
   1fbd4: 01 47        	mv	a4, zero
   1fbd6: d1 bf        	j	0x1fbaa <__subtf3+0x84e>
   1fbd8: 51 93        	srli	a4, a4, 52
   1fbda: 05 8b        	andi	a4, a4, 1
   1fbdc: 15 e7        	bnez	a4, 0x1fc08 <__subtf3+0x8ac>
   1fbde: 81 47        	mv	a5, zero
   1fbe0: e3 8a 06 8a  	beqz	a3, 0x1f494 <__subtf3+0x138>
   1fbe4: 85 46        	addi	a3, zero, 1
   1fbe6: 01 47        	mv	a4, zero
   1fbe8: 81 47        	mv	a5, zero
   1fbea: 31 a0        	j	0x1fbf6 <__subtf3+0x89a>
   1fbec: 85 47        	addi	a5, zero, 1
   1fbee: 01 47        	mv	a4, zero
   1fbf0: e3 82 06 8a  	beqz	a3, 0x1f494 <__subtf3+0x138>
   1fbf4: 81 46        	mv	a3, zero
   1fbf6: 09 46        	addi	a2, zero, 2
   1fbf8: 93 e7 17 00  	ori	a5, a5, 1
   1fbfc: e3 1f c9 b2  	bne	s2, a2, 0x1f73a <__subtf3+0x3de>
   1fc00: 89 cc        	beqz	s1, 0x1fc1a <__subtf3+0x8be>
   1fc02: 13 06 85 00  	addi	a2, a0, 8
   1fc06: b1 b6        	j	0x1f752 <__subtf3+0x3f6>
   1fc08: 81 47        	mv	a5, zero
   1fc0a: d5 b7        	j	0x1fbee <__subtf3+0x892>
   1fc0c: b6 85        	mv	a1, a3
   1fc0e: 32 85        	mv	a0, a2
   1fc10: d5 b7        	j	0x1fbf4 <__subtf3+0x898>
   1fc12: 85 46        	addi	a3, zero, 1
   1fc14: 85 47        	addi	a5, zero, 1
   1fc16: 15 b6        	j	0x1f73a <__subtf3+0x3de>
   1fc18: ed d4        	beqz	s1, 0x1fc02 <__subtf3+0x8a6>
   1fc1a: d9 fe        	bnez	a3, 0x1fbb8 <__subtf3+0x85c>
   1fc1c: a5 b8        	j	0x1f494 <__subtf3+0x138>
   1fc1e: 85 46        	addi	a3, zero, 1
   1fc20: 01 47        	mv	a4, zero
   1fc22: 85 47        	addi	a5, zero, 1
   1fc24: f1 bf        	j	0x1fc00 <__subtf3+0x8a4>
   1fc26: 01 48        	mv	a6, zero
   1fc28: 63 06 09 00  	beqz	s2, 0x1fc34 <__subtf3+0x8d8>
   1fc2c: 0d 46        	addi	a2, zero, 3
   1fc2e: 63 17 c9 00  	bne	s2, a2, 0x1fc3c <__subtf3+0x8e0>
   1fc32: 89 e8        	bnez	s1, 0x1fc44 <__subtf3+0x8e8>
   1fc34: 93 e7 57 00  	ori	a5, a5, 5
   1fc38: c2 85        	mv	a1, a6
   1fc3a: 41 bc        	j	0x1f6ca <__subtf3+0x36e>
   1fc3c: 09 46        	addi	a2, zero, 2
   1fc3e: 63 13 c9 00  	bne	s2, a2, 0x1fc44 <__subtf3+0x8e8>
   1fc42: ed f8        	bnez	s1, 0x1fc34 <__subtf3+0x8d8>
   1fc44: 7d 58        	addi	a6, zero, -1
   1fc46: 13 87 e6 ff  	addi	a4, a3, -2
   1fc4a: ed b7        	j	0x1fc34 <__subtf3+0x8d8>
   1fc4c: 3e 87        	mv	a4, a5
   1fc4e: 81 45        	mv	a1, zero
   1fc50: 01 48        	mv	a6, zero
   1fc52: 95 47        	addi	a5, zero, 5
   1fc54: 9d bc        	j	0x1f6ca <__subtf3+0x36e>
   1fc56: 81 45        	mv	a1, zero
   1fc58: 22 87        	mv	a4, s0
   1fc5a: 0d b3        	j	0x1f97c <__subtf3+0x620>
   1fc5c: 2a 88        	mv	a6, a0
   1fc5e: b5 b4        	j	0x1f6ca <__subtf3+0x36e>
   1fc60: c1 47        	addi	a5, zero, 16
   1fc62: e3 1b 87 a8  	bne	a4, s0, 0x1f6f8 <__subtf3+0x39c>
   1fc66: e3 0a 03 a2  	beqz	t1, 0x1f69a <__subtf3+0x33e>
   1fc6a: 13 d7 26 03  	srli	a4, a3, 50
   1fc6e: 05 8b        	andi	a4, a4, 1
   1fc70: e3 01 07 a8  	beqz	a4, 0x1f6f2 <__subtf3+0x396>
   1fc74: e3 15 08 a4  	bnez	a6, 0x1f6be <__subtf3+0x362>
   1fc78: 21 67        	lui	a4, 8
   1fc7a: 7d 17        	addi	a4, a4, -1
   1fc7c: 31 b0        	j	0x1f488 <__subtf3+0x12c>
   1fc7e: c1 47        	addi	a5, zero, 16
   1fc80: 63 17 87 00  	bne	a4, s0, 0x1fc8e <__subtf3+0x932>
   1fc84: e3 0c 08 a6  	beqz	a6, 0x1f6fc <__subtf3+0x3a0>
   1fc88: 75 8e        	and	a2, a2, a3
   1fc8a: e3 1a 06 a2  	bnez	a2, 0x1f6be <__subtf3+0x362>
   1fc8e: e3 18 08 a2  	bnez	a6, 0x1f6be <__subtf3+0x362>
   1fc92: ad b4        	j	0x1f6fc <__subtf3+0x3a0>

000000000001fc94 <__fixtfsi>:
   1fc94: 2a 86        	mv	a2, a0
   1fc96: f3 27 20 00  	frrm	a5
   1fc9a: 11 68        	lui	a6, 4
   1fc9c: 93 96 15 00  	slli	a3, a1, 1
   1fca0: 93 97 05 01  	slli	a5, a1, 16
   1fca4: c5 92        	srli	a3, a3, 49
   1fca6: 13 07 e8 ff  	addi	a4, a6, -2
   1fcaa: c1 83        	srli	a5, a5, 16
   1fcac: fd 91        	srli	a1, a1, 63
   1fcae: 63 47 d7 00  	blt	a4, a3, 0x1fcbc <__fixtfsi+0x28>
   1fcb2: bd e2        	bnez	a3, 0x1fd18 <__fixtfsi+0x84>
   1fcb4: c9 8f        	or	a5, a5, a0
   1fcb6: 01 45        	mv	a0, zero
   1fcb8: b1 ef        	bnez	a5, 0x1fd14 <__fixtfsi+0x80>
   1fcba: 82 80        	ret
   1fcbc: 13 07 d8 01  	addi	a4, a6, 29
   1fcc0: 63 54 d7 02  	bge	a4, a3, 0x1fce8 <__fixtfsi+0x54>
   1fcc4: 37 07 00 80  	lui	a4, 524288
   1fcc8: 13 47 f7 ff  	not	a4, a4
   1fccc: 3b 05 b7 00  	addw	a0, a4, a1
   1fcd0: b1 c5        	beqz	a1, 0x1fd1c <__fixtfsi+0x88>
   1fcd2: 79 08        	addi	a6, a6, 30
   1fcd4: 41 47        	addi	a4, zero, 16
   1fcd6: 63 94 06 05  	bne	a3, a6, 0x1fd1e <__fixtfsi+0x8a>
   1fcda: 93 d6 17 01  	srli	a3, a5, 17
   1fcde: a1 e2        	bnez	a3, 0x1fd1e <__fixtfsi+0x8a>
   1fce0: be 17        	slli	a5, a5, 47
   1fce2: d1 8f        	or	a5, a5, a2
   1fce4: 85 eb        	bnez	a5, 0x1fd14 <__fixtfsi+0x80>
   1fce6: 82 80        	ret
   1fce8: 05 47        	addi	a4, zero, 1
   1fcea: 42 17        	slli	a4, a4, 48
   1fcec: d9 8f        	or	a5, a5, a4
   1fcee: 11 66        	lui	a2, 4
   1fcf0: 71 77        	lui	a4, 1048572
   1fcf2: 45 27        	addiw	a4, a4, 17
   1fcf4: 1b 06 f6 02  	addiw	a2, a2, 47
   1fcf8: 35 9f        	addw	a4, a4, a3
   1fcfa: bb 06 d6 40  	subw	a3, a2, a3
   1fcfe: 33 97 e7 00  	sll	a4, a5, a4
   1fd02: b3 d7 d7 00  	srl	a5, a5, a3
   1fd06: 49 8f        	or	a4, a4, a0
   1fd08: 1b 85 07 00  	sext.w	a0, a5
   1fd0c: 99 c1        	beqz	a1, 0x1fd12 <__fixtfsi+0x7e>
   1fd0e: 3b 05 a0 40  	negw	a0, a0
   1fd12: 01 cb        	beqz	a4, 0x1fd22 <__fixtfsi+0x8e>
   1fd14: 05 47        	addi	a4, zero, 1
   1fd16: 21 a0        	j	0x1fd1e <__fixtfsi+0x8a>
   1fd18: 01 45        	mv	a0, zero
   1fd1a: ed bf        	j	0x1fd14 <__fixtfsi+0x80>
   1fd1c: 41 47        	addi	a4, zero, 16
   1fd1e: 73 20 17 00  	csrs	fflags, a4
   1fd22: 82 80        	ret

000000000001fd24 <__floatsitf>:
   1fd24: 01 11        	addi	sp, sp, -32
   1fd26: 06 ec        	sd	ra, 24(sp)
   1fd28: 22 e8        	sd	s0, 16(sp)
   1fd2a: 26 e4        	sd	s1, 8(sp)
   1fd2c: 39 c1        	beqz	a0, 0x1fd72 <__floatsitf+0x4e>
   1fd2e: 9b 07 05 00  	sext.w	a5, a0
   1fd32: 93 54 f5 03  	srli	s1, a0, 63
   1fd36: 63 54 05 00  	bgez	a0, 0x1fd3e <__floatsitf+0x1a>
   1fd3a: bb 07 f0 40  	negw	a5, a5
   1fd3e: 13 94 07 02  	slli	s0, a5, 32
   1fd42: 01 90        	srli	s0, s0, 32
   1fd44: 22 85        	mv	a0, s0
   1fd46: ef 00 20 37  	jal	0x200b8 <__clzdi2>
   1fd4a: 91 65        	lui	a1, 4
   1fd4c: 9b 85 e5 03  	addiw	a1, a1, 62
   1fd50: 89 9d        	subw	a1, a1, a0
   1fd52: 45 35        	addiw	a0, a0, -15
   1fd54: b3 17 a4 00  	sll	a5, s0, a0
   1fd58: e2 60        	ld	ra, 24(sp)
   1fd5a: 42 64        	ld	s0, 16(sp)
   1fd5c: 9b 94 f4 00  	slliw	s1, s1, 15
   1fd60: c5 8d        	or	a1, a1, s1
   1fd62: c2 07        	slli	a5, a5, 16
   1fd64: c2 15        	slli	a1, a1, 48
   1fd66: c1 83        	srli	a5, a5, 16
   1fd68: a2 64        	ld	s1, 8(sp)
   1fd6a: 01 45        	mv	a0, zero
   1fd6c: dd 8d        	or	a1, a1, a5
   1fd6e: 05 61        	addi	sp, sp, 32
   1fd70: 82 80        	ret
   1fd72: 81 47        	mv	a5, zero
   1fd74: 81 45        	mv	a1, zero
   1fd76: 81 44        	mv	s1, zero
   1fd78: c5 b7        	j	0x1fd58 <__floatsitf+0x34>

000000000001fd7a <__extenddftf2>:
   1fd7a: 01 11        	addi	sp, sp, -32
   1fd7c: d3 07 05 e2  	fmv.x.d	a5, fa0
   1fd80: 06 ec        	sd	ra, 24(sp)
   1fd82: 22 e8        	sd	s0, 16(sp)
   1fd84: 26 e4        	sd	s1, 8(sp)
   1fd86: 73 27 20 00  	frrm	a4
   1fd8a: 13 d5 47 03  	srli	a0, a5, 52
   1fd8e: 13 75 f5 7f  	andi	a0, a0, 2047
   1fd92: 13 94 c7 00  	slli	s0, a5, 12
   1fd96: 93 d4 f7 03  	srli	s1, a5, 63
   1fd9a: 93 07 15 00  	addi	a5, a0, 1
   1fd9e: 93 f7 e7 7f  	andi	a5, a5, 2046
   1fda2: 31 80        	srli	s0, s0, 12
   1fda4: 91 cb        	beqz	a5, 0x1fdb8 <__extenddftf2+0x3e>
   1fda6: 91 67        	lui	a5, 4
   1fda8: 93 87 07 c0  	addi	a5, a5, -1024
   1fdac: 3e 95        	add	a0, a0, a5
   1fdae: 93 57 44 00  	srli	a5, s0, 4
   1fdb2: 72 14        	slli	s0, s0, 60
   1fdb4: 01 47        	mv	a4, zero
   1fdb6: b9 a8        	j	0x1fe14 <__extenddftf2+0x9a>
   1fdb8: 1d ed        	bnez	a0, 0x1fdf6 <__extenddftf2+0x7c>
   1fdba: 81 47        	mv	a5, zero
   1fdbc: 65 dc        	beqz	s0, 0x1fdb4 <__extenddftf2+0x3a>
   1fdbe: 22 85        	mv	a0, s0
   1fdc0: ef 00 80 2f  	jal	0x200b8 <__clzdi2>
   1fdc4: 1b 07 05 00  	sext.w	a4, a0
   1fdc8: b9 47        	addi	a5, zero, 14
   1fdca: 63 c0 e7 02  	blt	a5, a4, 0x1fdea <__extenddftf2+0x70>
   1fdce: bd 47        	addi	a5, zero, 15
   1fdd0: 89 9f        	subw	a5, a5, a0
   1fdd2: 1b 07 15 03  	addiw	a4, a0, 49
   1fdd6: b3 57 f4 00  	srl	a5, s0, a5
   1fdda: 33 14 e4 00  	sll	s0, s0, a4
   1fdde: 11 67        	lui	a4, 4
   1fde0: 1b 07 c7 c0  	addiw	a4, a4, -1012
   1fde4: 3b 05 a7 40  	subw	a0, a4, a0
   1fde8: f1 b7        	j	0x1fdb4 <__extenddftf2+0x3a>
   1fdea: 9b 07 15 ff  	addiw	a5, a0, -15
   1fdee: b3 17 f4 00  	sll	a5, s0, a5
   1fdf2: 01 44        	mv	s0, zero
   1fdf4: ed b7        	j	0x1fdde <__extenddftf2+0x64>
   1fdf6: 21 c0        	beqz	s0, 0x1fe36 <__extenddftf2+0xbc>
   1fdf8: 85 46        	addi	a3, zero, 1
   1fdfa: 13 97 36 03  	slli	a4, a3, 51
   1fdfe: 33 37 e4 00  	sltu	a4, s0, a4
   1fe02: 93 57 44 00  	srli	a5, s0, 4
   1fe06: be 16        	slli	a3, a3, 47
   1fe08: 21 65        	lui	a0, 8
   1fe0a: 1b 17 47 00  	slliw	a4, a4, 4
   1fe0e: d5 8f        	or	a5, a5, a3
   1fe10: 72 14        	slli	s0, s0, 60
   1fe12: 7d 15        	addi	a0, a0, -1
   1fe14: 93 95 07 01  	slli	a1, a5, 16
   1fe18: 9b 97 f4 00  	slliw	a5, s1, 15
   1fe1c: 5d 8d        	or	a0, a0, a5
   1fe1e: 42 15        	slli	a0, a0, 48
   1fe20: c1 81        	srli	a1, a1, 16
   1fe22: c9 8d        	or	a1, a1, a0
   1fe24: 19 c3        	beqz	a4, 0x1fe2a <__extenddftf2+0xb0>
   1fe26: 73 60 18 00  	csrsi	fflags, 16
   1fe2a: e2 60        	ld	ra, 24(sp)
   1fe2c: 22 85        	mv	a0, s0
   1fe2e: 42 64        	ld	s0, 16(sp)
   1fe30: a2 64        	ld	s1, 8(sp)
   1fe32: 05 61        	addi	sp, sp, 32
   1fe34: 82 80        	ret
   1fe36: 21 65        	lui	a0, 8
   1fe38: 81 47        	mv	a5, zero
   1fe3a: 7d 15        	addi	a0, a0, -1
   1fe3c: a5 bf        	j	0x1fdb4 <__extenddftf2+0x3a>

000000000001fe3e <__trunctfdf2>:
   1fe3e: 73 28 20 00  	frrm	a6
   1fe42: a1 68        	lui	a7, 8
   1fe44: 13 d7 05 03  	srli	a4, a1, 48
   1fe48: 13 8e f8 ff  	addi	t3, a7, -1
   1fe4c: fd 57        	addi	a5, zero, -1
   1fe4e: 93 d6 07 01  	srli	a3, a5, 16
   1fe52: 33 77 c7 01  	and	a4, a4, t3
   1fe56: 13 d6 f5 03  	srli	a2, a1, 63
   1fe5a: 13 03 17 00  	addi	t1, a4, 1
   1fe5e: f5 8d        	and	a1, a1, a3
   1fe60: f9 18        	addi	a7, a7, -2
   1fe62: 8e 05        	slli	a1, a1, 3
   1fe64: 93 56 d5 03  	srli	a3, a0, 61
   1fe68: b3 78 13 01  	and	a7, t1, a7
   1fe6c: cd 8e        	or	a3, a3, a1
   1fe6e: 01 28        	sext.w	a6, a6
   1fe70: 93 15 35 00  	slli	a1, a0, 3
   1fe74: 63 81 08 10  	beqz	a7, 0x1ff76 <__trunctfdf2+0x138>
   1fe78: f1 78        	lui	a7, 1048572
   1fe7a: 93 88 08 40  	addi	a7, a7, 1024
   1fe7e: 46 97        	add	a4, a4, a7
   1fe80: 93 08 e0 7f  	addi	a7, zero, 2046
   1fe84: 63 d4 e8 02  	bge	a7, a4, 0x1feac <__trunctfdf2+0x6e>
   1fe88: 63 05 08 20  	beqz	a6, 0x20092 <__trunctfdf2+0x254>
   1fe8c: 0d 47        	addi	a4, zero, 3
   1fe8e: 63 18 e8 00  	bne	a6, a4, 0x1fe9e <__trunctfdf2+0x60>
   1fe92: 63 0f 06 20  	beqz	a2, 0x200b0 <__trunctfdf2+0x272>
   1fe96: 13 07 e0 7f  	addi	a4, zero, 2046
   1fe9a: 15 45        	addi	a0, zero, 5
   1fe9c: 59 a2        	j	0x20022 <__trunctfdf2+0x1e4>
   1fe9e: 09 47        	addi	a4, zero, 2
   1fea0: 63 14 e8 00  	bne	a6, a4, 0x1fea8 <__trunctfdf2+0x6a>
   1fea4: 63 18 06 20  	bnez	a2, 0x200b4 <__trunctfdf2+0x276>
   1fea8: fd 57        	addi	a5, zero, -1
   1feaa: f5 b7        	j	0x1fe96 <__trunctfdf2+0x58>
   1feac: 63 47 e0 0a  	bgtz	a4, 0x1ff5a <__trunctfdf2+0x11c>
   1feb0: 93 07 c0 fc  	addi	a5, zero, -52
   1feb4: 63 46 f7 0c  	blt	a4, a5, 0x1ff80 <__trunctfdf2+0x142>
   1feb8: 85 47        	addi	a5, zero, 1
   1feba: ce 17        	slli	a5, a5, 51
   1febc: 93 08 d0 03  	addi	a7, zero, 61
   1fec0: dd 8e        	or	a3, a3, a5
   1fec2: b3 88 e8 40  	sub	a7, a7, a4
   1fec6: 93 07 f0 03  	addi	a5, zero, 63
   1feca: 01 27        	sext.w	a4, a4
   1fecc: 63 c5 17 07  	blt	a5, a7, 0x1ff36 <__trunctfdf2+0xf8>
   1fed0: 93 07 d0 03  	addi	a5, zero, 61
   1fed4: 1b 05 37 00  	addiw	a0, a4, 3
   1fed8: 99 9f        	subw	a5, a5, a4
   1feda: b3 d7 f5 00  	srl	a5, a1, a5
   1fede: b3 95 a5 00  	sll	a1, a1, a0
   1fee2: b3 35 b0 00  	snez	a1, a1
   1fee6: cd 8f        	or	a5, a5, a1
   1fee8: b3 96 a6 00  	sll	a3, a3, a0
   1feec: d5 8f        	or	a5, a5, a3
   1feee: 13 97 17 00  	slli	a4, a5, 1
   1fef2: 93 76 77 00  	andi	a3, a4, 7
   1fef6: d9 e6        	bnez	a3, 0x1ff84 <__trunctfdf2+0x146>
   1fef8: 93 f5 77 00  	andi	a1, a5, 7
   1fefc: 01 47        	mv	a4, zero
   1fefe: 85 46        	addi	a3, zero, 1
   1ff00: 01 45        	mv	a0, zero
   1ff02: 85 c1        	beqz	a1, 0x1ff22 <__trunctfdf2+0xe4>
   1ff04: 89 45        	addi	a1, zero, 2
   1ff06: 63 0f b8 12  	beq	a6, a1, 0x20044 <__trunctfdf2+0x206>
   1ff0a: 8d 45        	addi	a1, zero, 3
   1ff0c: 63 08 b8 10  	beq	a6, a1, 0x2001c <__trunctfdf2+0x1de>
   1ff10: 63 1d 08 12  	bnez	a6, 0x2004a <__trunctfdf2+0x20c>
   1ff14: 93 f5 f7 00  	andi	a1, a5, 15
   1ff18: 11 45        	addi	a0, zero, 4
   1ff1a: 63 9b a5 0e  	bne	a1, a0, 0x20010 <__trunctfdf2+0x1d2>
   1ff1e: 05 45        	addi	a0, zero, 1
   1ff20: e9 e6        	bnez	a3, 0x1ffea <__trunctfdf2+0x1ac>
   1ff22: 93 06 f0 7f  	addi	a3, zero, 2047
   1ff26: 8d 83        	srli	a5, a5, 3
   1ff28: 63 1b d7 08  	bne	a4, a3, 0x1ffbe <__trunctfdf2+0x180>
   1ff2c: c9 cb        	beqz	a5, 0x1ffbe <__trunctfdf2+0x180>
   1ff2e: 85 47        	addi	a5, zero, 1
   1ff30: ce 17        	slli	a5, a5, 51
   1ff32: 01 46        	mv	a2, zero
   1ff34: 69 a0        	j	0x1ffbe <__trunctfdf2+0x180>
   1ff36: 75 55        	addi	a0, zero, -3
   1ff38: 19 9d        	subw	a0, a0, a4
   1ff3a: 13 03 00 04  	addi	t1, zero, 64
   1ff3e: 33 d5 a6 00  	srl	a0, a3, a0
   1ff42: 81 47        	mv	a5, zero
   1ff44: 63 86 68 00  	beq	a7, t1, 0x1ff50 <__trunctfdf2+0x112>
   1ff48: 1b 07 37 04  	addiw	a4, a4, 67
   1ff4c: b3 97 e6 00  	sll	a5, a3, a4
   1ff50: cd 8f        	or	a5, a5, a1
   1ff52: b3 37 f0 00  	snez	a5, a5
   1ff56: c9 8f        	or	a5, a5, a0
   1ff58: 59 bf        	j	0x1feee <__trunctfdf2+0xb0>
   1ff5a: 1e 05        	slli	a0, a0, 7
   1ff5c: 33 35 a0 00  	snez	a0, a0
   1ff60: f1 91        	srli	a1, a1, 60
   1ff62: 4d 8d        	or	a0, a0, a1
   1ff64: 93 97 46 00  	slli	a5, a3, 4
   1ff68: c9 8f        	or	a5, a5, a0
   1ff6a: 1d 89        	andi	a0, a0, 7
   1ff6c: 81 46        	mv	a3, zero
   1ff6e: 59 f9        	bnez	a0, 0x1ff04 <__trunctfdf2+0xc6>
   1ff70: 8d 83        	srli	a5, a5, 3
   1ff72: 01 45        	mv	a0, zero
   1ff74: a9 a0        	j	0x1ffbe <__trunctfdf2+0x180>
   1ff76: d5 8d        	or	a1, a1, a3
   1ff78: 05 e7        	bnez	a4, 0x1ffa0 <__trunctfdf2+0x162>
   1ff7a: 93 17 f6 03  	slli	a5, a2, 63
   1ff7e: b1 c9        	beqz	a1, 0x1ffd2 <__trunctfdf2+0x194>
   1ff80: 85 47        	addi	a5, zero, 1
   1ff82: 09 47        	addi	a4, zero, 2
   1ff84: 89 46        	addi	a3, zero, 2
   1ff86: 13 f5 77 00  	andi	a0, a5, 7
   1ff8a: 63 01 d8 08  	beq	a6, a3, 0x2000c <__trunctfdf2+0x1ce>
   1ff8e: 8d 46        	addi	a3, zero, 3
   1ff90: 63 0f d8 04  	beq	a6, a3, 0x1ffee <__trunctfdf2+0x1b0>
   1ff94: 63 02 08 04  	beqz	a6, 0x1ffd8 <__trunctfdf2+0x19a>
   1ff98: 01 47        	mv	a4, zero
   1ff9a: 85 46        	addi	a3, zero, 1
   1ff9c: 25 f5        	bnez	a0, 0x1ff04 <__trunctfdf2+0xc6>
   1ff9e: b1 a0        	j	0x1ffea <__trunctfdf2+0x1ac>
   1ffa0: 63 81 05 10  	beqz	a1, 0x200a2 <__trunctfdf2+0x264>
   1ffa4: 01 45        	mv	a0, zero
   1ffa6: 63 17 c7 01  	bne	a4, t3, 0x1ffb4 <__trunctfdf2+0x176>
   1ffaa: c9 92        	srli	a3, a3, 50
   1ffac: 93 c6 16 00  	xori	a3, a3, 1
   1ffb0: 13 95 46 00  	slli	a0, a3, 4
   1ffb4: 85 47        	addi	a5, zero, 1
   1ffb6: 01 46        	mv	a2, zero
   1ffb8: 13 07 f0 7f  	addi	a4, zero, 2047
   1ffbc: ce 17        	slli	a5, a5, 51
   1ffbe: 13 77 f7 7f  	andi	a4, a4, 2047
   1ffc2: b2 07        	slli	a5, a5, 12
   1ffc4: b1 83        	srli	a5, a5, 12
   1ffc6: 52 17        	slli	a4, a4, 52
   1ffc8: 5d 8f        	or	a4, a4, a5
   1ffca: 93 17 f6 03  	slli	a5, a2, 63
   1ffce: d9 8f        	or	a5, a5, a4
   1ffd0: 5d e5        	bnez	a0, 0x2007e <__trunctfdf2+0x240>
   1ffd2: 53 85 07 f2  	fmv.d.x	fa0, a5
   1ffd6: 82 80        	ret
   1ffd8: 93 75 f7 00  	andi	a1, a4, 15
   1ffdc: 91 46        	addi	a3, zero, 4
   1ffde: 11 07        	addi	a4, a4, 4
   1ffe0: 63 99 d5 00  	bne	a1, a3, 0x1fff2 <__trunctfdf2+0x1b4>
   1ffe4: 85 46        	addi	a3, zero, 1
   1ffe6: 01 47        	mv	a4, zero
   1ffe8: 0d f1        	bnez	a0, 0x1ff0a <__trunctfdf2+0xcc>
   1ffea: 0d 45        	addi	a0, zero, 3
   1ffec: 1d bf        	j	0x1ff22 <__trunctfdf2+0xe4>
   1ffee: 4d f6        	bnez	a2, 0x1ff98 <__trunctfdf2+0x15a>
   1fff0: 21 07        	addi	a4, a4, 8
   1fff2: 85 46        	addi	a3, zero, 1
   1fff4: e2 16        	slli	a3, a3, 56
   1fff6: b3 75 d7 00  	and	a1, a4, a3
   1fffa: 93 b6 15 00  	seqz	a3, a1
   1fffe: 01 47        	mv	a4, zero
   20000: 11 f1        	bnez	a0, 0x1ff04 <__trunctfdf2+0xc6>
   20002: 0d 45        	addi	a0, zero, 3
   20004: 99 dd        	beqz	a1, 0x1ff22 <__trunctfdf2+0xe4>
   20006: 8d 83        	srli	a5, a5, 3
   20008: 05 45        	addi	a0, zero, 1
   2000a: 55 bf        	j	0x1ffbe <__trunctfdf2+0x180>
   2000c: 75 f2        	bnez	a2, 0x1fff0 <__trunctfdf2+0x1b2>
   2000e: 69 b7        	j	0x1ff98 <__trunctfdf2+0x15a>
   20010: 91 07        	addi	a5, a5, 4
   20012: 05 45        	addi	a0, zero, 1
   20014: 99 c6        	beqz	a3, 0x20022 <__trunctfdf2+0x1e4>
   20016: 13 e5 26 00  	ori	a0, a3, 2
   2001a: 21 a0        	j	0x20022 <__trunctfdf2+0x1e4>
   2001c: 0d c6        	beqz	a2, 0x20046 <__trunctfdf2+0x208>
   2001e: 05 45        	addi	a0, zero, 1
   20020: 85 ea        	bnez	a3, 0x20050 <__trunctfdf2+0x212>
   20022: 93 d6 77 03  	srli	a3, a5, 55
   20026: 85 8a        	andi	a3, a3, 1
   20028: e3 8d 06 ee  	beqz	a3, 0x1ff22 <__trunctfdf2+0xe4>
   2002c: 05 07        	addi	a4, a4, 1
   2002e: 93 06 f0 7f  	addi	a3, zero, 2047
   20032: 63 01 d7 02  	beq	a4, a3, 0x20054 <__trunctfdf2+0x216>
   20036: 93 06 f0 1f  	addi	a3, zero, 511
   2003a: d2 16        	slli	a3, a3, 52
   2003c: 8d 83        	srli	a5, a5, 3
   2003e: fd 16        	addi	a3, a3, -1
   20040: f5 8f        	and	a5, a5, a3
   20042: b5 bf        	j	0x1ffbe <__trunctfdf2+0x180>
   20044: 79 d6        	beqz	a2, 0x20012 <__trunctfdf2+0x1d4>
   20046: a1 07        	addi	a5, a5, 8
   20048: e9 b7        	j	0x20012 <__trunctfdf2+0x1d4>
   2004a: 05 45        	addi	a0, zero, 1
   2004c: e9 f6        	bnez	a3, 0x20016 <__trunctfdf2+0x1d8>
   2004e: d1 bf        	j	0x20022 <__trunctfdf2+0x1e4>
   20050: 85 46        	addi	a3, zero, 1
   20052: d1 b7        	j	0x20016 <__trunctfdf2+0x1d8>
   20054: 13 65 55 00  	ori	a0, a0, 5
   20058: 63 06 08 02  	beqz	a6, 0x20084 <__trunctfdf2+0x246>
   2005c: 8d 47        	addi	a5, zero, 3
   2005e: 63 1a f8 00  	bne	a6, a5, 0x20072 <__trunctfdf2+0x234>
   20062: 0d c2        	beqz	a2, 0x20084 <__trunctfdf2+0x246>
   20064: 13 07 f0 7f  	addi	a4, zero, 2047
   20068: 52 17        	slli	a4, a4, 52
   2006a: 93 17 f6 03  	slli	a5, a2, 63
   2006e: 7d 17        	addi	a4, a4, -1
   20070: 39 a8        	j	0x2008e <__trunctfdf2+0x250>
   20072: 89 47        	addi	a5, zero, 2
   20074: e3 18 f8 fe  	bne	a6, a5, 0x20064 <__trunctfdf2+0x226>
   20078: 11 e6        	bnez	a2, 0x20084 <__trunctfdf2+0x246>
   2007a: 83 b7 01 83  	ld	a5, -2000(gp)
   2007e: 73 20 15 00  	csrs	fflags, a0
   20082: 81 bf        	j	0x1ffd2 <__trunctfdf2+0x194>
   20084: 13 07 f0 7f  	addi	a4, zero, 2047
   20088: 93 17 f6 03  	slli	a5, a2, 63
   2008c: 52 17        	slli	a4, a4, 52
   2008e: d9 8f        	or	a5, a5, a4
   20090: fd b7        	j	0x2007e <__trunctfdf2+0x240>
   20092: 13 07 f0 7f  	addi	a4, zero, 2047
   20096: 93 17 f6 03  	slli	a5, a2, 63
   2009a: 52 17        	slli	a4, a4, 52
   2009c: d9 8f        	or	a5, a5, a4
   2009e: 15 45        	addi	a0, zero, 5
   200a0: f9 bf        	j	0x2007e <__trunctfdf2+0x240>
   200a2: 93 07 f0 7f  	addi	a5, zero, 2047
   200a6: 13 17 f6 03  	slli	a4, a2, 63
   200aa: d2 17        	slli	a5, a5, 52
   200ac: d9 8f        	or	a5, a5, a4
   200ae: 15 b7        	j	0x1ffd2 <__trunctfdf2+0x194>
   200b0: 93 07 f0 7f  	addi	a5, zero, 2047
   200b4: d2 17        	slli	a5, a5, 52
   200b6: e5 b7        	j	0x2009e <__trunctfdf2+0x260>

00000000000200b8 <__clzdi2>:
   200b8: 93 57 85 03  	srli	a5, a0, 56
   200bc: 8d eb        	bnez	a5, 0x200ee <__clzdi2+0x36>
   200be: 93 07 80 03  	addi	a5, zero, 56
   200c2: e1 17        	addi	a5, a5, -8
   200c4: 91 c7        	beqz	a5, 0x200d0 <__clzdi2+0x18>
   200c6: 33 57 f5 00  	srl	a4, a0, a5
   200ca: 13 77 f7 0f  	andi	a4, a4, 255
   200ce: 75 db        	beqz	a4, 0x200c2 <__clzdi2+0xa>
   200d0: 37 37 02 00  	lui	a4, 35
   200d4: 93 06 00 04  	addi	a3, zero, 64
   200d8: 9d 8e        	sub	a3, a3, a5
   200da: 33 55 f5 00  	srl	a0, a0, a5
   200de: 93 07 07 9c  	addi	a5, a4, -1600
   200e2: aa 97        	add	a5, a5, a0
   200e4: 03 c5 07 00  	lbu	a0, 0(a5)
   200e8: 3b 85 a6 40  	subw	a0, a3, a0
   200ec: 82 80        	ret
   200ee: 93 07 80 03  	addi	a5, zero, 56
   200f2: f9 bf        	j	0x200d0 <__clzdi2+0x18>
