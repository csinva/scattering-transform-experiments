# sizes
- J=1
	ims.shape before NCHW (2, 3, 32, 32)
	M 32 N 32 J 1
	scat.shape after NCHW (?, 27, 16, 16)
- J=2
	scat.shape after NCHW (?, 243, 8, 8)
- J=3
	scat.shape after NCHW (?, 651, 4, 4)
- J=4
	scat.shape after NCHW (?, 1251, 2, 2)
- sizes:
	- Sx(u) has a size equal to 3 ×  (1 + JL + 1/2 J(J − 1)L^2)
	- 1 is for the low-freq phi