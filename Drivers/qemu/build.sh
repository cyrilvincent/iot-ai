env -i HOME=$HOME TERM=$TERM PATH=/usr/bin:/bin bash
git clone https://github.com/buildroot/buildroot && cd buildroot && make freescale_imx8mpevk_defconfig && make -j$(nproc) && cd output/images && zip imx_qemu_ready.zip Image imx8mp-evk.dtb sdcard.img
