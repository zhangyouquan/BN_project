###########
	Ubuntu 18.04.1上安装配置REALSENSOR D435i， 编程获取图像的颜色、深度、IMU信息，并调用YOLOv4实现物体识别。
	node：（未完成GPU加速且部分功能待调试。不同环境会出现不同问题，其中BUG请自行寻找途径解决。）--by zyq 2021.5.28      
															############

1、安装SDK 
从官网上安装sdk2以及相关依赖可参考文档：

https://realsense.intel.com/sdk-2/#install   

https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

SDK2.0(2.17.0)版本下载链接：

https://github.com/IntelRealSense/librealsense/releases/tag/v2.17.0

2、CMake、librealsense、pyrealsense2的配置安装
下载并安装CMake

1）下载cmake-3.8.2.tar.gz : https://cmake.org/files/

or https://blog.csdn.net/qq_42393859/article/details/85251356

2) 在主文件夹下新建tools/文件夹，将cmake-3.8.2.tar.gz解压之后放在tools/中，为了防止出现权限不足问题，直接对文件更改权限：
解压：sudo tar -zxvf cmake-3.8.2.tar.gz
赋权限：sudo chmod -R 777 cmake-3.8.2

3)安装gcc-c++:
sudo apt-get install build-essential
或者直接执行这两条命令

sudo apt-get install gcc
sudo apt-get install g++

4)进入文件夹下执行以下命令：

 sudo ./bootstrap
 sudo make
 sudo make install

5)查看是否安装成功以及安装版本：

cmake --version

下载librealsense驱动  https://github.com/IntelRealSense/librealsense/

注意： USB接口必须为3.0

1)先确定内核版本：

    uname -r

如果>=4.4.0-50的版本就可以继续向下进行了，否则需要升级Ubuntu内核。

2）安装一些依赖：

sudo apt-get install libusb-1.0-0-dev pkg-config libgtk-3-dev

3）安装glfw3库：

sudo apt-get install libglfw3-dev

4）下载驱动安装包：

git clone https://github.com/IntelRealSense/librealsense

5）进入该文件夹

cd librealsense/

6)在 librealsense 文件夹下执行

 sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade

7）进入下载的 librealsense 路径下，执行如下：

$ mkdir build

$ cd build

$ cmake ../

$ cmake ../ -DBUILD_EXAMPLES=true

$ make && sudo make install

8)在 librealsense 文件夹下安装Video4Linux视频内核驱动，注意不要插上RealSense R200摄像头。
在librealsense的路径下执行：

sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && udevadm trigger

安装Openssl库：

sudo apt-get install libssl-dev

编译配置文件：

./scripts/patch-realsense-ubuntu-xenial.sh

提示完成后，插上RealSense，再执行：

sudo dmesg | tail -n 50

此时可进入/usr/local/lib中查看或者librealsense下的example文件夹下，执行：

./cpp-capture.cpp

然后出现画面，代表安装成功！

pyrealsense2安装：

https://blog.csdn.net/qq_42393859/article/details/85044330

3、安装Opencv 4.4 吧，之前调试了好几个例程，发现如果用YOLOv4， opencv应对应opencv 4.4。不然会报错不支持 mish 激活函数

4、后期写界面或者用他SDK里带的API，都得依赖opengl。
安装opengl：

sudo apt-get install build-essential

sudo apt-get install libgl1-mesa-dev

sudo apt-get install libglu1-mesa-dev

sudo apt-get install libglut-dev

sudo apt-get install freeglut3-dev

5、调用YOLO完成视频流的实时检测需要用到GPU加速：还得配置GPU加速环境：英伟达显卡驱动+cuda+cuDnn（版本要对应）
显卡驱动安装参考：https://zhuanlan.zhihu.com/p/59618999
cuda+cuDnn配置参考：https://blog.csdn.net/choimroc/article/details/104735680?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-5&spm=1001.2101.3001.4242

--by zyq 2021.5.28 
######——————————########




