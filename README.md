# image-fusion
 图像融合

步骤：

1.读入两幅图像，苹果和橘子（要保证输入图片的大小size一样，可用resize(）函数）

2.构建苹果和橘子的高斯金字塔（6层）

3.根据高斯金字塔计算拉普拉斯金字塔

4.在拉普拉斯的每一层进行图像融合（苹果的左边与右边融合）

5.根据融合后的图像金字塔重建原始图像



