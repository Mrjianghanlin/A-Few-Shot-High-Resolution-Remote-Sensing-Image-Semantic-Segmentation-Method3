# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 13:34:59 2021

@author: 洋洋
"""

# -*- coding:utf8 -*-

import os


class BatchRename():
    '''
    批量重命名文件夹中的图片文件为Rural_数字.png格式
    '''

    def __init__(self):
        self.path = r'D:\BaiduNetdiskDownload\2021LoveDA\Test\Urban\images_png'  # 存放图片的文件夹路径

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1

        for item in filelist:
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), 'Urban_test_' + str(i) + '.png')

                try:
                    os.rename(src, dst)
                    print("converting %s to %s ..." % (src, dst))
                    i = i + 1
                except:
                    continue

        print("total %d to rename & converted %d pngs" % (total_num, i - 1))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()
