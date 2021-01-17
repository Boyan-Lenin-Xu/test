import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='paper image')
parser.add_argument('--x_crop', type=int, default=100,
                    help='the x of corp image')
parser.add_argument('--y_crop', type=int, default=100,
                    help='the y of corp image')
parser.add_argument('--image_num', type=str, default=74,
                    help='which image to SR')
args = parser.parse_args()


def crop_VDSR(x_crop=100, y_crop=100):
    pass


def crop_typical(x_crop=100, y_crop=100, mode='EDSR', crop_w=0, crop_h=0):
    if mode in ['EDSR', 'GCSR', 'RCAN']:
        GT = cv2.imread('./' + mode + '/img' + args.image_num + '_x4_SR.png')
        crop_a = GT[y_crop:(y_crop + crop_h), x_crop:(x_crop + crop_w)]
        crop_a_resized = cv2.resize(crop_a, (crop_w*4, crop_h*4))
        return crop_a_resized
    elif mode == 'HR':
        GT = cv2.imread('./HR/img' + args.image_num + '.png')
        crop_a = GT[y_crop:(y_crop + crop_h), x_crop:(x_crop + crop_w)]
        crop_a_resized = cv2.resize(crop_a, (crop_w*4, crop_h*4))
        return crop_a_resized
    elif mode == 'DBPN':
        GT = cv2.imread('./DBPN/img' + args.image_num + 'x4.png')
        crop_a = GT[y_crop:(y_crop + crop_h), x_crop:(x_crop + crop_w)]
        crop_a_resized = cv2.resize(crop_a, (crop_w*4, crop_h*4))
        return crop_a_resized
    elif mode == 'LR':
        LR = cv2.imread('./LR/img' + args.image_num + 'x4.png')
        image_array = np.array(LR, dtype=np.int32)
        size = image_array.shape
        LR = cv2.resize(LR, (size[1]*4, size[0]*4))
        crop_a = LR[y_crop:(y_crop + crop_h), x_crop:(x_crop + crop_w)]
        crop_a_resized = cv2.resize(crop_a, (crop_w*4, crop_h*4))
        return crop_a_resized


def main():
    GT = cv2.imread('./HR/img' + args.image_num + '.png')
    image_array = np.array(GT, dtype=np.int32)
    size = image_array.shape
    height = size[0]    # height: the height of whole image
    width = size[1]     # width: the width of whole image
    ch = int(height / 10 * 4)    # ch: the height of image slide in comparison
    cw = ch * 2             # cw: the width of image slide in comparison
    fw = width + (20 * 3) + (cw * 3)  # hw: the width of image comparison in paper
    crop_w = int(cw / 4)       # crop_w:
    crop_h = int(ch / 4)
    ch = crop_h * 4   # fine tune
    cw = crop_w * 4
    final_matrix = np.zeros((height, fw, 3), np.uint8)
    final_matrix = cv2.bitwise_not(final_matrix)
    GT = cv2.rectangle(GT, (args.x_crop, args.y_crop), ((args.x_crop + crop_w),
                                                        (args.y_crop + crop_h)), (0, 0, 255), 2)
    final_matrix[0:height, 0:width] = GT
    HR = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='HR', crop_w=crop_w, crop_h=crop_h)
    final_matrix[0:ch, (width + 20):(width + 20 + cw)] = HR
    GCSR = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='GCSR', crop_w=crop_w, crop_h=crop_h)
    final_matrix[int(height / 2):int(height / 2 + ch), (width + 20*2 + cw):(width + 20*2 + cw*2)] = GCSR
    EDSR = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='EDSR', crop_w=crop_w, crop_h=crop_h)
    final_matrix[int(height / 2):int(height / 2 + ch), (width + 20):(width + 20 + cw)] = EDSR
    RCAN = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='RCAN', crop_w=crop_w, crop_h=crop_h)
    final_matrix[int(height / 2):int(height / 2 + ch), (width + 20*3 + cw*2):(width + 20*3 + cw*3)] = RCAN
    DBPN = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='DBPN', crop_w=crop_w, crop_h=crop_h)
    final_matrix[0:ch, (width + 20*3 + cw*2):(width + 20*3 + cw*3)] = DBPN
    LR = crop_typical(x_crop=args.x_crop, y_crop=args.y_crop, mode='LR', crop_w=crop_w, crop_h=crop_h)
    final_matrix[0:ch, (width + 20*2 + cw):(width + 20*2 + cw*2)] = LR
    cv2.imshow('cop', final_matrix)
    cv2.waitKey(10000)
    cv2.imwrite(args.image_num + 'crop.png', final_matrix)


if __name__ == '__main__':
    main()
