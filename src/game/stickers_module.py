import glob
import time
import cv2
import numpy as np


class Sticker:
    def __init__(self, img):  # TODO: add more inputs
        self.img = img[:, :, :3]
        self.mask = img[:, :, -1]
        self.start_time = time.time()
        self.fade_in = 0.2
        self.full = 0.2
        self.fade_out = 1.0
        self.total_time = self.fade_in + self.full + self.fade_out
        self.loc_w = 900
        self.loc_h = 100
        self.angle = np.random.randint(20) - 10  # TODO: use this
        self.move_w = np.random.randint(60) - 30
        self.move_h = np.random.randint(60) - 30
        self.done = False

    def add_to_image(self, img):
        time_from_start = time.time() - self.start_time
        ratio = time_from_start / self.total_time
        if time_from_start < self.fade_in:
            beta = time_from_start / self.fade_in
            alpha = 1 - beta
        elif time_from_start < self.fade_in + self.full:
            beta = 1
            alpha = 1 - beta
        elif time_from_start < self.fade_in + self.full + self.fade_out:
            alpha = (time_from_start - (self.fade_in + self.full)) / self.fade_out
            beta = 1 - alpha
        else:
            self.done = True

        if not self.done:
            move_h = int(ratio * self.move_h)
            move_w = int(ratio * self.move_w)
            location = [self.loc_h + move_h, self.loc_h + self.img.shape[0] + move_h,
                        self.loc_w + move_w, self.loc_w + self.img.shape[1] + move_w]
            blend = img[location[0]:location[1], location[2]:location[3]]
            mask = np.where(self.mask)
            blend[mask] = cv2.addWeighted(blend[mask], alpha, self.img[mask], beta, 0.0).squeeze()
            # img[location[0]:location[1], location[2]:location[3]] =\
            #     cv2.addWeighted(img[location[0]:location[1], location[2]:location[3]],
            #                     alpha, self.img, beta, 0.0)
        return img

class StickersModule:
    def __init__(self, path):
        stickers_path = glob.glob(path + '*.png')
        self.stickers = []
        for sticker_path in stickers_path:
            img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
            self.stickers.append(cv2.resize(img, (200, int(img.shape[1]/img.shape[0]*200))))
        self.sticker_sequence = []
        self.next_sticker_score = 3

    def update_stickers(self, img, scores):
        if scores.final_score > self.next_sticker_score:
            self.add_sticker()
            self.next_sticker_score += 1
        self.draw_sequence(img)
        return img


    def add_sticker(self):
        img = self.stickers[np.random.randint(len(self.stickers))]
        self.sticker_sequence.append(Sticker(img=img))

    def draw_sequence(self, img):
        for sticker in self.sticker_sequence:
            if not sticker.done:
                img = sticker.add_to_image(img)
            if sticker.done:
                self.sticker_sequence.remove(sticker)
        return img



