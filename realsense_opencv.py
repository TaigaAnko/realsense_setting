import numpy as np

import cv2
import pyrealsense2 as rs

class RealSense:
    def __init__(self, height = 480, width = 640) -> None:
         # RealSenseカメラの初期化
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)

        self.pipeline = rs.pipeline()
        self.pipeline.start(config)

        # Alignオブジェクト生成
        align_to = rs.stream.color
        self.align = rs.align(align_to)
    
    def main(self):
        try:
            while True:
                # 画角調整
                aligned_frames = self.align.process(self.frames)
                # 深度フレームを取得
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if not depth_frame or not color_frame:
                    continue

                #imageをnumpy arrayに
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                #depth imageをカラーマップに変換
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)

                #画像表示
                color_image_s = cv2.resize(color_image, (640, 360))
                depth_colormap_s = cv2.resize(depth_colormap, (640, 360))
                images = np.hstack((color_image_s, depth_colormap_s))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

                if cv2.waitKey(1) & 0xff == 27: #ESCで終了
                    cv2.destroyAllWindows()
                    break

        finally:
            #ストリーミング停止
            self.pipeline.stop()




if __name__ == '__main__':
    RealSense().main()