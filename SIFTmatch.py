import cv2

def sift_matching(img1, img2):
    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测特征点和计算特征描述符
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()

    # 使用knnMatch进行特征点匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 进行 Lowe's ratio test 进行筛选
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 可视化匹配结果
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None)

    return img_matches

# 读取图像
img1 = cv2.imread('texture3.jpg')
img2 = cv2.imread('texture4.jpg')

# 进行图像匹配
result = sift_matching(img1, img2)

# 显示匹配结果
cv2.namedWindow('SIFT Matching Result',cv2.WINDOW_NORMAL)
cv2.imshow('SIFT Matching Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


