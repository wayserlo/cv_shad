import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    mean = np.mean(matrix,axis = 1)
    matrix_centered = matrix - mean[:,None]
    # Найдем матрицу ковариации
    C = np.cov(matrix_centered, rowvar=True)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = np.linalg.eigh(C)
    # Посчитаем количество найденных собственных векторов
    num_eig_vec = eig_vec.shape[1]
    # Сортируем собственные значения в порядке убывания
    indexes = np.argsort(eig_val)[::-1]
    sorted_eig_val = eig_val[indexes]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eig_vec = eig_vec[:, indexes]
    # Оставляем только p собственных векторов
    sliced_eig_vec = sorted_eig_vec[:, :p]
    # Проекция данных на новое пространство
    proj = sliced_eig_vec.T @ matrix_centered

    return sliced_eig_vec, proj, mean


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        result_img.append(comp[0] @ comp [1] + comp[2][:, None])
        # Your code here

    return np.clip(np.stack(result_img, axis=2), 0, 255).astype('uint8')


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            compressed.append(pca_compression(img[..., j], p))
        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here
    ycbcr = np.array([0., 128., 128.])[None, None, :] + np.dot(img.astype('float64'), np.array([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [0.5, -0.4187, -0.0813]]).T)
    
    return np.clip(ycbcr, 0, 255).astype('uint8')


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    # Your code here
    rgb = np.dot((img.astype('float64')-np.array([0., 128., 128.])[None, None, :]), np.array([[1., 0., 1.402], [1, -0.34414, -0.71414], [1., 1.77, 0]]).T)
    return np.clip(rgb, 0, 255).astype('uint8')


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here

    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    #описение задания не подходит под тесты
    # Your code here
    component_blurred = gaussian_filter(component, sigma=10)

    return component_blurred[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here
    coordinates = [(x, y) for x in range(8) for y in range(8)]
    G = np.zeros(8*8).reshape(8,8).astype('float64')
    for i, j in coordinates:
        for x, y in coordinates:
            G[i,j] += block[x, y] * np.cos(((2*x+1)*i*np.pi)/16) * np.cos(((2*y+1)*j*np.pi)/16)

    G[:, 0] /= np.sqrt(2)
    G[0, :] /= np.sqrt(2)
    G /= 4
    return G



# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here

    return np.round(np.divide(block, quantization_matrix))


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if 1 <= q < 50:
        s = 5000./q
    elif 50 <= q <= 99:
        s = 200 - 2 * q
    elif q == 100:
        s = 1

    own_qm = np.floor((50 + s * default_quantization_matrix) / 100.)
    own_qm[own_qm == 0] = 1
    return own_qm


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    zigzag_list = []
    for index in range(1, block.shape[0]+1):
        slice = [i[:index] for i in block[:index]]
        diag = [slice[i][len(slice)-i-1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag_list += diag

    for index in range(1, block.shape[0]):
        slice = [i[index:] for i in block[index:]]
        diag = [slice[i][len(slice)-i-1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag_list += diag

    return zigzag_list


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    encoded_list = []
    count = 0

    for elem in zigzag_list:
        if elem == 0:
            count += 1
        elif count > 0:
            encoded_list.extend([0, count])
            count = 0
            encoded_list.append(elem)
        else:
            encoded_list.append(elem)

    if count > 0:
        encoded_list.extend([0, count])

    return encoded_list


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    # Переходим из RGB в YCbCr
    ycbcr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    result_list = []
    for k in range(3):
        compressed = []
        layer = ycbcr[..., k]
        if k:
            layer = downsampling(layer)

        for i in range(layer.shape[0] // 8):
            for j in range(layer.shape[1] // 8):
                dct_block = dct(layer[i * 8 : (i+1)*8, j * 8 : (j+1)*8] - 128)
                if k:
                    q_block = quantization(dct_block, quantization_matrixes[1])
                else:
                    q_block = quantization(dct_block, quantization_matrixes[0])
                compressed.append(compression(zigzag(q_block)))
        result_list.apprend(compressed)
    
    return result_list


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    decompressed = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i]:
            decompressed.append(compressed_list[i])
        else:
            decompressed.extend(np.zeros(compressed_list[i+1]).astype('uint8'))
            i += 1
        i += 1

    return decompressed


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    block = np.zeros(8*8).reshape(8,8)
    for index in range(1, block.shape[0]+1):
        diag = input[:index]
        input = input[index:]
        if len(diag) % 2:
            diag.reverse()
        for j in range(index):
            block[j][index-1-j] = diag[j]

    for index in range(1, block.shape[0]):
        diag = input[-index:]
        input = input[:-index]
        if len(diag) % 2:
            diag.reverse()
        for j in range(index):
            block[7 - (index-1-j)][7- j] = diag[j]

    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here
    coordinates = [(x, y) for x in range(8) for y in range(8)]
    f = np.zeros(8*8).reshape(8,8).astype('float64')
    for x, y in coordinates:
        for u, v in coordinates:
            if u == 0: alpha_u = 1 / np.sqrt(2)
            else: alpha_u = 1
            if v == 0: alpha_v = 1 / np.sqrt(2)
            else: alpha_v = 1
            f[x,y] += alpha_u * alpha_v * block[u, v] * np.cos(((2*x+1)*u*np.pi)/16) * np.cos(((2*y+1)*v*np.pi)/16)
    return np.round(f / 4)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here

    return np.repeat(np.repeat(component, 2, axis=1), 2, axis=0)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here
    ycbcr = []
    for k, layer in enumerate(result):
        component = []
        I = result_shape[0] // (8 * (2 if k else 1))
        J = result_shape[1] // (8 * (2 if k else 1))
        for i in range(I):
            row = []
            for j in range(J):
                beforeqm = inverse_zigzag(inverse_compression(layer[i * J + j]))
                if k:
                    block = upsampling(inverse_dct(inverse_quantization(beforeqm, quantization_matrixes[1])))
                else:
                    block = inverse_dct(inverse_quantization(beforeqm, quantization_matrixes[0]))
                row.append(block)
            component.append(np.concatenate(row, axis=1))
        component = np.concatenate(component, axis=0) + 128
        ycbcr.append(component)
    img = ycbcr2rgb(np.stack(ycbcr, axis=2))

    return img

    return ...


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        jpeg, _ = compression_pipeline(img.copy(), 'jpeg', p)

        axes[i // 3, i % 3].imshow(jpeg)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    compressed = np.array(compressed, dtype=np.object_)
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
