import math
import time

import gin
import numpy as np
from gym import spaces

from gyms.hhh.label import Label


@gin.configurable
class ImageGenerator:
    def __init__(self,
                 img_width_px=512,  # the image width (pixels)
                 address_space=16,  # size of the address space=(img height-1), hierarchy levels are [ADDRESS_SPACE,32]
                 # if the
                 # maximum HHH pixel value is at least SQUASH_THRESHOLD times larger than the 2nd smallest value,
                 # then values will be squashed together with log (smallest value will (likely) be 0, so take 2nd
                 # smallest)
                 hhh_squash_threshold=1,
                 crop_standalone_hhh_image=True,
                 normalize=True,
                 mode='single'  # single=sum, multi=(sum,mean,std,min,max)
                 ):
        is_power_of_two = (img_width_px & (img_width_px - 1) == 0) and img_width_px != 0
        assert is_power_of_two
        self.img_width_px = img_width_px
        self.address_space = address_space
        self.hhh_squash_threshold = hhh_squash_threshold
        self.crop_standalone_hhh_image = crop_standalone_hhh_image
        self.normalize = normalize
        assert mode in ['single', 'multi']
        self.mode = mode

    def get_hhh_img_spec(self):
        # create dummy input for image gen, feed it through img gen and return img spec
        class DummyHHHAlg:
            def __init__(self, addr_space):
                self.addr_space = addr_space

            def query_all(self):
                return np.array([[x, 0, x] for x in range(self.addr_space, 33)])

        if self.mode == 'single':
            shape = self.generate_hhh_image(DummyHHHAlg(self.address_space), crop=self.crop_standalone_hhh_image,
                                            normalize=self.normalize).shape
        else:
            shape = self.generate_multi_channel_hhh_image(DummyHHHAlg(self.address_space),
                                                          normalize=self.normalize).shape
        return self._shape_to_gym_spec(shape)

    def _shape_to_gym_spec(self, shape):
        return spaces.Box(-3.0, 3.0, shape=shape, dtype=np.float32)

    def generate_image(self, hhh_algo, hhh_query_result):
        if hhh_query_result is not None:  # return two-channel image (HHH and Filter)
            raise NotImplementedError('not supported anymore')
        else:  # return one-channel image (HHH only)
            if self.mode == 'single':
                return self.generate_hhh_image(hhh_algo, crop=self.crop_standalone_hhh_image, normalize=self.normalize)
            else:
                return self.generate_multi_channel_hhh_image(hhh_algo, normalize=self.normalize)

    def generate_hhh_image(self, hhh_algo, crop, normalize):
        # start = time.time()

        max_addr = 2 ** self.address_space - 1
        # the bounds of the bins to separate the address space (x-axis)
        bounds = self._get_x_axis_bounds(max_addr)

        # query HHH for all undiscounted counters, filter hierarchy to be in address space
        res = hhh_algo.query_all()
        res = np.asarray(res)

        # init output image (x-axis=address bins, y-axis=hierarchy levels)
        image = np.zeros((self.address_space + 1, self.img_width_px), dtype=np.float32)

        if res.size == 0:  # if no counters (last step in env when trace ended), return all zero image
            image = np.expand_dims(image, 2)  # output should be (height, width, channels=1)
            return image

        res = res[res[:, 0] >= self.address_space, :]

        if np.max(res[:, 1]) > max_addr:
            raise ValueError(
                f'HHH algo contains addresses larger than currently configured MAX_ADDR ({np.max(res[:, 1])}' > max_addr
            )

        # unique hierarchy levels (should be [ADDRESS_SPACE..32])
        unique_indices = np.unique(res[:, 0], return_index=True)[1]

        # 33-ADDRESS_SPACE lists in ascending order of hierarchy levels (start with ADDRESS_SPACE, end with 32)
        # each list has shape num_items(level) X 2; columns are IP, count
        split_by_level = np.split(res[:, 1:], np.sort(unique_indices)[1:])[::-1]
        assert len(split_by_level) == self.address_space + 1

        # iterate over all hierarchy levels and the corresponding item lists
        for level, l in enumerate(split_by_level):
            # bin the item lists according to the IP address
            l_binned = np.digitize(l[:, 0], bins=bounds) - 1

            shifted_level = level + self.address_space  # in [ADDRESS_SPACE, 32]
            if shifted_level != 32:  # if not at the bottom, consider possible adjacent bins in subnet
                subnet_size = Label.subnet_size(level + self.address_space)
                l_end = l[:, 0] + subnet_size - 1
                l_end_binned = np.digitize(l_end, bins=bounds) - 1
            else:  # if at the bottom, only consider the bin which contains the current IP
                l_end_binned = l_binned

            # increment the image pixels for each level and bin according the item list's count
            for bin_index, count in enumerate(l[:, 1]):
                first_bin = l_binned[bin_index]
                last_bin = l_end_binned[bin_index]
                if first_bin == last_bin:  # only increment current bin
                    image[level, first_bin] += count
                else:  # increment all bins covered by the current item
                    for bin_to_increment in range(first_bin, last_bin + 1):
                        image[level, bin_to_increment] += count

        if self.hhh_squash_threshold != -1:
            # if needed squash distant values together to increase visibility of smaller IP sources
            second_smallest_value = np.partition(np.unique(image.flatten()), 1)[1]
            # print(f'ratio={np.max(image) / second_smallest_value}')
            if np.max(image) / second_smallest_value >= self.hhh_squash_threshold:
                image = np.log(image, where=image > 1, out=np.zeros_like(image))

        if crop:
            max_level = int(math.log2(self.img_width_px)) + 1
            image = image[:max_level, :]

        if normalize:
            # normalize values to have zero mean, unit variance
            image = (image - np.mean(image)) / (np.std(image) + 1e-8)

        # output (height, width, channels=1)
        image = np.expand_dims(image, 2)
        # print(f'[hhh] query_all and CNN image (shape {image.shape}) build time={time.time() - start}')
        return image

    def generate_multi_channel_hhh_image(self, hhh_algo, normalize):
        start = time.time()

        max_addr = 2 ** self.address_space - 1
        # the bounds of the bins to separate the address space (x-axis)
        bounds = self._get_x_axis_bounds(max_addr)

        # query HHH for all undiscounted counters, filter hierarchy to be in address space
        res = hhh_algo.query_all()
        res = np.asarray(res)

        channels = 5  # counter sum, mean, std, min, max

        # init output image (x-axis=address bins, y-axis=hierarchy levels, 3 channels (sum, mean, std)
        image = np.zeros((self.address_space + 1, self.img_width_px, channels), dtype=np.float32)

        counters = np.empty((self.address_space + 1, self.img_width_px, 1), dtype=np.object)
        for i in np.ndindex(counters.shape):
            counters[i] = []

        if res.size == 0:  # if no counters (last step in env when trace ended), return all zero image
            return image

        res = res[res[:, 0] >= self.address_space, :]

        if np.max(res[:, 1]) > max_addr:
            raise ValueError(
                f'HHH algo contains addresses larger than currently configured MAX_ADDR ({np.max(res[:, 1])}' > max_addr
            )

        # unique hierarchy levels (should be [ADDRESS_SPACE..32])
        unique_indices = np.unique(res[:, 0], return_index=True)[1]

        # 33-ADDRESS_SPACE lists in ascending order of hierarchy levels (start with ADDRESS_SPACE, end with 32)
        # each list has shape num_items(level) X 2; columns are IP, count
        split_by_level = np.split(res[:, 1:], np.sort(unique_indices)[1:])[::-1]
        assert len(split_by_level) == self.address_space + 1

        # iterate over all hierarchy levels and the corresponding item lists
        for level, l in enumerate(split_by_level):
            # bin the item lists according to the IP address
            l_binned = np.digitize(l[:, 0], bins=bounds) - 1

            shifted_level = level + self.address_space  # in [ADDRESS_SPACE, 32]
            if shifted_level != 32:  # if not at the bottom, consider possible adjacent bins in subnet
                subnet_size = Label.subnet_size(level + self.address_space)
                l_end = l[:, 0] + subnet_size - 1
                l_end_binned = np.digitize(l_end, bins=bounds) - 1
            else:  # if at the bottom, only consider the bin which contains the current IP
                l_end_binned = l_binned

            # increment the image pixels for each level and bin according the item list's count
            for bin_index, count in enumerate(l[:, 1]):
                first_bin = l_binned[bin_index]
                last_bin = l_end_binned[bin_index]
                if first_bin == last_bin:  # only adapt current bin
                    counters[level, first_bin, 0].append(count)
                else:  # adapt all bins covered by the current item
                    for bin_to_increment in range(first_bin, last_bin + 1):
                        counters[level, bin_to_increment, 0].append(count)

        def mean(arr):
            if len(arr) == 0:
                return 0
            else:
                return np.mean(arr)

        def std(arr):
            if len(arr) == 0:
                return 0
            else:
                return np.std(arr)

        def my_min(arr):
            if len(arr) == 0:
                return 0
            else:
                return np.min(arr)

        def my_max(arr):
            if len(arr) == 0:
                return 0
            else:
                return np.max(arr)

        image[:, :, 0] = np.vectorize(sum)(counters[:, :, 0])
        image[:, :, 1] = np.vectorize(mean)(counters[:, :, 0])
        image[:, :, 2] = np.vectorize(std)(counters[:, :, 0])
        image[:, :, 3] = np.vectorize(my_min)(counters[:, :, 0])
        image[:, :, 4] = np.vectorize(my_max)(counters[:, :, 0])

        if self.hhh_squash_threshold != -1:
            # if needed squash distant values together to increase visibility of smaller IP sources
            # not for channel 2 (std)
            for c in [0, 1, 3, 4]:
                second_smallest_value = np.partition(np.unique(image[:, :, c].flatten()), 1)[1]
                # print(f'ratio={np.max(image) / second_smallest_value}')
                if np.max(image[:, :, c]) / second_smallest_value >= self.hhh_squash_threshold:
                    image[:, :, c] = np.log(image[:, :, c], where=image[:, :, c] > 1, out=np.zeros_like(image[:, :, c]))

        if normalize:
            # for channels that visualize counter values, use same std and mean
            # such that normalized values are comparable
            mean = np.mean(image[:, :, [0, 1, 3, 4]])
            std = np.std(image[:, :, [0, 1, 3, 4]])
            for c in [0, 1, 3, 4]:
                image[:, :, c] = (image[:, :, c] - mean) / (std + 1e-8)

            # normalize std with itself (no absolute counters)
            image[:, :, 2] = (image[:, :, 2] - np.mean(image[:, :, 2])) / (np.std(image[:, :, 2]) + 1e-8)

        return image

    def _get_x_axis_bounds(self, max_addr):
        step_size = int((max_addr + 1) / self.img_width_px)
        bounds = np.arange(0, max_addr + 1 + step_size, step=step_size, dtype=np.int)
        assert bounds[-1] == max_addr + 1
        assert len(bounds) == self.img_width_px + 1
        return bounds
