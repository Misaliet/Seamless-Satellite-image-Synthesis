import random

default_x = 256
default_y = 256
default_z = 1
default_z1_row = 4
default_z1_column = 4
target_img_size = 512
single_img_size = 256
zoom_level = 4
half = 2
buffer_multiple = 2

# current size
# TODO: solve fixed max size
max_size = 1024

# All coordinate values correspond to the coordinates of z1
class navigation:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = default_z
        self.z1_row = default_z1_row
        self.z1_column = default_z1_column
        # TODO: find a suitable strike for move operation
        self.strike = 16

        self.buffer_x = single_img_size
        self.buffer_y = single_img_size

        self.update_real_coordinate()

        self.replace_level = -1
        self.replace_index = -1
        self.replace_x = -1
        self.replace_y = -1

    def setup(self):
        self.x = 0
        self.y = 0
        self.z = default_z
        self.z1_row = default_z1_row
        self.z1_column = default_z1_column
        self.buffer_x = single_img_size
        self.buffer_y = single_img_size

        self.update_real_coordinate()

    def get_imgs_index(self):
        # TODO: Determine the special circumstances of boundaries
        row = int(self.real_x / (single_img_size / pow(zoom_level, self.z-1)))
        column = int(self.real_y / (single_img_size / pow(zoom_level, self.z-1)))
        index_base = row * self.z1_row * pow(zoom_level, self.z-1) + column
        index_0 = index_base - self.z1_row * pow(zoom_level, self.z-1) - 1
        index_1 = index_0 + 1
        index_2 = index_0 + 2
        index_3 = index_0 + 3
        index_4 = index_base - 1
        index_5 = index_base
        index_6 = index_base + 1
        index_7 = index_base + 2
        index_8 = index_base + self.z1_row * pow(zoom_level, self.z-1) - 1
        index_9 = index_8 + 1
        index_10 = index_8 + 2
        index_11 = index_8 + 3
        index_12 = index_8 + self.z1_row * pow(zoom_level, self.z-1)
        index_13 = index_12 + 1
        index_14 = index_12 + 2
        index_15 = index_12 + 3
        self.x = int(index_0/(self.z1_row*pow(zoom_level, self.z-1))) * (single_img_size / pow(zoom_level, self.z-1))
        self.y = int(index_0%(self.z1_row*pow(zoom_level, self.z-1))) * (single_img_size / pow(zoom_level, self.z-1))

        img_A_paths = [index_0, index_1, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, index_10, index_11, index_12, index_13, index_14, index_15]
        # print(img_A_paths)
        img_B_paths = img_A_paths

        return img_A_paths, img_B_paths

    def get_coordinate(self):
        return self.buffer_x, self.buffer_y

    def get_real_coordinate(self):
        return self.real_x, self.real_y

    def get_frame_coordinate(self):
        return self.x, self.y

    def get_level(self):
        return self.z

    def get_form(self):
        return self.z1_row, self.z1_column

    def zoom_in(self):
        self.real_x = self.real_x + int(target_img_size/half/pow(zoom_level, self.z-1)) - int(single_img_size/zoom_level/pow(zoom_level, self.z-1))
        self.real_y = self.real_y + int(target_img_size/half/pow(zoom_level, self.z-1)) - int(single_img_size/zoom_level/pow(zoom_level, self.z-1))
        self.z += 1

    def zoom_out(self):
        self.z -= 1
        if self.z == 2:
            self.real_x = self.real_x - int(target_img_size/half/pow(zoom_level, self.z-1)) + int(single_img_size/zoom_level/pow(zoom_level, self.z-1))
            self.real_y = self.real_y - int(target_img_size/half/pow(zoom_level, self.z-1)) + int(single_img_size/zoom_level/pow(zoom_level, self.z-1))
        elif self.z == 1:
            # FIXME: z2 zoom out
            self.real_x = default_x
            self.real_y = default_y
        
    
    def left(self):
        if self.real_y - self.strike/pow(zoom_level, self.z-1) <= single_img_size/2:
            return
        self.buffer_y -= self.strike
        self.update_real_coordinate()

    def up(self):
        if self.real_x - self.strike/pow(zoom_level, self.z-1) <= single_img_size/2:
            return
        self.buffer_x -= self.strike
        self.update_real_coordinate()

    def down(self):
        print(self.real_x + self.strike/pow(zoom_level, self.z-1))
        if self.real_x + self.strike/pow(zoom_level, self.z-1) >= max_size - single_img_size/2 - target_img_size/pow(zoom_level, self.z-1):
            return
        self.buffer_x += self.strike
        self.update_real_coordinate()

    def right(self):
        if self.real_y + self.strike/pow(zoom_level, self.z-1) >= max_size - single_img_size/2 - target_img_size/pow(zoom_level, self.z-1):
            return
        self.buffer_y += self.strike
        self.update_real_coordinate()

    def update_real_coordinate(self):
        self.real_x = self.x + self.buffer_x/pow(zoom_level, self.z-1)
        self.real_y = self.y + self.buffer_y/pow(zoom_level, self.z-1)
    
    def update_frame_coordinate(self):
        row = int(self.real_x / (single_img_size / pow(zoom_level, self.z-1)))
        column = int(self.real_y / (single_img_size / pow(zoom_level, self.z-1)))
        index_base = row * self.z1_row * pow(zoom_level, self.z-1) + column
        index_0 = index_base - self.z1_row * pow(zoom_level, self.z-1) - 1
        self.x = int(index_0/(self.z1_row*pow(zoom_level, self.z-1))) * (single_img_size / pow(zoom_level, self.z-1))
        self.y = int(index_0%(self.z1_row*pow(zoom_level, self.z-1))) * (single_img_size / pow(zoom_level, self.z-1))
    
    def update_buffer_coordinate(self):
        self.buffer_x = (self.real_x - self.x) * pow(zoom_level, self.z-1)
        self.buffer_y = (self.real_y - self.y) * pow(zoom_level, self.z-1)

    def refresh_buffer(self):
        # if (self.real_x - self.x) <= (single_img_size / pow(zoom_level, self.z-1)) or (self.real_y - self.y) <= (single_img_size / pow(zoom_level, self.z-1)) or (self.x - self.real_x + 512) <= (single_img_size / pow(zoom_level, self.z-1)) or  (self.y - self.real_y + 512) <= (single_img_size / pow(zoom_level, self.z-1)):
        # if self.buffer_x <= self.strike or self.buffer_y <= self.strike or (target_img_size - self.buffer_x) <= self.strike or (target_img_size - self.buffer_y) <= self.strike:
        # if self.buffer_x <= 0 or self.buffer_y <= 0 or (target_img_size - self.buffer_x) <= 0 or (target_img_size - self.buffer_y) <= 0:
        if self.buffer_x <= single_img_size/2 or self.buffer_y <= single_img_size/2 or (target_img_size - self.buffer_x) <= single_img_size/2 or (target_img_size - self.buffer_y) <= single_img_size/2:
            return True
        else:
            return False

    def random(self):
        self.replace_level = self.z
        total_index = self.z1_row * self.z1_column * pow(zoom_level, self.replace_level-1) * pow(zoom_level, self.replace_level-1)
        random_index = random.randint(0, total_index-1)
        self.replace_index = random_index
        self.replace_x = self.real_x + int(single_img_size/zoom_level/pow(zoom_level, self.z-1))
        self.replace_y = self.real_y + int(single_img_size/zoom_level/pow(zoom_level, self.z-1))

    def get_random(self):
        return self.replace_level, self.replace_index, self.replace_x, self.replace_y