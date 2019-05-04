import tensorflow as tf
import numpy.distutils.system_info as sysinfo


def list_devices():
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


def numpy_bitness():
    print(sysinfo.platform_bits)


if __name__ == "__main__":
    list_devices()
    numpy_bitness()
