import struct
import os
import sys
import glob

"""
搜狗的scel词库就是保存的文本的unicode编码，每两个字节一个字符（中文汉字或者英文字母）
找出其每部分的偏移位置即可
主要两部分
1.全局拼音表，貌似是所有的拼音组合，字典序
      格式为(index,len,pinyin)的列表
      index: 两个字节的整数 代表这个拼音的索引
      len: 两个字节的整数 拼音的字节长度
      pinyin: 当前的拼音，每个字符两个字节，总长len

2.汉语词组表
      格式为(same,py_table_len,py_table,{word_len,word,ext_len,ext})的一个列表
      same: 两个字节 整数 同音词数量
      py_table_len:  两个字节 整数
      py_table: 整数列表，每个整数两个字节,每个整数代表一个拼音的索引

      word_len:两个字节 整数 代表中文词组字节数长度
      word: 中文词组,每个中文汉字两个字节，总长度word_len
      ext_len: 两个字节 整数 代表扩展信息的长度，好像都是10
      ext: 扩展信息 前两个字节是一个整数(不知道是不是词频) 后八个字节全是0

     {word_len,word,ext_len,ext} 一共重复same次 同音词 相同拼音表
 """

def read_utf16_str(f, offset=-1, len=2):
    if offset >= 0:
        f.seek(offset)
    str = f.read(len)
    return str.decode('UTF-16LE')

def read_uint16(f):
    return struct.unpack('<H', f.read(2))[0]

def get_word_from_sogou_cell_dict(fname):
    f = open(fname, 'rb')
    file_size = os.path.getsize(fname)

    hz_offset = 0
    buf = f.read(128)
    mask = struct.unpack('B', bytes([buf[4]]))[0]
    if mask == 0x44:
        hz_offset = 0x2628
    elif mask == 0x45:
        hz_offset = 0x26c4
    else:
        sys.exit(1)

    title = read_utf16_str(f, 0x130, 0x338 - 0x130)
    type = read_utf16_str(f, 0x338, 0x540 - 0x338)
    desc = read_utf16_str(f, 0x540, 0xd40 - 0x540)
    samples = read_utf16_str(f, 0xd40, 0x1540 - 0xd40)

    py_map = {}
    f.seek(0x1540 + 4)

    while True:
        py_code = read_uint16(f)
        py_len = read_uint16(f)
        py_str = read_utf16_str(f, -1, py_len)

        if py_code not in py_map:
            py_map[py_code] = py_str

        if py_str == 'zuo':
            break

    f.seek(hz_offset)
    while f.tell() != file_size:
        word_count = read_uint16(f)
        pinyin_count = int(read_uint16(f) / 2)

        py_set = []
        for i in range(pinyin_count):
            py_id = read_uint16(f)
            py_set.append(py_map[py_id])
        py_str = "'".join(py_set)

        for i in range(word_count):
            word_len = read_uint16(f)
            word_str = read_utf16_str(f, -1, word_len)
            f.read(12)
            yield py_str, word_str

    f.close()

def showtxt(records):
    for (pystr, utf8str) in records:
        print(len(utf8str), utf8str)

def store(records, f):
    for (pystr, utf8str) in records:
        print(pystr, utf8str)
        f.write("%s\t%s\n" % (pystr, utf8str))

def main():
    if len(sys.argv) != 3:
        print(
            "Unknown Option \n usage: python %s file.scel new.txt" %
            (sys.argv[0]))
        exit(1)

    # Specify the param of scel path as a directory, you can place many scel
    # file in this dirctory, the this process will combine the result in one
    # txt file
    if os.path.isdir(sys.argv[1]):
        for fileName in glob.glob(sys.argv[1] + '*.scel'):
            print(fileName)
            generator = get_word_from_sogou_cell_dict(fileName)
            with open(sys.argv[2], "a") as f:
                store(generator, f)

    else:
        generator = get_word_from_sogou_cell_dict(sys.argv[1])
        with open(sys.argv[2], "w") as f:
            store(generator, f)

if __name__ == "__main__":
    main()
