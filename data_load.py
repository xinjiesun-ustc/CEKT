def read_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    for i in range(0, len(lines), 4):
        student_data = tuple(lines[i:i + 4])
        data.append(student_data)

    return data


if __name__ == '__main__':
    data=read_file("./data/Afterprocess/train.txt")
    pass