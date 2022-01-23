

if __name__ == '__main__':

    labels = ('dog', 'car', 'bird', 'fish', 'tiger', 'banana', 'cucumber', 'watermelon', 'knife', 'windscreen', 'carton', 'milk')

    print(*['{0:2d}: {1:10s} |'.format(i, labels[i]) for i in range(len(labels))], sep='\n', end='\n')


