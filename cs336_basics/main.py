# from tokenizer import pretokenize
# from sortedcontainers import SortedDict

# string = "okfこん"

# byte = string.encode("utf-8")
# z = bytes([1]) + bytes([2])
# print(type(z), z)
# print(bytes([])

a = (4, "0".encode("utf-8"), "0".encode("utf-8"))
b = (4, "u".encode("utf-8"), "n".encode("utf-8"))
print(max(a,b))
a = (4, "0", "0")
b = (4, "u", "n")
print(max(a,b))