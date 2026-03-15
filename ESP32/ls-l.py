import uos

def ls_l(path="/"):
    for entry in uos.ilistdir(path):
        name, etype, inode, size = entry
        type_char = "d" if etype == 0x4000 else "-"
        print(f"{type_char}  {size:>8} bytes  {name}")

ls_l("/")