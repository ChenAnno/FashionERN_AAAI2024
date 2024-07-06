import os
import cv2
from tqdm import tqdm
import base64


def vis_ann(reftarcap_dict, html_file):
    html_file_fp = open(html_file, "w")
    html_file_fp.write("<html>\n<body>\n")
    html_file_fp.write('<meta charset="utf-8">\n')

    html_file_fp.write("<p>\n")
    html_file_fp.write('<table border="0" align="center">\n')
    img_root = "/home/chenyanzhe/AAAI2023/fashion-iq/images/"
    for i, (reftarcap, sort_list) in tqdm(enumerate(reftarcap_dict.items())):
        if i > 200:
            break
        html_file_fp.write("<tr>\n")
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')

        reftar, cap = reftarcap.split("*")
        ref, tar = reftar.split("&")
        img_path = os.path.join(img_root, ref + ".png")
        img = cv2.imread(img_path)
        imgdata = cv2.imencode(".jpg", img)[1].tobytes()
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                <br> ref name: %s
                <br> caption: %s
            </td>
            """
            % ("white", base64.b64encode(imgdata).decode(), ref, cap)
        )

        # tar
        img_path = os.path.join(img_root, tar + ".png")
        img = cv2.imread(img_path)
        imgdata = cv2.imencode(".jpg", img)[1].tobytes()
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                <br> target name: %s
            </td>
            """
            % ("white", base64.b64encode(imgdata).decode(), tar)
        )

        for name in sort_list.split(","):
            if name == "":
                continue
            img_path = os.path.join(img_root, name + ".png")
            img = cv2.imread(img_path)
            imgdata = cv2.imencode(".jpg", img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                </td>
                """
                % ("white", base64.b64encode(imgdata).decode(), name)
            )

        html_file_fp.write("</tr>\n")
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')

    html_file_fp.write("</table>\n")
    html_file_fp.write("</p>\n")
    html_file_fp.write("</body>\n</html>")


if __name__ == "__main__":
    input_path = "16891465670356793.txt"
    reftar_sort_dict = {}
    with open(input_path) as fr:
        lines = fr.readlines()
        for line in tqdm(lines):
            reftarcap, sort = line.split("@")
            reftar, cap = reftarcap.split("*")
            ref, tar = reftar.split("&")
            reftar_sort_dict[reftarcap] = sort[:-1]

    output_path = "show1.html"
    vis_ann(reftar_sort_dict, output_path)
