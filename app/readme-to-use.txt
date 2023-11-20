Launch the api service:
In terminal run:
python mainaip.py

##New routes:
http://localhost:5000/v1/64bit_string?number=23423423523523525
http://localhost:5000/v1/nbit_string?bit_size=64&number=23423423523523525

http://localhost:5000/v1/image?number=23423423523523525
http://localhost:5000/v1/bigger_image?number=23423423523523525&x=8&y=8

http://localhost:5000/v1/inflate_server_image?x=4&y=4&server_image_path=./lalala.png
http://localhost:5000/v1/inflate_server_image?x=4&y=4&server_image_path=./the_file.png

http://localhost:5000/v1/inflate_image_from_url?x=4&y=4&image_url=https://opengameart.org/sites/default/files/icons_32x32_7.png

unsure if these is actually useable:
http://localhost:5000/v1/pillow_image?number=23423423523523525
http://localhost:5000/v1/pillow_image_bytes?number=23423423523523525

####TODO: edit to match new routes

Launch browser and navigate to any of the following or similar:
http://localhost:5000/v1/64bit_string?number=4564747
http://localhost:5000/string_to_image?number_in_a_string=888888888885647790
http://localhost:5000/string_to_inflatedimage?x=16&y=16&number_in_a_string=765756865685
http://localhost:5000/inflate_image_from_image_url?x=2&y=2&image_url=http://www.small-icons.com/packs/32x32-free-design-icons.png

# wtih absolute path
http://localhost:5000/inflate_image_from_image_path?x=8&y=8&image_path=/home/rest_of_absolute_local_path/the_file.png

# with relative path
http://localhost:5000/inflate_image_from_image_path?x=8&y=8&image_path=the_file.png

curl -X POST -F "image=@/path/to/your/local/image.jpg" -F "output_name=your_output_name" http://yourdomain.com/save_image


curl -X POST -F "image=@the_file.png" -F "output_name=the_modified_file.png" http://localhost:5000/save_image


