from flask import Flask, request, make_response, jsonify
import numpy as np
from PIL import Image
import io
import requests
import os

import time
from io import BytesIO
import json
import base64

# This app is meant to be run (using gunicorn) from the next folder up from the current working directory.
byt_io = io.BytesIO()

provider_site_link = "csvup.com"

app = Flask(__name__)

def put_in_quotes(input_string):
    return f'"{input_string}"'

@app.route('/')
def hello():
    example_links = [
    "http://localhost:5000/v1/64bit_string?number=1234567890",
    "http://localhost:5000/v1/nbit_string?number=1234567890&bit_size=128",
    "http://localhost:5000/v1/base64_encoded_image?number=1234567890",
    "http://localhost:5000/v1/scaled_2d_64bit_string?number=1234567890&x=8&y=8",
    "http://localhost:5000/v1/color_array?number=1234567890",
    "http://localhost:5000/v1/base64_image_array_as_a_list?number=1234567890",

    # The following will download the resulting image file to the client's device:
    "http://localhost:5000/v1/image?number=1234567890",
    "http://localhost:5000/v1/bigger_image?number=1234567890&x=8&y=8",
    "http://localhost:5000/v1/deflate_image?x=4&y=4",
    "http://localhost:5000/v1/deflate_image?x=4&y=4&input_file_path=./app/inflated_image_16by16.png",
    "http://localhost:5000/v1/inflate_image_from_url?x=8&y=8&image_url=https://opengameart.org/sites/default/files/blocks_0.png",
    "http://localhost:5000/v1/inflate_image_from_path?x=8&y=8&image_path=./app/inflated_image_16by16.png",
    ]
    hello_message='''Hello, Flask and Gunicorn! Try navigating to the api endpoints.'''
    hello_message += "\nHere are some example endpoints:"
    for i in range(len(example_links)):
        example_link = example_links[i]
        the_line = "<p>"+"<a href="+put_in_quotes(example_link)+f">{example_link}</a>"+"</p>"
        hello_message += the_line
    return hello_message

def validate_positive_integer_input(value, param_name):
    try:
        validated_value = int(value)
        if validated_value <= 0:
            raise ValueError(f"Invalid input for {param_name}.  {param_name} must be a positive integer.")
        return validated_value
    except ValueError as e:
        return generate_detailed_error_response(
            details=f"Invalid input for {param_name}. Please enter a positive integer.",
            message=e,
        )

def validate_string_fits_within(string, constraint):
    try:
        string_length = len(string)
        if string_length <= 0:
            return generate_detailed_error_response(
                details=f"Invalid input.  {string} is longer than {constraint} characters.",
                message="",
            )
    except ValueError:
        return generate_detailed_error_response(
            details=f"Invalid input for string.  Ensure the string can be cast to an integer.",
            message="",
        )

def generate_detailed_error_response(details, message, code=400):
    return jsonify({"error": {"details": f"{details}", "message": f"{message}"}, "code": code})

def generate_successful_response_jsonified(data, code=200):
    return jsonify({"data": {"value": data}, "code": code})

def convert_number_to_nbit_string(bit_size, number):
   as_number = int(number)
   if as_number < 0:
       raise ValueError("Error.  Negative number detected.  Make sure the number field has a non-negative integer value.")

   binary_number = ""
   
   while as_number > 0:
       remainder = as_number % 2
       binary_number = str(remainder) + binary_number
       as_number = as_number // 2

   if binary_number == "":
       raise ValueError("Error.  Blank number detected.  Make sure the number field has a value.")

   padded_binary_string = binary_number.zfill(bit_size)
   
   if len(padded_binary_string) > bit_size:
       raise ValueError(f"Error. The number was too big to fit within the bit_size constraint. {number} doesn't fit in {bit_size} bits.")

   return padded_binary_string


def convert_number_to_64bit_string(number):
    bit_size = 64
    try:
        output = convert_number_to_nbit_string(bit_size, number)
    except ValueError as e:
        raise ValueError(e)
    return output

def convert_64bit_string_to_scaled_2d_64bit_string(number, x, y):
    base_string = convert_number_to_64bit_string(number)

    expanded_string = ""
    for i in range(len(base_string)):
        replaced_value = x*y*base_string[i:i+1]
        expanded_string += replaced_value
    return expanded_string

def convert_string_to_colors(s):
    if type(s) != type("a-string"):
        raise TypeError("Passed value for s must be a string.")
    required_string_length = 64
    if len(s) != required_string_length:
        raise ValueError("The length of the input string must be .")
    output = [[255, 255, 255] if char == '0' else [0, 0, 0] for char in s]
    if (type(output) != type([1,2,3])):
        raise TypeError("Does this trigger?  Error. The passed color data was not a string.  Please ensure this method get's passed a string.")
        if (len(output) != 64):
            raise ValueError("Not sure if this is ever reached: Error. The length of the output array is not 64.")
    return output

@app.route('/v1/color_array', methods=['GET'])
def color_array_route():
    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    try:
        output = convert_string_to_colors(convert_number_to_64bit_string(number))
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    return generate_successful_response_jsonified(output)

# RESTful endpoint for base64 encoded image
@app.route('/v1/base64_encoded_image', methods=['GET'])
def base64_encoded_image_route():
    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    try:
        the_number_values_as_string = convert_number_to_64bit_string(number)
    except TypeError as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    image_data = convert_string_to_colors(the_number_values_as_string)
    if type(the_number_values_as_string) != type("a-stirng"):
        return generate_detailed_error_response(
            details="Failed to parse image data.",
            message="Failure.",
        )
    if (type(image_data) != type([1,2,3])):
        return generate_detailed_error_response(
            details=image_data["error"]["details"],
            message=image_data["error"]["message"],
            code=image_data["code"]
        )
    image_array = np.array(image_data).reshape(8, 8, 3)  # Reshape to 8x8x3 array

    # Convert NumPy array to PNG image bytes
    image = Image.fromarray(image_array.astype(np.uint8))
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()

    # Encode bytes to base64 with padding
    image_base64 = base64.b64encode(image_bytes).decode('ascii')

    return generate_successful_response_jsonified(image_base64)

# RESTful endpoint for base64 image array as a list
@app.route('/v1/base64_image_array_as_a_list', methods=['GET'])
def base64_image_array_route():
    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    try:
        the_number_values_as_string = convert_number_to_64bit_string(number)
    except TypeError as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    image_data = convert_string_to_colors(the_number_values_as_string)
    if type(the_number_values_as_string) != type("a-string"):
        return generate_detailed_error_response(
            details="Failed to parse image data.",
            message="Failure.",
        )
    if (type(image_data) != type([1,2,3])):
        return generate_detailed_error_response(
            details=image_data["error"]["details"],
            message=image_data["error"]["message"],
            code=image_data["code"]
        )
    the_result = np.array(image_data).reshape(8,8,3).tolist()

    return generate_successful_response_jsonified(the_result)

def convert_string_to_8by8_pillow_image(values):
    try:
        square_size = 8
        output = Image.fromarray(np.array([v[0] for v in values]).reshape(square_size, square_size).astype(np.uint8))
    except Exception as e:
        raise Exception(e)
    return output

def convert_string_to_scaled_8by8_pillow_image(values, x, y, file_name):
    square_size = 8
    base_image = convert_string_to_8by8_pillow_image(values)
    bigger_image = base_image.resize((x*square_size, y*square_size))
    return process_image_response(bigger_image, file_name)

def convert_string_to_8xby8y_pillow_image(values, x, y):
    square_size = 8
    return Image.fromarray(np.array([v[0] for v in values]).reshape(8*x, 8*y).astype(np.uint8))

def convert_string_to_8by8_2d_pillow_image(values, x, y):
    return Image.fromarray(np.array([v[0] for v in values]).reshape(8*x, 8*y).astype(np.uint8))

def _convert_image_to_bytes_array(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def _download_bytes_array(bytes_array, filename_prefix):
    response = make_response(bytes_array)
    response.headers.set('Content-Type', 'image/png')
    response_filename = filename_prefix + ".png"
    response.headers.set('Content-Disposition', 'attachment', filename=response_filename)
    return response

def process_image_response(image, filename_prefix):
    img_byte_arr = _convert_image_to_bytes_array(image)
    return _download_bytes_array(img_byte_arr, filename_prefix)

def process_image_data_response(image):
    img_byte_arr = _convert_image_to_bytes_array(image)
    return img_byte_arr

def inflate(x, y, input_image):
    input_array = np.array(input_image)
    width, height = input_image.size
    inflated_array = np.zeros((height * y, width * x, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            inflated_array[i * y:(i + 1) * y, j * x:(j + 1) * x] = input_array[i, j]

    return Image.fromarray(inflated_array)

def deflate(x, y, input_image):
    input_array = np.array(input_image)
    deflated_array = np.zeros((input_array.shape[0] // y, input_array.shape[1] // x, 3), dtype=np.uint8)

    for i in range(deflated_array.shape[0]):
        for j in range(deflated_array.shape[1]):
            block = input_array[i * y:(i + 1) * y, j * x:(j + 1) * x]
            color = np.mean(block, axis=(0, 1)).astype(np.uint8)
            deflated_array[i, j] = color

    return Image.fromarray(deflated_array)

def convert_number_to_64bit_only_image(number):
    the_64bit_string = convert_number_to_64bit_string(number)
    the_color_array_of_arrays = convert_string_to_colors(the_64bit_string)
    the_image = convert_string_to_8by8_pillow_image(the_color_array_of_arrays)
    return the_image

def convert_number_to_64bit_image(number, filename_prefix):
    the_64bit_string = convert_number_to_64bit_string(number)
    the_color_array_of_arrays = convert_string_to_colors(the_64bit_string)
    the_image = convert_string_to_8by8_pillow_image(the_color_array_of_arrays)
    return process_image_response(the_image, filename_prefix)

def convert_number_to_2d_64bit_image(number, x, y, filename_prefix):
    the_64bit_string = convert_number_to_64bit_string(number)
    output_file_name = filename_prefix + ".png"
    the_image_to_use = convert_string_to_scaled_8by8_pillow_image(the_64bit_string, x, y, output_file_name)
    return process_image_response(the_image_to_use, filename_prefix)

# RESTful endpoint for nbit string
@app.route('/v1/nbit_string', methods=['GET'])
def nbit_string_route():
    try:
        bit_size = int(request.args.get('bit_size'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: bit_size.",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: bit_size must be a positive integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (bit_size<1):
        return generate_detailed_error_response(
            details="Invalid value: bit_size must be a positive integer string.",
            message="",
        )

    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    try:
        the_result = convert_number_to_nbit_string(bit_size, number)
    except ValueError as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    return generate_successful_response_jsonified(the_result)

# RESTful endpoint for 64bit string
@app.route('/v1/64bit_string', methods=['GET'])
def a64bit_string_route():
    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    try:
        output = convert_number_to_64bit_string(number)
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details="Default error.",
            message=e,
        )
    if (type(output) != type("a-string")):
        return generate_detailed_error_response(
            details=output["error"]["details"],
            message=output["error"]["message"],
        )
    return generate_successful_response_jsonified(output)

# RESTful endpoint for scaled 2d 64bit string
@app.route('/v1/scaled_2d_64bit_string', methods=['GET'])
def scaled_2d_64bit_string_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )

    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    
    return generate_successful_response_jsonified(convert_64bit_string_to_scaled_2d_64bit_string(number, x, y))

# RESTful endpoint for image
@app.route('/v1/image', methods=['GET'])
def image_route():
    try:
        number = validate_positive_integer_input(request.args.get('number'), 'number')
        # number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    try:
        the_result = convert_number_to_64bit_image(number, f"{provider_site_link}_{number}_64bit")
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    return the_result

# RESTful endpoint for bigger image
@app.route('/v1/bigger_image', methods=['GET'])
def bigger_image_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )

    try:
        number = int(request.args.get('number'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: number",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: number must be a non-negative integer.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    the_image_to_use = convert_number_to_64bit_only_image(number)
    the_expanded_image = the_image_to_use.resize((8*x,8*y), Image.NEAREST)
    return process_image_response(the_expanded_image, f"{provider_site_link}_scaled_{x}_by_{y}_{number}_64bit")

# RESTful endpoint for image
@app.route('/v1/named_image', methods=['GET'])
def named_image_route():
    number = validate_positive_integer_input(request.args.get('number'), 'number')
    file_name = None
    file_name = request.args.get('file_name')
    if file_name is None:
        return generate_detailed_error_response(
            details="Missing parameter: file_name",
            message="",
        )
    return convert_number_to_64bit_image(number, file_name)

# RESTful endpoint for inflating server image
@app.route('/v1/inflate_server_image', methods=['GET'])
def inflate_server_image_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )

    try:
        server_image_path = request.args.get('server_image_path')
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: server_image_path",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: server_image_path must be a non-empty string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details="invalid value: server_image_path must be a non-empty string.",
            message=e,
        )
    if server_image_path is None:
        return generate_detailed_error_response(
            details="Missing parameter: server_image_path",
            message="",
        )

    try:
        the_server_image = Image.open(server_image_path)
    except TypeError as e:
        return generate_detailed_error_response(
            details="bad",
            message=e,
        )
    except FileNotFoundError as e:
        return generate_detailed_error_response(
            details="The requested file was not found on the server.",
            message=e,
            code=404,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    inflated_image = inflate(x, y, the_server_image)
    return process_image_response(inflated_image, f"inflated_image_{x}by{y}")

# RESTful endpoint for inflating images from file path
@app.route('/v1/inflate_image_from_path', methods=['GET'])
def inflate_image_from_path_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )

    image_path = None
    try:
        image_path = request.args.get('image_path')
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: image_path",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: image_path must be a string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if image_path is None:
        return generate_detailed_error_response(
            details="Missing parameter: image_path",
            message="",
        )

    try:
        input_image = Image.open(image_path)
    except FileNotFoundError as e:
        return generate_detailed_error_response(
            details=f"Missing file. Searched for image on server at {image_path} but did not find it there.",
            message=e,
            code=404
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    
    inflated_image = inflate(x, y, input_image)
    return process_image_response(inflated_image, f"inflated_image_{x}by{y}")

# RESTful endpoint for inflating images from URL
@app.route('/v1/inflate_image_from_url', methods=['GET'])
def inflate_image_from_image_url_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )
    
    image_url = None
    image_url = request.args.get('image_url')
    if image_url is None:
        return generate_detailed_error_response(
            details="Missing parameter: image_url",
            message="",
        )

    try:
        r = requests.get(image_url, stream=True)
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: image_url",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Image was not found at the requested image_url.",
            message=e,
            code=404
        )
    except ConnectionResetError as e:
        return generate_detailed_error_response(
            details="Failed to connect.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    if r.status_code != 200:
        return generate_detailed_error_response(
            details=f"Missing result.  Code recieved: {r.status_code}",
            message="",
        )
    input_image = Image.open(io.BytesIO(r.content))
    rgb_image = input_image.convert("RGB")
    inflated_image = inflate(x, y, rgb_image)
    filepath = f"fetched_and_scaled_{x}_by_{y}"
    return process_image_response(inflated_image, filepath)

# RESTful endpoint for deflating images that are located on the server.
@app.route('/v1/deflate_image', methods=['GET'])
def deflate_image_route():
    try:
        x = int(request.args.get('x'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: x",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (x<1):
        return generate_detailed_error_response(
            details="Invalid value: x must be a positive integer string.",
            message="",
        )

    try:
        y = int(request.args.get('y'))
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: y",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )
    try:
        input_file_path = request.args.get('input_file_path')
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )
    if input_file_path is None:
        input_file_path = "./app/inflated_image_16by16.png"  # Replace with the actual file path
    try:
        input_image = Image.open(input_file_path)
    except FileNotFoundError as e:
        return generate_detailed_error_response(
            details="missing file.",
            message=e,
            code=404
        )
    except Exception as e:
        return generate_detailed_error_response(
            details=type(e),
            message=e,
        )

    deflated_image = deflate(x, y, input_image)
    return process_image_response(deflated_image, f"{input_file_path.split('/')[-1]}_deflated_with_{x}by{y}")

if __name__ == '__main__':
    app.run(debug=True)
