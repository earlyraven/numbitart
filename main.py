from flask import Flask, request, make_response
import numpy as np
from PIL import Image
import io
from Service_Settings import provider_site_link
import requests

app = Flask(__name__)

def validate_positive_integer_input(value, param_name):
    try:
        validated_value = int(value)
        if validated_value <= 0:
            return generate_error_response(details=f"Invalid input for {param_name}. {param_name} must be a positive integer.", code=400)
        return validated_value
    except ValueError:
        return generate_error_response(details=f"Invalid input for {param_name}. Please enter a positive integer.", code=400)

def validate_string_fits_within(string, constraint):
    try:
        string_length = len(string)
        if string_length <= 0:
            return generate_error_response(details=f"Invalid input.  {string} is longer than {constraint} characters.", code=400)
    except ValueError:
        return generate_error_response(f"Invalid input for string.  Ensure the string can be cast to an integer."), 400

def generate_error_response(details, code=400):
    return {"error": {"details": f"{details}"}, "code": code}

def generate_successful_response(data, code=200):
    return {"data": {"value": f"{data}"}, "code": code}

def generate_detailed_error_response(details, message, code=400):
    return {"error": {"details": f"{details}", "message": f"{message}"}, "code": code}

def convert_number_to_nbit_string(bit_size, number):
    as_number = int(number)

    binary_number = ""
    
    while as_number > 0:
        remainder = as_number % 2
        binary_number = str(remainder) + binary_number
        as_number = as_number // 2
    
    if binary_number == "":
        return {"error": "Error. Negative values for number are not allowed.  Please use a non-negative integer."}

    padded_binary_string = binary_number.zfill(bit_size)
    
    if len(padded_binary_string) > bit_size:
        e_message = f"Error. The value was too big. {number} doesn't fit in {bit_size} bits."
        return generate_error_response(details=e_message, code=400)
    
    return padded_binary_string

def convert_number_to_64bit_string(number):
    bit_size = 64
    return convert_number_to_nbit_string(bit_size, number)

def convert_64bit_string_to_scaled_2d_64bit_string(number, x, y):
    base_string = convert_number_to_64bit_string(number)

    expanded_string = ""
    for i in range(len(base_string)):
        replaced_value = x*y*base_string[i:i+1]
        expanded_string += replaced_value
    return expanded_string

def convert_string_to_colors(s):
    return [[255, 255, 255] if char == '0' else [0, 0, 0] for char in s]

def convert_string_to_8by8_pillow_image(values):
    return Image.fromarray(np.array([v[0] for v in values]).reshape(8, 8).astype(np.uint8))

def convert_string_to_scaled_8by8_pillow_image(values, x, y, file_name):
    square_size = 8
    base_image = Image.fromarray(np.array([v[0] for v in values]).reshape(square_size, square_size).astype(np.uint8))
    bigger_image = base_image.resize((x*square_size, y*square_size))
    return process_image_response(bigger_image, file_name)

def convert_string_to_8xby8y_pillow_image(values, x, y):
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
            details="",
            message=e,
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
            details="",
            message=e,
        )

    # return generate_successful_response(convert_number_to_nbit_string(bit_size, number))
    the_result = convert_number_to_nbit_string(bit_size, number)
    return generate_successful_response(the_result)

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
            details="",
            message=e,
        )

    return generate_successful_response(convert_number_to_64bit_string(number))

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
            details="",
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
            details="",
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
            details="",
            message=e,
        )
    
    return generate_successful_response(convert_64bit_string_to_scaled_2d_64bit_string(number, x, y))

# RESTful endpoint for pillow image
@app.route('/v1/pillow_image', methods=['GET'])
def pillow_image_route():
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
            details="",
            message=e,
        )
    the_number_values_as_string = convert_number_to_64bit_string(number)
    return generate_successful_response(convert_string_to_8by8_pillow_image(the_number_values_as_string))

# RESTful endpoint for pillow image bytes
@app.route('/v1/pillow_image_bytes', methods=['GET'])
def pillow_image_bytes_route():
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
            details="",
            message=e,
        )
    the_number_values_as_string = convert_number_to_64bit_string(number)
    return generate_successful_response(convert_string_to_8by8_pillow_image(the_number_values_as_string).tobytes())

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
            details="",
            message=e,
        )
    return convert_number_to_64bit_image(number, f"{provider_site_link}_{number}_64bit")

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
            details="",
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
            details="",
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
            details="",
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
            details="",
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
            details="",
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
    
    try:
        the_server_image = Image.open(server_image_path)
    except FileNotFoundError as e:
        return generate_detailed_error_response(
            details="The requested file was not found on the server.",
            message=e,
            code=404,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details="",
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
            details="",
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
            details="",
            message=e,
        )
    if (y<1):
        return generate_detailed_error_response(
            details="Invalid value: y must be a positive integer string.",
            message="",
        )
    
    try:
        path = request.args.get('path')
    except TypeError as e:
        return generate_detailed_error_response(
            details="Missing parameter: path",
            message=e,
        )
    except ValueError as e:
        return generate_detailed_error_response(
            details="Invalid value: path must be a string.",
            message=e,
        )
    except Exception as e:
        return generate_detailed_error_response(
            details="",
            message=e,
        )
    
    try:
        input_image = Image.open(path)
    except FileNotFoundError as e:
        return generate_detailed_error_response(
            details="missing file.",
            message=e,
            code=404
        )
    except Exception as e:
        return generate_detailed_error_response(
            details="",
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
            details="",
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
            details="",
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
            details="Image was not found at theddddd requested image_url.",
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
            details="",
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

# RESTful endpoint for deflating images
@app.route('/v1/deflate_image', methods=['GET'])
def deflate_image_route():
    x = validate_positive_integer_input(request.args.get('x'), 'x')
    y = validate_positive_integer_input(request.args.get('y'), 'y')
    
    input_file_path = "./inflated_image_16by16.png"  # Replace with the actual file path
    input_image = Image.open(input_file_path)
    deflated_image = deflate(x, y, input_image)
    return process_image_response(deflated_image, f"{input_file_path.split('/')[-1]}_deflated_with_{x}by{y}")

if __name__ == '__main__':
    app.run(debug=False)
    # app.run(debug=True)
