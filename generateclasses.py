import json
import os
import re
from collections import defaultdict


def get_parameter_declaration(cpp_type):
    """
    Returns the parameter declaration string based on the C++ type.
    Uses const reference for types that benefit from it (e.g., std::string, std::vector, std::map, etc.).
    
    Args:
        cpp_type (str): The C++ type of the parameter.
    
    Returns:
        str: The parameter declaration string.
    """
    # Types that should be passed by const reference to avoid unnecessary copying
    const_ref_types = [
        'std::string',
        'std::pair<int, int>',
        'std::vector<int>',
        'std::vector<std::string>',
        'std::map<std::string, std::string>',
        'std::vector<uint8_t>',
        'std::vector<double>',
        'std::vector<float>',
        'std::vector<uint64_t>',
        'std::vector<int64_t>',
        # Add more types as needed
    ]
    
    if cpp_type in const_ref_types:
        return f'const {cpp_type}& value'
    else:
        return f'{cpp_type} value'


def to_valid_cpp_identifier(name):
    """
    Converts a string to a valid C++ identifier by:
    - Replacing invalid characters with underscores.
    - Appending an underscore if the name is a C++ keyword.
    - Prepending an underscore if the name starts with a digit.
    """
    # List of C++ keywords to check against
    cpp_keywords = {
        'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor',
        'bool', 'break', 'case', 'catch', 'char', 'char16_t', 'char32_t', 'class',
        'compl', 'const', 'constexpr', 'const_cast', 'continue', 'decltype', 'default',
        'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum', 'explicit', 'export',
        'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int',
        'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
        'operator', 'or', 'or_eq', 'private', 'protected', 'public', 'register',
        'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static',
        'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local',
        'throw', 'true', 'try', 'typedef', 'typeid', 'typename', 'union', 'unsigned',
        'using', 'virtual', 'void', 'volatile', 'wchar_t', 'while', 'xor', 'xor_eq'
    }

    # Replace invalid characters with underscores
    identifier = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # If the identifier is a C++ keyword, append an underscore
    if identifier in cpp_keywords:
        identifier += '_'

    # If the identifier starts with a digit, prepend an underscore
    if re.match(r'^\d', identifier):
        identifier = '_' + identifier

    return identifier

def capitalize_first_letter(s):
    """
    Capitalizes the first letter of a string.
    """
    return s[0].upper() + s[1:] if s else s

# General type mapping from FFmpeg types to C++ types
type_mapping = {
    'Flags': 'int',  # 'Flags' options will be treated as integers
    'Integer': 'int',
    'Integer64': 'int64_t',
    'Unsigned Integer64': 'uint64_t',
    'Double': 'double',
    'Float': 'float',
    'String': 'std::string',
    'Rational': 'std::pair<int, int>',  # Use pair<int, int> for fractions
    'Binary': 'std::vector<uint8_t>',
    'Dictionary': 'std::map<std::string, std::string>',
    'Constant': 'const int',  # Constants can be represented as const ints
    'Image Size': 'std::pair<int, int>',  # Width and height
    'Pixel Format': 'std::string',  # Changed from 'int' to 'std::string'
    'Sample Format': 'std::string',  # Changed from 'int' to 'std::string'
    'Video Rate': 'std::pair<int, int>',
    'Duration': 'int64_t',
    'Color': 'std::string',  # Could be custom Color class
    'Boolean': 'bool',
    'Channel Layout': 'uint64_t',  # Could be custom type if needed
    'Flag Array': 'std::vector<int>',
    'Unknown': 'void*',  # Fallback to void* if type is unknown
}

# Per-option type overrides for specific options that require special handling
option_type_overrides = {
    'sample_fmts': 'std::vector<std::string>',  # Changed from 'std::vector<int>' to 'std::vector<std::string>'
    'sample_rates': 'std::vector<int>',
    'channel_layouts': 'std::vector<uint64_t>',
    'pix_fmts': 'std::vector<std::string>',  # Changed from 'std::vector<int>' to 'std::vector<std::string>'
    # Add more specific mappings as needed
}

def get_cpp_default_value(cpp_type, default_value, possible_values=None):
    """
    Converts the default value from the JSON to a valid C++ default value literal.
    """
    # Since we're not generating enums here, treat possible_values as documentation only
    if cpp_type.startswith('std::vector'):
        return '{}'
    elif cpp_type == 'std::string':
        if default_value and default_value.lower() != 'no default':
            escaped_default = default_value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped_default}"'
        else:
            return '""'
    elif cpp_type == 'bool':
        return 'true' if str(default_value).lower() == 'true' else 'false'
    elif cpp_type == 'std::pair<int, int>':
        if default_value.lower() in ['no default', '']:
            return '{0, 1}'
        else:
            if isinstance(default_value, str) and '/' in default_value:
                num, den = default_value.split('/')
                return f'{{{num.strip()}, {den.strip()}}}'
            else:
                return '{0, 1}'
    elif cpp_type == 'int':
        if default_value.lower() in ['no default', '']:
            return '0'
        else:
            if re.match(r'^-?\d+$', default_value):
                return default_value
            else:
                return '0'
    elif cpp_type in ['uint64_t', 'int64_t']:
        if default_value.lower() in ['no default', '']:
            return '0'
        else:
            if re.match(r'^\d+$', default_value):
                return default_value + 'ULL'
            else:
                return '0'
    elif cpp_type.startswith('const int'):
        return default_value if default_value and default_value.lower() != 'no default' else '0'
    else:
        if re.match(r'^-?\d+(\.\d+)?$', str(default_value)):
            return default_value
        else:
            return '0'

def adjust_default_value_for_binding(cpp_type, cpp_default_value):
    """
    Adjusts the default value to be acceptable in the binding code.
    """
    if cpp_type == 'std::string':
        return cpp_default_value  # Should be in the format '"string"'
    elif cpp_type == 'bool':
        return cpp_default_value  # 'true' or 'false'
    elif cpp_type in ['int', 'double', 'float', 'uint64_t', 'int64_t']:
        return cpp_default_value  # Numeric value
    elif cpp_type.startswith('std::pair'):
        # Remove outer braces if any
        value_without_braces = cpp_default_value.strip('{}')
        # Extract types from std::pair<type1, type2>
        match = re.match(r'std::pair<\s*([^,]+)\s*,\s*([^>]+)\s*>', cpp_type)
        if match:
            type1, type2 = match.groups()
            return f'std::make_pair<{type1.strip()}, {type2.strip()}>({value_without_braces})'
        else:
            return f'std::make_pair({value_without_braces})'
    elif cpp_type.startswith('std::vector'):
        # Extract the inner type
        match = re.match(r'std::vector<\s*([^>]+)\s*>', cpp_type)
        if match:
            inner_type = match.group(1).strip()
            return f'std::vector<{inner_type}>()'
        else:
            return '{}'
    elif cpp_type.startswith('std::map'):
        return '{}'
    else:
        return None  # Cannot specify default value

def generate_variable_name(option_name, help_text, unit=''):
    """
    Generates a more descriptive variable name based on the option's help text and unit.
    """
    # If the option name is more than one character, use it
    if len(option_name) > 1:
        return to_valid_cpp_identifier(option_name)
    else:
        # Try to extract meaningful words from help text or unit
        source_text = help_text.strip() or unit.strip()
        # Remove leading 'set the', 'set', or 'Set the'
        source_text = re.sub(r'^(set the|Set the|set)\s+', '', source_text, flags=re.IGNORECASE)
        # Remove common endings like 'expression', 'value', 'option'
        source_text = re.sub(r'\b(expression|value|option)\b', '', source_text, flags=re.IGNORECASE)
        # Remove any non-word characters
        source_text = re.sub(r'[^\w\s]', '', source_text)
        # Split into words
        words = source_text.strip().split()
        if not words:
            return to_valid_cpp_identifier(option_name)
        # Generate camelCase name
        var_name = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        var_name = to_valid_cpp_identifier(var_name)
        return var_name

def ensure_unique_name(name, used_names):
    """
    Ensures that the name is unique within used_names by appending numbers if necessary.
    """
    original_name = name
    counter = 1
    while name in used_names:
        name = f'{original_name}_{counter}'
        counter += 1
    used_names.add(name)
    return name

def is_audio_filter(filter_info):
    """
    Determines if a filter is an audio filter based on its options' flags.

    A filter is considered an audio filter if any of its options have
    "flags" containing "Audio" or "Audio Filtering".

    Args:
        filter_info (dict): The filter information from JSON.

    Returns:
        bool: True if the filter is an audio filter, False otherwise.
    """
    for option in filter_info.get('options', []):
        flags = option.get('flags', '')
        # Check if it contains "Audio Filtering" or "Audio"
        if "Audio Filtering" in flags or "Audio" in flags:
            return True
    return False

def generate_cpp_class(filter_info):
    """
    Generates the C++ class definition and implementation for a given FFmpeg filter,
    handling option aliases, and selecting the longest name as primary.

    Returns:
        tuple: (header_code, source_code, binding_code, member_variables)
    """
    filter_name = filter_info['filter_name']
    options = filter_info['options']
    filter_description = filter_info.get('description', f'{filter_name} filter.')

    # Convert filter name to a valid C++ class name
    class_name = to_valid_cpp_identifier(capitalize_first_letter(filter_name))

    # Initialize used names sets for this class
    used_variable_names = set()
    used_method_names = set()

    # Start building the class definition (.hpp)
    class_code = []
    class_code.append('#pragma once')  # Include guard
    class_code.append('#include "FilterBase.hpp"')  # Include the base class
    class_code.append('#include <string>')
    class_code.append('#include <vector>')
    class_code.append('#include <map>')
    class_code.append('#include <utility>')
    class_code.append('')
    class_code.append(f'class {class_name} : public FilterBase {{')
    class_code.append('public:')
    class_code.append('    /**')
    class_code.append(f'     * {filter_description}')
    class_code.append('     */')

    # Group options by their 'help', 'type', 'default', and 'unit' to find aliases
    option_groups = defaultdict(list)
    for option in options:
        key = (
            option.get('help', ''),
            option.get('type', ''),
            option.get('default', ''),
            option.get('unit', '')
        )
        option_groups[key].append(option)

    # Keep track of member variables with types and default values
    member_variables = {}

    for group_key, option_group in option_groups.items():
        # Sort options to prefer longer names
        option_group.sort(key=lambda opt: len(opt['name']))
        # Use the option with the longest name as the primary
        primary_option = option_group[-1]
        alias_names = [opt['name'] for opt in option_group if opt['name'] != primary_option['name']]
        help_text = primary_option.get('help', '').replace('\n', ' ')
        # Generate a more descriptive variable name
        option_name_candidate = generate_variable_name(primary_option['name'], help_text, primary_option.get('unit', ''))
        option_name = ensure_unique_name(option_name_candidate, used_variable_names)
        method_name = capitalize_first_letter(option_name)
        method_name = ensure_unique_name(method_name, used_method_names)

        option_type = primary_option['type']

        # Check for per-option overrides
        cpp_type = option_type_overrides.get(primary_option['name'])
        if not cpp_type:
            # Use the general type mapping
            cpp_type = type_mapping.get(option_type, 'void*')  # Default to void* if type unknown

        required = primary_option.get('required', False)
        default = primary_option.get('default', '')
        unit = primary_option.get('unit', '')
        possible_values = primary_option.get('possible_values', [])
        deprecated = primary_option.get('deprecated', False)
        readonly = primary_option.get('readonly', False)

        # Skip deprecated options
        if deprecated:
            continue

        # Handle all options uniformly, including 'Flags' as integers
        cpp_default_value = get_cpp_default_value(cpp_type, default, possible_values)
        member_variables[option_name] = {
            'cpp_type': cpp_type,
            'cpp_default_value': cpp_default_value,
            'method_name': method_name,
            'help_text': help_text,
            'alias_names': alias_names,
            'option_type': option_type,
            'required': required,
            'default': default,
            'unit': unit,
            'possible_values': possible_values,
            'deprecated': deprecated,
            'readonly': readonly,
            'original_option_name': primary_option['name']  # **Added Line**
        }

        # Add comments with option details
        class_code.append(f'    /**')
        class_code.append(f'     * {help_text}')
        if alias_names:
            aliases = ', '.join(alias_names)
            class_code.append(f'     * Aliases: {aliases}')
        if unit:
            class_code.append(f'     * Unit: {unit}')
        if possible_values:
            values_str = ', '.join([f'{val["name"]} ({val["value"]})' for val in possible_values])
            class_code.append(f'     * Possible Values: {values_str}')
        if readonly:
            class_code.append(f'     * Read-Only')
        class_code.append(f'     * Type: {option_type}')
        class_code.append(f'     * Required: {"Yes" if required else "No"}')
        class_code.append(f'     * Default: {default}')
        class_code.append(f'     */')

        # Generate setter and getter declarations, skip setters for read-only options
        if not readonly:
            # Use the helper function to get the correct parameter declaration
            param_declaration = get_parameter_declaration(cpp_type)
            # Extract parameter type from the declaration
            param_type = param_declaration.rsplit(' ', 1)[0]
            class_code.append(f'    void set{method_name}({param_declaration});')
        class_code.append(f'    {cpp_type} get{method_name}() const;')
        class_code.append('')

    # Generate constructor with all required arguments and defaults where possible
    parameters_with_defaults = []
    for option_name, var_info in member_variables.items():
        if var_info['readonly']:
            continue  # Skip read-only options in constructor
        cpp_type = var_info['cpp_type']
        cpp_default_value = var_info['cpp_default_value']
        default_value_for_param = adjust_default_value_for_binding(cpp_type, cpp_default_value)
        if default_value_for_param is not None:
            if cpp_type == 'std::string':
                param_str_with_default = f'const {cpp_type}& {option_name} = {default_value_for_param}'
            else:
                param_str_with_default = f'{cpp_type} {option_name} = {default_value_for_param}'
        else:
            # Use the helper function to conditionally add 'const &' only for std::string
            param_str_with_default = f'const {cpp_type}& {option_name}' if cpp_type == 'std::string' else f'{cpp_type} {option_name}'
        parameters_with_defaults.append(param_str_with_default)

    params_str_with_defaults = ', '.join(parameters_with_defaults)

    # Add constructor declaration
    class_code.append(f'    {class_name}({params_str_with_defaults});')
    class_code.append(f'    virtual ~{class_name}();')  # Destructor
    class_code.append('')
    # Add getFilterDescription declaration
    class_code.append('    std::string getFilterDescription() const override;')
    class_code.append('')
    class_code.append('private:')
    class_code.append('    // Option variables')

    for option_name, var_info in member_variables.items():
        cpp_type = var_info['cpp_type']
        class_code.append(f'    {cpp_type} {option_name}_;')

    class_code.append('};')
    class_code.append('')

    # Start building the class implementation (.cpp)
    source_code = []
    # Adjust the include path to the header file
    source_code.append(f'#include "{class_name}.hpp"')
    source_code.append('#include <sstream>')
    source_code.append('')
    # Constructor implementation
    if parameters_with_defaults:
        param_list = ', '.join([
                f'const {var_info["cpp_type"]}& {option_name}' if var_info["cpp_type"] == 'std::string' else f'{var_info["cpp_type"]} {option_name}'
                for option_name, var_info in member_variables.items()
                if not var_info['readonly']
            ])

        source_code.append(f'{class_name}::{class_name}({param_list}) {{')
        source_code.append('    // Initialize member variables from parameters')
        for option_name, var_info in member_variables.items():
            if var_info['readonly']:
                continue
            source_code.append(f'    this->{option_name}_ = {option_name};')
        source_code.append('}')
    else:
        # Default constructor
        source_code.append(f'{class_name}::{class_name}() {{')
        source_code.append('    // Initialize member variables with default values')
        for option_name, var_info in member_variables.items():
            source_code.append(f'    this->{option_name}_ = {var_info["cpp_default_value"]};')
        source_code.append('}')
    source_code.append('')

    # Destructor implementation
    source_code.append(f'{class_name}::~{class_name}() {{')
    source_code.append('    // Destructor implementation (if needed)')
    source_code.append('}')
    source_code.append('')

    # Implement setters and getters
    for option_name, var_info in member_variables.items():
        method_name = var_info['method_name']
        cpp_type = var_info['cpp_type']
        original_option_name = var_info['original_option_name']  # **Retrieve Original Option Name**

        # Setter implementation, skip if read-only
        if not var_info['readonly']:
            if cpp_type == 'std::string' and var_info['possible_values']:
                # Generate setter with validation
                source_code.append(f'void {class_name}::set{method_name}(const std::string& value) {{')
                # Define valid values
                valid_values = ', '.join([f'"{val["name"]}"' for val in var_info['possible_values']])
                source_code.append(f'    static const std::vector<std::string> valid_values = {{{valid_values}}};')
                source_code.append(f'    if (std::find(valid_values.begin(), valid_values.end(), value) == valid_values.end()) {{')
                source_code.append(f'        throw std::invalid_argument("Invalid value for {original_option_name}: " + value);')
                source_code.append('    }')
                source_code.append(f'    {option_name}_ = value;')
                source_code.append('}')
                source_code.append('')
            else:
                # Use the helper function to get the correct parameter declaration
                param_declaration = get_parameter_declaration(cpp_type)
      # **Fix Starts Here**
                # Option 1: Using split()
                # param_name = param_declaration.split()[-1]
                
                # Option 2: Using regex
                match = re.match(r'.*\b(\w+)$', param_declaration)
                if match:
                    param_name = match.group(1)
                else:
                    raise ValueError(f"Cannot parse parameter name from declaration: {param_declaration}")
                # **Fix Ends Here**
                source_code.append(f'void {class_name}::set{method_name}({param_declaration}) {{')
                source_code.append(f'    {option_name}_ = {param_name};')
                source_code.append('}')
                source_code.append('')

        # Getter implementation
        source_code.append(f'{cpp_type} {class_name}::get{method_name}() const {{')
        source_code.append(f'    return {option_name}_;')
        source_code.append('}')
        source_code.append('')

    # Implement getFilterDescription
    source_code.append(f'std::string {class_name}::getFilterDescription() const  {{')
    source_code.append('    std::ostringstream desc;')
    source_code.append(f'    desc << "{filter_name}";')  # Correct usage of filter_name

    source_code.append('')
    source_code.append('    bool first = true;')
    source_code.append('')

    # Append options
    for option_name, var_info in member_variables.items():
        cpp_type = var_info['cpp_type']
        default_value = var_info['cpp_default_value']
        original_option_name = var_info['original_option_name']  # **Retrieve Original Option Name**

        if var_info['option_type'] == "Flags":
            # Handle Flags as integers
            condition = f'{option_name}_ != {default_value}'
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << {option_name}_;')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type == 'std::string':
            condition = f'!{option_name}_.empty()' if default_value == '""' else f'{option_name}_ != {default_value}'
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << {option_name}_;')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type in ['int', 'int64_t', 'uint64_t']:
            if default_value in ['0', '0ULL', '0LL']:
                condition = f'{option_name}_ != {default_value}'
            else:
                condition = f'{option_name}_ != {default_value}'
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << {option_name}_;')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type in ['double', 'float']:
            if default_value in ['0.0', '0.0f']:
                condition = f'{option_name}_ != {default_value}'
            else:
                condition = f'{option_name}_ != {default_value}'
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << {option_name}_;')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type == 'bool':
            condition = f'{option_name}_ != {default_value}'
            source_code.append(f'    if ({condition}) {{')
            # Represent boolean as 1 (true) or 0 (false)
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << ({option_name}_ ? "1" : "0");')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type.startswith('std::pair'):
            # Extract default numerator and denominator
            if default_value.startswith('{') and default_value.endswith('}'):
                default_pair = default_value.strip('{}').split(',')
                if len(default_pair) == 2:
                    num, den = default_pair
                    condition = f'{option_name}_.first != {num.strip()} || {option_name}_.second != {den.strip()}'
                else:
                    condition = 'true'  # Fallback condition
            else:
                condition = 'true'  # Fallback condition
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=" << {option_name}_.first << "/" << {option_name}_.second;')
            source_code.append('        first = false;')
            source_code.append('    }')
        elif cpp_type.startswith('std::vector'):
            # Compare with default vector
            if default_value == '{}':
                condition = f'!{option_name}_.empty()'
            else:
                # For simplicity, treat non-empty as different
                condition = f'!{option_name}_.empty()'
            source_code.append(f'    if ({condition}) {{')
            source_code.append(f'        desc << (first ? "=" : ":") << "{original_option_name}=";')
            source_code.append(f'        for (size_t i = 0; i < {option_name}_.size(); ++i) {{')
            if 'uint8_t' in cpp_type:
                source_code.append(f'            desc << static_cast<int>({option_name}_[i]);')
            else:
                source_code.append(f'            desc << {option_name}_[i];')
            source_code.append(f'            if (i != {option_name}_.size() - 1) desc << ",";')
            source_code.append('        }')
            source_code.append('        first = false;')
            source_code.append('    }')
        else:
            # Placeholder for other types
            source_code.append(f'    // Handle type {cpp_type} for option {original_option_name}')
            source_code.append('    /* Add appropriate handling */')

    source_code.append('')
    source_code.append('    return desc.str();')
    source_code.append('}')
    source_code.append('')

    # Generate binding code for this filter
    binding_header_code, binding_source_code = generate_binding_code(class_name, member_variables)

    # Return class code, source code, binding code, and member_variables
    return '\n'.join(class_code), '\n'.join(source_code), (binding_header_code, binding_source_code), member_variables

def generate_binding_code(class_name, member_variables):
    """
    Generates Pybind11 binding code for a given C++ class and its member variables.

    Args:
        class_name (str): The name of the C++ class.
        member_variables (dict): A dictionary where each key is the member variable name, and
                                 the value is another dict containing 'cpp_type', 'cpp_default_value',
                                 and 'method_name'.

    Returns:
        tuple: (binding_header_code, binding_source_code)
    """
    # Binding header file name
    binding_header_code = []
    binding_header_code.append(f'#pragma once')
    binding_header_code.append(f'#include "{class_name}.hpp"')
    binding_header_code.append('#include <pybind11/pybind11.h>')
    binding_header_code.append('#include <pybind11/stl.h>')
    binding_header_code.append('#include "FilterFactory.hpp"')  # Include FilterFactory for string_to_filter_type
    binding_header_code.append('')
    binding_header_code.append('namespace py = pybind11;')
    binding_header_code.append('')
    binding_header_code.append(f'void bind_{class_name}(py::module_ &m);')
    binding_header_code.append('')
    
    # Create binding source code
    binding_source_code = []
    binding_source_code.append(f'#include "{class_name}_bindings.hpp"')
    binding_source_code.append('')
    binding_source_code.append('namespace py = pybind11;')
    binding_source_code.append('')
    binding_source_code.append(f'void bind_{class_name}(py::module_ &m) {{')
    # Expose the class
    binding_source_code.append(f'    py::class_<{class_name}, FilterBase, std::shared_ptr<{class_name}>>(m, "{class_name}")')
    # Prepare constructor with argument types
    cpp_types = [var_info['cpp_type'] for var_info in member_variables.values() if not var_info['readonly']]
    if cpp_types:
        constructor_signature = f'py::init<{", ".join(cpp_types)}>()'
    else:
        constructor_signature = 'py::init<>()'  # Default constructor if no parameters
    # Prepare arguments with defaults
    arg_specs = []
    for option_name, var_info in member_variables.items():
        if var_info['readonly']:
            continue
        cpp_type = var_info['cpp_type']
        cpp_default_value = var_info['cpp_default_value']
        default_value_for_param = adjust_default_value_for_binding(cpp_type, cpp_default_value)
        if default_value_for_param is not None:
            arg_specs.append(f'py::arg("{option_name}") = {default_value_for_param}')
        else:
            arg_specs.append(f'py::arg("{option_name}")')
    if arg_specs:
        # Join arguments with proper indentation
        args_formatted = ',\n             '.join(arg_specs)
        constructor_with_args = f'        .def({constructor_signature},\n             {args_formatted})'
    else:
        constructor_with_args = f'        .def({constructor_signature})'
    binding_source_code.append(constructor_with_args)
    # Bind setters and getters
    for option_name, var_info in member_variables.items():
        method_name = var_info['method_name']
        cpp_type = var_info['cpp_type']
        # Setter implementation, skip if read-only
        if not var_info['readonly']:
            binding_source_code.append(f'        .def("set{method_name}", &{class_name}::set{method_name})')
        # Getter implementation
        binding_source_code.append(f'        .def("get{method_name}", &{class_name}::get{method_name})')
    binding_source_code.append('        ;')  # End of class binding
    
    binding_source_code.append('}')
    
    return binding_header_code, binding_source_code

def write_binding_files(hpp_output_dir, cpp_output_dir, class_name, binding_header_code, binding_source_code):
    """
    Writes the binding header and source files to the appropriate directories.

    Args:
        hpp_output_dir (str): Directory for header files.
        cpp_output_dir (str): Directory for source files.
        class_name (str): Name of the filter class.
        binding_header_code (list): Lines of the binding header file.
        binding_source_code (list): Lines of the binding source file.
    """
    # Write binding header file
    binding_header_filename = os.path.join(hpp_output_dir, f'{class_name}_bindings.hpp')
    with open(binding_header_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(binding_header_code))
    print(f'Generated binding header: {binding_header_filename}')

    # Write binding source file
    binding_source_filename = os.path.join(cpp_output_dir, f'{class_name}_bindings.cpp')
    with open(binding_source_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(binding_source_code))
    print(f'Generated binding source: {binding_source_filename}')

def generate_filter_base(hpp_output_dir, cpp_output_dir):
    """
    Generates the FilterBase class header and source files.

    Args:
        hpp_output_dir (str): Directory for header files.
        cpp_output_dir (str): Directory for source files.
    """
    # Header file content
    header_content = [
        '#pragma once',

        '#include <algorithm>',
        '#include <cstdint> // For fixed-width integer types',
        '#include <exception>',
        '#include <iostream>',
        '#include <memory>',
        '#include <ostream>   // For std::ostream',
        '#include <stdexcept> // For std::runtime_error',
        '#include <string>',
        '#include <vector>',
        '#include <pybind11/pybind11.h>',
        '#include <pybind11/stl.h>',
        '',
        'class FilterBase {',
        'public:',
        '    FilterBase();',
        '    virtual ~FilterBase();',
        '',
        '    /**',
        '     * Get a description of the filter and its options.',
        '     * This function should be overridden by subclasses.',
        '     */',
        '    virtual std::string getFilterDescription() const;',
        '',
        'protected:',
        '    // Shared protected members (if any)',
        '',
        'private:',
        '    // Shared private members (if any)',
        '};',
        ''
    ]
    # Source file content
    source_content = [
        '#include "FilterBase.hpp"',
        '',
        'FilterBase::FilterBase() {',
        '    // Base class constructor implementation (if needed)',
        '}',
        '',
        'FilterBase::~FilterBase() {',
        '    // Base class destructor implementation (if needed)',
        '}',
        '',
        'std::string FilterBase::getFilterDescription() const {',
        '    return "Base filter";',
        '}',
        ''
    ]
    # Write to header file
    header_filename = os.path.join(hpp_output_dir, 'FilterBase.hpp')
    with open(header_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header_content))
    print(f'Generated {header_filename}')

    # Write to source file
    source_filename = os.path.join(cpp_output_dir, 'FilterBase.cpp')
    with open(source_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_content))
    print(f'Generated {source_filename}')

def generate_filter_factory(hpp_output_dir, cpp_output_dir, filter_class_names):
    """
    Generates the FilterFactory header and source files with an enum and a factory function,
    including string-to-enum conversion using magic_enum.
    
    Args:
        hpp_output_dir (str): Directory for header files.
        cpp_output_dir (str): Directory for source files.
        filter_class_names (list): List of filter class names.
    """
    # Header file content
    header_content = [
        '#pragma once',
        '#include "FilterBase.hpp"',
        '#include <memory>',
        '#include <string>',
      
        '// Enum listing all available filters',
        'enum class FilterType {',
    ]
    for class_name in filter_class_names:
        header_content.append(f'    {class_name},  // {class_name} filter')
    header_content.append('};')
    header_content.append('')
    header_content.append('// Factory function to create filters')
    header_content.append('std::shared_ptr<FilterBase> CreateFilter(FilterType type);')
    header_content.append('')

    header_content.append('')
    
    # Source file content
    source_content = [
        '#include "FilterFactory.hpp"',
      
        '#include <algorithm>',
        '#include <cctype>',
        '#include <stdexcept>',
    ]
    # Include all filter headers
    for class_name in filter_class_names:
        header_content.append(f'#include "{class_name}.hpp"')
    header_content.append('')
    
    # Implement the CreateFilter function
    source_content.append('std::shared_ptr<FilterBase> CreateFilter(FilterType type) {')
    source_content.append('    switch(type) {')
    
    for class_name in filter_class_names:
        source_content.append(f'        case FilterType::{class_name}:')
        source_content.append(f'            return std::make_shared<{class_name}>();')
    
    source_content.append('        default:')
    source_content.append('            return nullptr;')
    source_content.append('    }')
    source_content.append('}')
    source_content.append('')
    
    # Write to header file
    header_filename = os.path.join(hpp_output_dir, 'FilterFactory.hpp')
    with open(header_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(header_content))
    print(f'Generated {header_filename}')
    
    # Write to source file
    source_filename = os.path.join(cpp_output_dir, 'FilterFactory.cpp')
    with open(source_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_content))
    print(f'Generated {source_filename}')

def generate_filter_factory_bindings(filter_class_names):
    """
    Generates the binding code for the FilterFactory and FilterType enum.

    Args:
        filter_class_names (list): List of filter class names.

    Returns:
        list: Lines of binding code for the factory.
    """
    factory_binding_code = []
    # Expose FilterType enum
    factory_binding_code.append("    // Binding for FilterType enum")
    factory_binding_code.append("    py::enum_<FilterType>(m, \"FilterType\")")
    for class_name in filter_class_names:
        factory_binding_code.append(f"        .value(\"{class_name}\", FilterType::{class_name})")
    factory_binding_code.append("       ;")
    factory_binding_code.append("")
    # Bind CreateFilter function
    factory_binding_code.append("    // Binding for CreateFilter function")
    factory_binding_code.append("    m.def(\"CreateFilter\", &CreateFilter, py::arg(\"type\"));")
    factory_binding_code.append("")
    
    return factory_binding_code

def generate_filter_bindings_hpp(hpp_output_dir):
    """
    Generates the binding header file that declares the register_filters function.

    Args:
        hpp_output_dir (str): Directory where 'filter_bindings.hpp' will be saved.
    """
    binding_header_content = [
        '#pragma once',
        '#include <pybind11/pybind11.h>',
        '#include <pybind11/stl.h>',
        '',
        'namespace py = pybind11;',
        '',
        'void register_filters(py::module_ &m);',
        ''
    ]

    # Write the bindings.hpp file
    bindings_filename = os.path.join(hpp_output_dir, 'filter_bindings.hpp')
    with open(bindings_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(binding_header_content))
    print(f'Generated {bindings_filename}')

def generate_bindings_cpp(cpp_output_dir, chunk_headers, filter_class_names):
    """
    Generates the main binding source file that includes chunk headers and registers them.

    Args:
        cpp_output_dir (str): Directory for source files.
        chunk_headers (list): List of chunk header filenames without extension.
        filter_class_names (list): List of filter class names.
    """
    binding_source = [
        '#include <pybind11/pybind11.h>',
        '#include "FilterBase.hpp"',
        '#include "FilterFactory.hpp"',
        '#include "filter_bindings.hpp"',
        '// Include chunk headers',
    ]
    for chunk_name in chunk_headers:
        binding_source.append(f'#include "{chunk_name}.hpp"')
    
    # Implement the register_filters function
    register_filters_content = [
        'namespace py = pybind11;',
        '',
        'void register_filters(py::module_ &m) {',
        ' py::class_<FilterBase, std::shared_ptr<FilterBase>>(m, "FilterBase")',
        '    .def(py::init<>())',
        '    .def("getFilterDescription", &FilterBase::getFilterDescription);',
    ]
    
    # Register each chunk
    for chunk_name in chunk_headers:
        register_filters_content.append(f'    register_{chunk_name}(m);')
    
    # Register FilterFactory and FilterType
    register_filters_content.append('')
    register_filters_content.append('    // Register FilterType enum and CreateFilter function')
    register_filters_content.extend(generate_filter_factory_bindings(filter_class_names))
    register_filters_content.append('}')
    
    # Append the register_filters function to the source
    binding_source.append('// Implement the register_filters function')
    binding_source.extend(register_filters_content)
    binding_source.append('')
    
    # Write the bindings.cpp file
    bindings_filename = os.path.join(cpp_output_dir, 'filter_bindings.cpp')
    with open(bindings_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(binding_source))
    print(f'Generated {bindings_filename}')

def clean_dirs(hpp_output_dir, cpp_output_dir):
    """
    Cleans the output directories by deleting all files.

    Args:
        hpp_output_dir (str): Directory for header files.
        cpp_output_dir (str): Directory for source files.
    """
    # Fully clean the directories.
    for root, dirs, files in os.walk(hpp_output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f'Removed {file_path}')
            except Exception as e:
                print(f'Error removing {file_path}: {e}')
    for root, dirs, files in os.walk(cpp_output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
                print(f'Removed {file_path}')
            except Exception as e:
                print(f'Error removing {file_path}: {e}')
    print('Cleaned directories.')

def chunk_list(lst, n):
    """
    Splits a list into chunks of size n.

    Args:
        lst (list): The list to split.
        n (int): The maximum size of each chunk.

    Returns:
        list: A list of chunks.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_chunk_bindings(hpp_output_dir, cpp_output_dir, filter_class_names, chunk_size=50):
    """
    Generates chunked binding files for the filters.

    Args:
        hpp_output_dir (str): Directory for header files.
        cpp_output_dir (str): Directory for source files.
        filter_class_names (list): List of filter class names.
        chunk_size (int): Number of filters per chunk.
    """
    chunks = list(chunk_list(filter_class_names, chunk_size))
    chunk_headers = []
    for idx, chunk in enumerate(chunks):
        chunk_name = f'filter_bindings_chunk_{idx}'
        chunk_headers.append(chunk_name)
        hpp_content = [
            '#pragma once',
            '#include <pybind11/pybind11.h>',
            '',
            'namespace py = pybind11;',
            '',
            f'void register_{chunk_name}(py::module_ &m);',
            ''
        ]
        cpp_content = [
            f'#include "{chunk_name}.hpp"',
            '// Include filter binding headers for this chunk',
        ]
        for class_name in chunk:
            cpp_content.append(f'#include "{class_name}_bindings.hpp"')
        cpp_content.append('')
        cpp_content.append('namespace py = pybind11;')
        cpp_content.append('')
        cpp_content.append(f'void register_{chunk_name}(py::module_ &m) {{')
        for class_name in chunk:
            cpp_content.append(f'    bind_{class_name}(m);')
        cpp_content.append('}')
        cpp_content.append('')

        # Write the header file
        hpp_filename = os.path.join(hpp_output_dir, f'{chunk_name}.hpp')
        with open(hpp_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(hpp_content))
        print(f'Generated {hpp_filename}')

        # Write the source file
        cpp_filename = os.path.join(cpp_output_dir, f'{chunk_name}.cpp')
        with open(cpp_filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cpp_content))
        print(f'Generated {cpp_filename}')

    return chunk_headers

def generate_pyi_stub(output_dir, filter_class_info):
    """
    Generates a 'filters.pyi' stub file containing type hints and documentation for all filters.

    Args:
        output_dir (str): The directory where 'filters.pyi' will be saved.
        filter_class_info (dict): A dictionary where keys are class names and values
                                  are dictionaries containing 'description' and 'methods'.
    """
    pyi_lines = [
        "# This file is autogenerated. Do not modify manually.",
        "from typing import List, Optional, Any, Tuple",
        "from enum import Enum",
        "",
        "class FilterBase:",
        '    """',
        '    Base class for all filters.',
        '    """',
        "    pass",
        "",
    ]

    # Mapping from C++ types to Python types
    cpp_to_py_type = {
        'int': 'int',
        'double': 'float',
        'float': 'float',
        'bool': 'bool',
        'std::string': 'str',
        'std::vector<int>': 'List[int]',
        'std::vector<uint64_t>': 'List[int]',
        'std::pair<int, int>': 'Tuple[int, int]',
        'uint64_t': 'int',
        'int64_t': 'int',
        'void*': 'Any',
        'std::vector<uint8_t>': 'List[int]',
        'std::map<std::string, std::string>': 'dict[str, str]',
        'const int': 'int',
        'std::vector<double>': 'List[float]',
        'std::vector<std::string>': 'List[str]',  # Added for string vectors
        # Enums are treated as strings
        'std::string': 'str',
    }

    for class_name, class_info in filter_class_info.items():
        description = class_info.get('description', f'{class_name} filter class.')
        methods = class_info.get('methods', [])
        pyi_lines.append(f"class {class_name}(FilterBase):")
        pyi_lines.append('    """')
        pyi_lines.append(f'    {description}')
        pyi_lines.append(f'# For more info on Filters and their respective arguments,
                         please visit https://ffmpeg.org/ffmpeg-filters.html')
        pyi_lines.append('    """')
        if not methods:
            pyi_lines.append("    pass")
            pyi_lines.append("")
            continue
        for method in methods:
            method_name = method['name']
            return_type = method['return_type']
            params = method['params']
            if method['is_setter']:
                # Setters typically return None
                param = params[0]
                cpp_type = param['type']
                if 'Flags' in cpp_type:
                    py_type = 'int'  # Since we're using ints instead of enums
                else:
                    py_type = cpp_to_py_type.get(cpp_type, 'Any')
                pyi_lines.append(f"    def {method_name}(self, value: {py_type}) -> None:")
                pyi_lines.append(f'        """')
                pyi_lines.append(f'        Sets the value for {method_name[3:].lower()}.')
                pyi_lines.append('')
                pyi_lines.append(f'        Args:')
                pyi_lines.append(f'            value ({py_type}): The value to set.')
                if cpp_type == 'std::string' and method.get('possible_values'):
                    possible_vals = method.get('possible_values', [])
                    if possible_vals:
                        values_str = ', '.join([f"{val['name']}={val['value']}" for val in possible_vals])
                        pyi_lines.append(f'            Possible Values: {values_str}')
                pyi_lines.append(f'        """')
                pyi_lines.append('        ...')
            else:
                # Getters return the type
                cpp_type = return_type
                if 'Flags' in cpp_type:
                    py_return_type = 'int'  # Since we're using ints instead of enums
                else:
                    py_return_type = cpp_to_py_type.get(return_type, 'Any')
                pyi_lines.append(f"    def {method_name}(self) -> {py_return_type}:")
                pyi_lines.append(f'        """')
                pyi_lines.append(f'        Gets the value for {method_name[3:].lower()}.')
                pyi_lines.append('')
                pyi_lines.append(f'        Returns:')
                pyi_lines.append(f'            {py_return_type}: The current value.')
                if cpp_type == 'std::string' and method.get('possible_values'):
                    possible_vals = method.get('possible_values', [])
                    if possible_vals:
                        values_str = ', '.join([f"{val['name']}={val['value']}" for val in possible_vals])
                        pyi_lines.append(f'            Possible Values: {values_str}')
                pyi_lines.append(f'        """')
                pyi_lines.append('        ...')
        pyi_lines.append("")

    # Add enum definitions for FilterType
    pyi_lines.append("class FilterType(Enum):")
    for class_name in filter_class_info.keys():
        pyi_lines.append(f"    {class_name} = '{class_name}'")
    pyi_lines.append("")

    # Add factory function with documentation
    pyi_lines.append("def CreateFilter(type: FilterType) -> Optional[FilterBase]:")
    pyi_lines.append('    """')
    pyi_lines.append('    Creates an instance of the specified filter.')
    pyi_lines.append('')
    pyi_lines.append('    Args:')
    pyi_lines.append('        type (FilterType): The type of filter to create.')
    pyi_lines.append('')
    pyi_lines.append('    Returns:')
    pyi_lines.append('        Optional[FilterBase]: An instance of the specified filter, or None if the type is invalid.')
    pyi_lines.append('    """')
    pyi_lines.append('    ...')
    pyi_lines.append("")

    # Write the pyi file
    pyi_filename = os.path.join(output_dir, 'filters.pyi')
    with open(pyi_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pyi_lines))
    print(f'Generated stub file: {pyi_filename}')

def main():
    # Input JSON file location
    input_json_path = r'C:\Users\tjerf\source\repos\CeLux\ffmpeg_filters.json'

    # Check if JSON file exists
    if not os.path.exists(input_json_path):
        print(f"Error: JSON file not found at {input_json_path}")
        return

    # Load the JSON data from the file
    with open(input_json_path, 'r', encoding='utf-8') as f:
        filters = json.load(f)

    # Output directories
    hpp_output_dir = os.path.abspath(r'C:\Users\tjerf\source\repos\CeLux\include\CeLux\filters')
    cpp_output_dir = os.path.abspath(r'C:\Users\tjerf\source\repos\CeLux\src\CeLux\filters')
    dist_output_dir = os.path.abspath(r'C:\Users\tjerf\source\repos\CeLux\celux')
    dist_output_cuda = os.path.abspath(r'C:\Users\tjerf\source\repos\CeLux\celux_cuda')

    # Clean output directories
    clean_dirs(hpp_output_dir, cpp_output_dir)
    # Create the output directories if they don't exist
    os.makedirs(hpp_output_dir, exist_ok=True)
    os.makedirs(cpp_output_dir, exist_ok=True)
    print(f'Include directory: {hpp_output_dir}')
    print(f'Source directory: {cpp_output_dir}')

    # Generate the base class
    generate_filter_base(hpp_output_dir, cpp_output_dir)

    # Keep track of all filter class names and their methods for pyi generation
    filter_class_names = []
    filter_class_info = {}  # Dictionary to store description and methods for each class

    for filter_info in filters:
        # Skip audio filters based on their options' flags
        if is_audio_filter(filter_info):
            print(f"Skipping audio filter: {filter_info.get('filter_name', 'Unnamed Filter')}")
            continue  # Omit this filter

        try:
            header_code, source_code, binding_code, member_variables = generate_cpp_class(filter_info)
        except Exception as e:
            print(f"Error generating class for filter '{filter_info.get('filter_name', 'Unnamed Filter')}': {e}")
            continue

        filter_name = filter_info['filter_name']
        class_name = to_valid_cpp_identifier(capitalize_first_letter(filter_name))
        filter_class_names.append(class_name)

        # Initialize method info list for this class with description
        filter_class_info[class_name] = {
            'description': filter_info.get('description', f'{class_name} filter class.'),
            'methods': []
        }

        # Collect method information from member_variables
        for option_name, var_info in member_variables.items():
            method_name = var_info['method_name']
            cpp_type = var_info['cpp_type']
            readonly = var_info['readonly']

            # Since we're using strings for enum-like options, include possible_values for documentation
            possible_values = var_info.get('possible_values', [])

            # Setter method, skip if read-only
            if not readonly:
                setter_info = {
                    'name': f'set{method_name}',
                    'return_type': 'None',
                    'params': [{'name': 'value', 'type': 'str'}] if cpp_type == 'std::string' else [{'name': 'value', 'type': cpp_type}],
                    'is_setter': True,
                    'possible_values': possible_values
                }
                filter_class_info[class_name]['methods'].append(setter_info)

            # Getter method
            getter_info = {
                'name': f'get{method_name}',
                'return_type': 'str' if cpp_type == 'std::string' else cpp_type,
                'params': [],
                'is_setter': False,
                'possible_values': possible_values
            }
            filter_class_info[class_name]['methods'].append(getter_info)

        # Write to header file
        header_filename = os.path.join(hpp_output_dir, f'{class_name}.hpp')
        with open(header_filename, 'w', encoding='utf-8') as f:
            f.write(header_code)
        print(f'Generated {header_filename}')

        # Write to source file
        source_filename = os.path.join(cpp_output_dir, f'{class_name}.cpp')
        with open(source_filename, 'w', encoding='utf-8') as f:
            f.write(source_code)
        print(f'Generated {source_filename}')

        # Collect binding code
        binding_header_code, binding_source_code = binding_code

        # Write binding files for this filter
        write_binding_files(hpp_output_dir, cpp_output_dir, class_name, binding_header_code, binding_source_code)

    # Generate the FilterFactory with enum and factory function
    if filter_class_names:
        generate_filter_factory(hpp_output_dir, cpp_output_dir, filter_class_names)

    # Generate the chunked bindings
    chunk_headers = generate_chunk_bindings(hpp_output_dir, cpp_output_dir, filter_class_names)

    # Generate the bindings.hpp file
    generate_filter_bindings_hpp(hpp_output_dir)

    # Generate the main bindings.cpp file
    if filter_class_names:
        generate_bindings_cpp(cpp_output_dir, chunk_headers, filter_class_names)

    # Generate the filters.pyi stub file with enhanced documentation
    generate_pyi_stub(dist_output_dir, filter_class_info)
    generate_pyi_stub(dist_output_cuda, filter_class_info)

if __name__ == '__main__':
    main()
