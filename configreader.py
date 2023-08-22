import configparser

def get_variable_from_ini(variable_label):
    config = configparser.ConfigParser()
    config.read('config.ini')

    settings_section = config['Settings']

    if settings_section.get(variable_label) == "cpu or cuda:0":
        return "cpu"
    return settings_section.get(variable_label)