from configparser import ConfigParser


def config(filename='step_1_setup_data/database.ini', section='postgresql') -> dict:
    parser = ConfigParser()
    parser.read(filename)           # use external database.ini file to secure access data

    # get db parameters from postgresql section
    db: dict = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


# database.ini needs to look something like this:
# [postgresql]
# host=localhost
# database=mimic
# user=postgres
# password=xxxxxxxxxxxxxx

# Reference: https://www.postgresqltutorial.com/postgresql-python/connect/
