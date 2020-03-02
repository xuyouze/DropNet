# coding:utf-8
# @Time         : 2019/10/27 
# @Author       : xuyouze
# @File Name    : logger_config.py

import logging.config
import re


class ColorFilter(logging.Filter):
    def filter(self, record):
        color_dict = {
            "NORMAL": '\x1b[0m',  # normal
            "DEBUG": '\x1b[35m',  # pink
            "INFO": '\x1b[32m',  # green
            "WARNING": '\x1b[33m',  # yellow
            "ERROR": '\x1b[31m',  # red
            "CRITICAL": '\x1b[31m'  # red
        }
        record.msg = color_dict[record.levelname] + str(record.msg) + color_dict["NORMAL"]  # normal
        return record


class FileFilter(logging.Filter):
    def filter(self, record):
        record.msg = re.findall(r"\x1b\[[0-9]{1,2}m(.*)\x1b\[0m", record.msg)[0]
        return record


config = {
    'version': 1,

    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'filters': {
        'ColorFilterStreamHandler': {
            '()': ColorFilter,
        },
        'FileFilter': {
            '()': FileFilter,
        }
    },

    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            # "stream": "ext://sys.stdout",
            "filters": ['ColorFilterStreamHandler']
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logging.log',
            'level': 'INFO',
            'formatter': 'simple',
            "filters": ['FileFilter']

        },
    },
    'loggers': {
        'TrainLogger': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        'TestLogger': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
    }
}
