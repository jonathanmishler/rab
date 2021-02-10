from typing import Optional
import re
import numpy as np

""" 
Set of functions to asist in checking the validity of
the Brazilian Tax ID (CNPJ, CPF).
"""


def strip_non_digits(x: str) -> str:
    """ Removes all non digit characters """
    exp = re.compile("[^\d]+")
    return re.sub(exp, "", x)


def check_digits_cnpj(x: str, n: int) -> int:
    """ Finds the valid check digit in the CNPJ """
    check_vec = np.array([6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2])
    digits = np.array(list(x[: -3 + n])).astype("int")
    result = np.dot(check_vec[2 - n :], digits) % 11

    return 0 if result < 2 else 11 - result


def valid_cnpj(x: Optional[str]) -> bool:
    """ Returns if the CNPJ is a valid number within the rules """
    if x is None:
        return False
    x = strip_non_digits(str(x))
    if x == "" or len(x) != 14:
        return False
    check_val = np.array(list(x[-2:])).astype("int")
    return ([check_digits_cnpj(x, n) for n in [1, 2]] == check_val).all()


def check_digits_cpf(x: str, n: int) -> int:
    """ Finds the valid check digit in the CPF """
    check_vec = np.flip(np.arange(2, 10 + n))
    digits = np.array(list(x[: 8 + n])).astype("int")
    result = np.dot(check_vec, digits) % 11

    return 0 if result < 2 else 11 - result


def valid_cpf(x):
    """ Returns if the CPF is a valid number within the rules """
    if x is None:
        return False
    x = strip_non_digits(str(x))
    if x == "" or len(x) != 11:
        return False
    check_val = np.array(list(x[-2:])).astype("int")
    return ([check_digits_cpf(x, n) for n in [1, 2]] == check_val).all()


def format_cnpj(x: str) -> str:
    """ Properly formats the CNPJ as xx.xxx.xxx/xxxx-xx """
    x = strip_non_digits(str(x))
    return f"{x[:2]}.{x[2:5]}.{x[5:8]}/{x[8:12]}-{x[12:14]}"


def format_cpf(x: str) -> str:
    """ Properly formats the CPF as xxx.xxx.xxx-xx """
    x = strip_non_digits(str(x))
    return f"{x[:3]}.{x[3:6]}.{x[6:9]}-{x[9:12]}"
