import py7zr
import multiprocessing
import os

import subprocess

import os
import subprocess

def compress_and_split(input_file: str, output_prefix: str, output_folder: str, volume_size_mb: int = 99):
    """
    Comprime un archivo CSV en volúmenes de tamaño especificado.

    Args:
        input_file (str): Ruta al archivo CSV de entrada.
        output_prefix (str): Prefijo para los archivos comprimidos de salida.
        output_folder (str): Carpeta de destino para los archivos comprimidos y divididos.
        volume_size_mb (int): Tamaño máximo de cada volumen en MB.
    """
    volume_size = volume_size_mb * 1024 * 1024  # Convertir MB a bytes

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Especificar el número de hilos a utilizar (70% de los hilos disponibles)
    num_cpus = os.cpu_count()
    num_threads = max(1, int(0.7 * num_cpus))

    # Dividir el archivo 7z en volúmenes del tamaño especificado utilizando 7z
    subprocess.run(['7z', 'a', '-mmt'+str(num_threads), '-v'+str(volume_size), os.path.join(output_folder, f'{output_prefix}.7z'), '-v'+str(volume_size), input_file])

    print(f'{input_file} ha sido comprimido en volúmenes de {volume_size_mb} MB utilizando {num_threads} hilos.')


import os
import py7zr

def decompress_volumes(prefix: str, output_dir: str, folder: str):
    """
    Descomprime archivos divididos en volúmenes.

    Args:
        prefix (str): Prefijo de los archivos divididos.
        output_dir (str): Ruta al directorio donde se descomprimirán los archivos.
        folder (str): Carpeta donde buscar los archivos de volúmenes.
    """
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listar todos los volúmenes en orden
    volumes = [f'{folder}/{f}' for f in os.listdir(folder) if f.startswith(prefix + ".7z.")]

    # Verificar que hay al menos un volumen
    if not volumes:
        raise FileNotFoundError("No se encontraron volúmenes para descomprimir.")

    # Combina los volúmenes en un solo archivo
    combined_file = os.path.join(output_dir, f'{prefix}.7z')
    with open(combined_file, 'ab') as outfile:
        for volume in volumes:
            with open(volume, 'rb') as infile:
                outfile.write(infile.read())

    # Descomprime el archivo combinado
    with py7zr.SevenZipFile(combined_file, mode='r') as archive:
        archive.extractall(path=output_dir)

    # Elimina el archivo combinado después de la extracción
    os.unlink(combined_file)

    print(f'Archivos descomprimidos en {output_dir}')
