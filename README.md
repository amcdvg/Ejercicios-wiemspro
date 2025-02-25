# Proyecto: Análisis de Movimiento en Ejercicios mediante Visión por Computadora

El presente proyecto es para determinar la posición de de los puntos de la articulación de los antebrazos, en el presente código mediante los ejes de articulación de los barazos, se determiina el movimento para realizar el ejercicio de biceps. Además, tiene un asistente para contar el ejercicio, entregando ala usuario las metricas de ROM,  Velocidad máxima, velocidad media, tiempo del desarrollo de cada repetición y el porcentaje de trayectoria del ejercicio.


## Modo de ejecución
El presente desarrollo está solo diseñado para funcionamiento local. Por lo tanto se procede a clonar el proyecto
#### Clonar el proyecto

```bash
  git clone https://link-to-project](https://github.com/amcdvg/Ejercicios_wimspro.git
```


## Dependencias 
Una vez que se haya clonado el proyecto en su equipo local, se procederá a instalar las dependencia, por tanto se creará un entorno vistual para ello

#### Entorno virtual 
* Windows 

```bash
  
  python -m venv nombre_del_entorno
  
```
 - Activar del entorno 

    ```bash
    
       nombre_del_entorno\Scripts\activate
    
    ```
 - Desactivar el entorno virtual

    ```bash
    
       deactivate

    
    ```
* IOS

```bash
  
  python3 -m venv nombre_del_entorno

  
```
 - Activar del entorno 

    ```bash
    
       source nombre_del_entorno/bin/activate

    ```
 - Desactivar el entorno virtual

    ```bash
    
       deactivate
    
    ```
#### Instalación de dependencias
Una vez que tenga activado el entorno virtual, se ejecutará el  siguiente comando 

```bash
  pip install -r requirements.txt
```

## ERjecución del script

Desdde una terminal, teniendo el entorno virtual activado, se procederá a ejecutar el siguiente comando

```bash
  python video.py
```
