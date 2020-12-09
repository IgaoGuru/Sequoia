![Sample Gif](/readmes/sequoia_samplebanner.gif)

# Csgo Data-Collector 
[![C++](https://img.shields.io/badge/language-python-blue)](https://www.python.org/) 
[![CS:GO](https://img.shields.io/badge/game-CS%3AGO-yellow.svg?style=plastic)](https://store.steampowered.com/app/730/CounterStrike_Global_Offensive/) 
[![Windows](https://img.shields.io/badge/platform-Windows-0078d7.svg?style=plastic)](https://en.wikipedia.org/wiki/Microsoft_Windows) 
[![x86](https://img.shields.io/badge/arch-x86-red.svg?style=plastic)](https://en.wikipedia.org/wiki/X86) 
[![License](https://img.shields.io/badge/license-GPL--3.0-brightgreen)](LICENSE)
<br>![Windows](https://github.com/danielkrupinski/Osiris/workflows/Windows/badge.svg?branch=master&event=push)

A neural network for CounterStrike:GlobalOffensive character detection and classification. Built on a custom-made dataset, powered by the [csgo-data-collector](https://github.com/IgaoGuru/csgo-data) software.
The project incorporates a fine-tuned version of [Ultralitic's YOLOv5](https://github.com/ultralytics/yolov5), while also using a secondary helper-NN for aiding in the classification task.  
After inference, the bounding boxes are processed by the **yolo_inference.py** file, and the mouse movement is calculated based on the distance from the crosshair (aim) to the enemy's location.

**DISCLAIMER: This software was developed with the intention of only being used in private matches, as a proof of concept, causing no harm to other player's experiences. If you wish to cheat in online matches, be aware that there are many other much more practical and efficient ways of doing so.**

## Prerequisites
* [Python 8.x](https://python.org) additional libraries:

    <details>
    <summary> list of required libraries:</summary>

    ahk>=0.11.1
    Cython
    matplotlib>=3.2.2
    mss>=6.1.0
    numpy>=1.18.5
    opencv-python>=4.1.2
    pandas>=1.1.5
    pillow
    PyYAML>=5.3
    scipy>=1.4.1
    tensorboard>=2.2
    torch>=1.6.0
    torchvision>=0.7.0
    tqdm>=4.41.0

    </details>

    After installing python, you can install all the necessary libraries with the following command:
        ```
        pip install -r requirements.txt
        ```

* **strongly recommended**: [CUDA Toolkit 10.2](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal):
    CUDA is Nvidia's framework for high performance GPU computing. This software uses it to run the AI inside the GPU, instead of relying on the CPU, which is much slower. This means **you need an Nvidia GPU** to get the best possible performance out of the software.

## Test

You can download the already compiled dll, and inject it into the game either with [Extreme injector](https://github.com/master131/ExtremeInjector/releases/tag/v3.7.3) (recommended), or [Xenos Injector](https://github.com/DarthTon/Xenos/releases/tag/2.3.2).

### Setting up the game

Remember to configurate steam to start CS:GO in [insecure mode](https://csgg.in/csgo-guide-to-launch-options/) (with the "-untrusted") flag), and run the game. This flag will ensure that Valve's anticheat isn't activated when you inject the dll (eliminating the risk of banning you steam account from cs:go's servers).

In csgo's video setting, remember to change the in-game resolution to your current [img_rez]() (default=`[1280x720]`).
**place the csgo window on the top-left corner of you primary monitor**

For data-collection results, it's best to run "deathmatch" private matches, that is because there are no interruptions (like round intervals or timeouts) during the game, and you can switch freely between teams.

<details>
<summary> Recommended CS:GO deathmatch settings:</summary>
<br>

After starting the private match, make sure your [Developer Console](https://gamepros.gg/csgo/articles/how-to-open-the-console-csgo-enable-and-use-developer-console) is activated in CS:GO's settings. After that, I recommend setting the following commands:
```
sv_cheats 1
bind t noclip
bind y god
mp_dm_bonus_length_max 0
mp_dm_time_between_bonus_max 9999
cl_teamid_overhead_mode 0
mp_roundtime 60
mp_restartgame 1 60
god
```
with these commands, you can use <kbd>t</kbd> to fly through te map, making it easier to spot other players, and use <kbd>y</kbd> to make the LocalPlayer(you) immortal. 
</details>
.

> note: because of a minor difference between the capturing of the screen and the bounding box output, moving the mouse too fast may cause inaccuracies in the data collection. Try to move the mouse at a steady rate, with no fast or abrupt movements.

### Running the Dll

Inside Extreme Injector's configuration menu, change the Injection Method from "Standard Injection" to "Manual Map".

Before injecting, create a folder named named `"csgolog"` inside `"C:\"`, so that it's path is `C:\csgolog\`.

Select the game's process in the injector, and select the dll in `"ADD DLL"` option. You can now inject the dll by clicking `Inject`. 

After starting a private match with bots, open the menu with <kbd>INSERT</kbd>, and click on the `ESP` option.

In the ESP menu, you can enable either enemy and ally bounding boxes (or both at the same time) with the "Box" checkbox.
> By default, the bounding boxes are not rendered into the game (so you won't be able to see them while playing). Later on, an option for toggling bbox rendering will be added.

<img src="readmeimages/csgodatareadme2.png" alt="drawing" width="800"/>

Enabling the "Box" option will automatically start outputing bboxes for the specific target. (in this case "Visible Enemies");

After enabling the preffered bounding boxes, a text file will be created in the **stardard path** `(C:/csgolog/csgo_log.txt)`. You don't need to edit or open the file. This file will be read and processed by the `main_cycle.py` script.

> note: in order to modify the standard csv path (not recommended), you will need to edit/compile the dll's code from source
<details>
<summary> If you want to modify the standard csv path:</summary>
<br>

After opening the dll's code in VisualStudio, head over to the `StreamProofEsp.cpp` file under the `Hacks` folder. In there, you should find a `PlayerAnnotate` function, and there you can modify the "myfile.open('your_path_here')" path.
<img src="readmeimages/csgodatareadme3.png" alt="drawing" width="800"/>
</details>


### Collecting images and outputs

Now that the preferred bboxes are being outputed, its time to collect the actual images that correspond to the outputed bboxes. For collecting the images and pairing them with the dll's output, we will use the **main_cycle.py** script.

Open the **main_cycle.py** script with you editor of choice, and change the "output_path" variable to your directory of choice. This path will contain the outputs from the data-collector.

With the dll running, run the **main_cycle.py** script. A folder will be created every time the script is ran. Inside, there will be the annotation from the screencaptures inside the "imgs" folder. This annotation file is the final output of the data-collector. 

## Important variables and files

#### main_cycle.py 
This file manages the screencapturing, saving, merging and outputing of the data-collector. 
All functions used by this file come from the **csgodata** module (inside the **utils** folder).

Every run of this file generates a new [session folder](####sessionfolder)

#### Session folder
A session is a complete run of the main_cycle.py script. This folder contains the outputs from main_cycle.py.
The folder contains:
* imgs folder (contains the screenshots)
* [annotation file](#annotationfile) (contains the formatted bbox annotations)

#### Annotation file
The annotation file contains the following format:
* corresponding image's name
* LocalPlayer's team (2 = Terrorist, 3 = CounterTerrorist)
* bboxes's enemy state (if bbox is from the opposite team from LocalPlayer = 1; else = 0)
* x0, y0, x1, y1

## Acknowledgments

* [Daniel Krupiński](https://github.com/danielkrupinski) for developing and maintaining the open-source original software.
* [ocornut](https://github.com/ocornut) and [contributors](https://github.com/ocornut/imgui/graphs/contributors) for creating and maintaining an amazing GUI library - [Dear imgui](https://github.com/ocornut/imgui).
* [Zer0Mem0ry](https://github.com/Zer0Mem0ry) - for great tutorials on reverse engineering and game hacking

## License

> Copyright (c) 2020-2020 Igor Rocha

This project is licensed under the [MIT License](https://opensource.org/licenses/mit-license.php) - see the [LICENSE](https://github.com/danielkrupinski/Osiris/blob/master/LICENSE) file for details.
