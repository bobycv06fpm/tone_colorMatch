# The Color Measuring Pipeline

## Tone Overview
Tone is a project that aims to address the challenge of getting an accurate foundation makeup match. The current best method for getting accurately matching foundation makeup is to go to a brick and mortar store and try on different shades. Making matters worse, skin tone and skin needs change throughout the year for many people which means the user will often need to go back to the store if their skin changes. 

Tone works to address the core issues by measuring the users skin tone with a mobile app and match them to the best foundation makeup for their needs.

## Tone Projects
|Repo | |
|---|---|
| [Tone Color Match (This Repo)](https://github.com/dmacewen/tone_colorMatch) | Primary image processing pipeline. Takes a set of images of the users face (taken by the iOS app) and records the processed colors to the database |
| [Tone iOS App](https://github.com/dmacewen/tone_ios) | Guides the user through capturing the images, preprocesses them, and sends them to the server |
| [Tone Server](https://github.com/dmacewen/tone_server) | RESTful API handling user data, authentication, and beta testing information. Receives images from the iOS app and starts color measuring jobs |
| [Tone Spectrum](https://github.com/dmacewen/tone_spectrum) | A deep dive into [metamerism](https://en.wikipedia.org/wiki/Metamerism_(color)) as a potential source of error for Tone in its current form |
| [Tone Database](https://github.com/dmacewen/tone_database) | SQL |
| [Tone SQS](https://github.com/dmacewen/tone_sqs) | Command line utility for sending SQS messages to the Color Match worker. Good for running updated Color Match versions on old captures |


## Tone Post Mortem
Ultimately I could not achieve the level of repeatablility and precision needed to turn Tone into a product. The goal of differentiating between ~40 different skin tones means the difference between two neighboring shades is tiny.

There are a number of potential sources of error in this project:
* Using the front facing camera to try to precisely measure color
    * Cannot recieve RAW images from this camera. There may be some modification happening to the image before we take control of it
* Uncalibrated hardware with unknown charateristics makes precise color measurements near impossible
    * Screens do *not* emit full spectrum light, and two screens emitting the same percieved color(i.e. approximately white light) probably do not emit exactly the same spectrum of light
    * Camera R, G, and B subpixels' spectral senstitivity varys from camera to camera
    * Partial spectrum light can make two different colored surfaces appear the same, i.e. [metamerism](https://en.wikipedia.org/wiki/Metamerism_(color))
    * Without knowing both of these beforehand they can possibly introduce error. Unsure how significanty these effect final output, but they have the potential to have a big infuence
    * See [Tone Spectrum](https://github.com/dmacewen/tone_spectrum) for an investigation into this
* Skin tone information is taken from the face
    * The face is a "noisy" skin tone source as there is often variation from blushing/rosatia/flushed/irritated skin/etc
* It is difficult to approximate the illumination of the phone screen on the face
    * This information would be useful for determining how diffusly reflective the skin is
    * Attempted to estimate in a few ways:
        * Size of the specular reflection in the eye multiplied by the intesity of the light, with the reflection size defined in terms of eye width
            * Eyes are a roughly similarly sized face feature in adults. Ultimately probably not similar enough
            * This approach is trying to estimate illuminance
        * Measure how the luminance of the sclera changes in proportion to the luminance of the skin
            * Error prone and the sclera is often not very white
            * Facial topography and uneven facial lighting from the phone screen makes it hard to estimate skin luminance
        * Both of these were inaccurate and at their best would have only provided information about the luminance around the region of the eye
            * Often the face was under different amounts of illumination depending on the phone screen position compared to the face


## What is Tone Color Match
Tone Color Match is the server side image processing pipeline. The 2 sentence description is: 

* Applying the concept of image diff-ing to a set of ~8 images and white balancing the result to the reflection of the screen flash pulled out of the pupil.
* This approach allows us to ignore ambient lighting.

The basic algorithm in a little more detail is:

1. User captures a set of images of their face under varying, known, lighting intensity
    * This is refered to as screen flash and is implemented approximately as illuminating 100%, 93%, 86%, ..., 50% of the screen
    * For each screen flash, an image is captured with fixed settings
    * The settings must be identical between each image, with the amount of screen illuminated as the only variable changing
2. Facial Landmarks are calculated on each image
    * As implemented, this happens on the iOS App side
    * Facial landmarks are used to divide the face into a set of facial regions: Left Cheek, Right Cheek, Chin, and Forehead
3. The images are aligned with each other
    * Since it is a set of images, people usually have moved a decent amount between the first and last capture
4. Extract the phone screen reflection from each eye
    * Extract lumiance, color, and possibly reflection size information
5. Run linear regression per color channel on each region of the face compared to the extracted screen reflections values
    * The slope of the red channel vs the green channel vs the blue channel regressions should be the skin tone (but not luminance/reflectance)
    * Approximate skin reflectance by approximating the skin luminance
       * Measure how the luminance of the sclera changes in proportion to the luminance of the skin
       * Size of the specular reflection in the eye multiplied by the intesity of the reflection, with size defined in terms of eye width
       * Shortcomings of each of these approaches is touched on in the Post Mortem
6. Record RGB and Luminance

Note: The results will only be comparable other values captured with this approach

 
## Details

* `runSteps.py` runs the image processing pipeline
* `getFaceColor.py` and `application.py` are both entry points into the application
    * `getFaceColor.py` is intended to be used from the command line
    * `application.py` is intended to be used by AWS Elastic Beanstalk Worker environemt
