# The Color Measuring Pipeline

## Tone Overview
Tone is a project that aims to address the challenge of getting an accurate foundation makeup match. The current best method for getting accurately matching foundation makeup is to go to a brick and mortar store and try on different shades. Making matters worse, skin tone and needs change through out the year for many people, which means the user often will need to go back to the store if their skin changes. 

Tone works to address the core issues by measuring the users skin tone with a mobile app and matches them to the best foundation makeup for their needs.

## Tone Projects
|Repo | |
|---|---|
| [Tone Color Match (This Repo)](https://github.com/dmacewen/tone_colorMatch) | Primary image processing pipeline. Takes a set of images of the users face (taken by the iOS app) and records the colors to the database |
| [Tone iOS App](https://github.com/dmacewen/tone_ios) | Guides the user through capturing the images, preprocesses, and sends to server |
| [Tone Server](https://github.com/dmacewen/tone_server) | RESTful API handling user data, authentication, and beta testing information. Receives images from iOS app and starts color measuring jobs |
| [Tone Spectrum](https://github.com/dmacewen/tone_spectrum) | A deep dive into [metamerism](https://en.wikipedia.org/wiki/Metamerism_(color)) as a potential source of error for Tone in its current form |
| [Tone Database](https://github.com/dmacewen/tone_database) | SQL |
| [Tone SQS](https://github.com/dmacewen/tone_sqs) | Command line utility for sending SQS messages to the Color Match worker. Good for running updated Color Match on old captures |


## Tone Post Mortem
Ultimately I could not achieve the level of repeatablility and precision needed to turn Tone into a product. The goal of differentiating between ~40 different skin tones means the difference between two neighboring shades is tiny.

There are a number of potential sources of error in this project:
* Using the front facing camera to try to precisely measure color
    * Cannot recieve RAW images from this camera. There may be some modification happening to the image before we take control of it
* Uncalibrated hardware with unknown charateristics makes precise color measurements near impossible
    * Screens do *not* emit full spectrum light, and two screens emitting the same color(i.e. approximately white light) probably do not emit the same spectrum of light
    * Camera R, G, and B subpixels spectral senstitivity varys from camera to camera
    * Partial spectrum light can make two difference color surfaces appear the same, i.e. [metamerism](https://en.wikipedia.org/wiki/Metamerism_(color))
    * Without knowing both of these beforehand they can possibly introduce error. Unsure how significanty these effect final output, but they have the potential to have a big infuence
    * See [Tone Spectrum](https://github.com/dmacewen/tone_spectrum) for an investigation into this
* Primary color information was taken from the face
    * The face is a "noisy" skin tone source as there is often variation from blushing/rosatia/flushed/irritated/etc
* It is difficult to approximate the amount of light shining on different parts of the face
    * This information would be useful for determining how reflective the skin is
    * Attempted to estimate in a few ways
        * Size of the specular reflection in the eye multiplied by the intesity of the light, with size defined in terms of eye width
            * Eyes are roughly similarly sized face feature in adults. Ultimately probably not similar enough
            * trying to estimate illuminance
        * Measure how the luminance of the sclera changes in proportion to the luminance of the skin
            * Error prone and the sclera is often not very white
        * Both of these were inaccurate and at their best would have only provided information about the luminance at the eye
            * Often the face was under different amounts of illumination depending on the phone screen position compared to the face


## What is Tone Color Match
Tone Color Match is the server side image processing pipeline. The one sentence description is: 

* Applying the concept of image diff-ing to a set of ~8 images and white balancing the result to the reflection of the screen flash pulled out of the pupil.

The basic algorithm in a little more detail is:

1. User captures a set of images of their face under varying, known, lighting intensity
    * This is refered to as screen flash and is implemented approximately as illuminating 100%, 93%, 86%, ..., 50% of the screen
    * For each screen flash, an image is captured with fixes settings
    * The settings between each image must be identical, with the only variable changing being the amount of the screen illuminated
2. Facial Landmarks are calculated on each image
    * As implemented, this happens on the iOS App side
    * Faceial landmarks are used to divide the face into a set of facial regions: Left Cheek, Right Cheek, Chin, and Forehead
3. The images are aligned with each other
    * Since it is a set of images, people usually have moved a decent amount between the first and last capture
4. Extract the phone screen reflection from each eye
    * Extract lumiance, color, and possibly reflection size information
5. Run linear regression per color channel on each region of the face compared to the extracted screen reflections
    * The slope of the red channel vs the green channel vs the blue channel regressions should be the skin tone (but not luminance/reflectance)
    * Approximate skin reflectance by approximating the skin luminance
       * Measure how the luminance of the sclera changes in proportion to the luminance of the skin
       * Size of the specular reflection in the eye multiplied by the intesity of the light, with size defined in terms of eye width
       * Shortcomings of each of these approaches is touched on in the Post Mortem
6. Record RGB and Luminance

Note: The results will only be comparable other values captured with this approach

 
## Details

* `runSteps.py` runs the image processing pipeline
* `getFaceColor.py` and `application.py` are both entry points into the application
    * `getFaceColor.py` is intended to be used from the command line
    * `application.py` is intended to be used by AWS Elastic Beanstalk Worker environemt
