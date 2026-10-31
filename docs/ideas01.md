# Brainstorm Hand Tracking Demo
*Exported (generic) — https://chatgpt.com/c/6ae5116f-0fbc-83ea-b1f9-844bde88bbd4*

---

Today 5:51 PM

I want to use Google Mediapipe hand tracking to create a fun and interactive demo.

-   Can I use mediapipe in a Macbook pro laptop or do I have to use other OS like ubuntu?
-   How much storage will Mediapipe take? Are there options to make it lighter?
-   I'm thinking about starting with a simple python app and from there once the idea, design, and prototype are complete, maybe try to make a web page capturing the implementation. Would that be possible and practical?
-   Previous ideas fall in the category using hands, probably finger separation to quickly draw geometrical forms like circles or squares with random coloring. Add some animation so this figures may move like if they were falling. How far can we take this idea?
-   I'm also thinking use hand tracking and possibly gestures to produce sound or even music, try to propose and idea around that.

Remember that this is a demo that are supposed to make potential students interested in studying data science and machine learning.

Let's the brainstorm begin

…

Worked for 57s

I’ll check the current MediaPipe support for macOS and web deployment, then build out a few demo concepts that are both fun to try and useful for introducing students to machine learning.

![](https://www.google.com/s2/favicons?domain=developers.google.com&sz=128)![](https://www.google.com/s2/favicons?domain=pypi.org&sz=128)

Searched 5 websites

I think we can turn this into something considerably more interesting than a simple hand-tracking demonstration.

Imagine a prospective student standing in front of a laptop, without touching the keyboard or mouse. They move their fingers, create a glowing circle in midair, release it, and watch it fall and bounce. They then use the other hand to interact with that object. A moment later, they switch to music mode, where their hands become a virtual musical instrument.

The interesting part is that the same machine learning model can power all these experiences. We only change how we interpret its predictions.

I would design this as an interactive showcase called AI Playground: Your Hands Are the Controller.

# 1\. Technical feasibility

![Apple MacBook Pro 14 With M2 Pro Review: Fast And Efficient | HotHardware](https://images.openai.com/static-rsc-4/SfQ1OjxlVX_7n8wzRhPKnfyY9bwtTZOSwgajPE_qJP4QT38RyhRvFY9EJPKuRRamRqxAYF86dRWbkjV20zzOCbZ94PXbHTXyPjRXKF8L9JFdExzj30rMIGgQBs0G4ETvnQAcuRTxjbCu8gTl8Tz5qfiRnMDRaIshcJRqOrAYGZs?purpose=inline)

Can you use your MacBook Pro? Absolutely.

MediaPipe officially supports macOS, including Apple Silicon. Your MacBook Pro is a suitable development platform. You do not need Ubuntu, an external GPU, or an additional camera for the initial prototype.

![](https://www.google.com/s2/favicons?domain=https://developers.google.com&sz=32)

Google for Developers+1

![Visualization of Mediapipe hand landmark locations based on hand... | Download Scientific Diagram](https://images.openai.com/static-rsc-4/x16BN7EsnqkLNLakEih6HLGzNVZ9gd445ZQo2EEbumwU6A1-W2ZvD2q9-FgBApLNyhuM3zJzy08vNTYi55SrArz9dSokokK1PuELcasVilOPMhEkTs78Sl-5aFnyZtBbAI9GibBKC8-LImuw5gwgGTXrLU3U4x7gWxP19Icv_KY?purpose=inline)

What does MediaPipe actually provide?

Its Hand Landmarker estimates 21 landmarks for each detected hand, including coordinates for fingers, knuckles, and wrists. It supports multiple hands and real-time video.

![](https://www.google.com/s2/favicons?domain=https://developers.google.com&sz=32)

Google for Developers+1

## Storage and lightweight alternatives

As of October 27, 2026, the MediaPipe 1.2.0 distribution for Apple Silicon has a compressed wheel size of approximately 39 MB. That is the MediaPipe package, not the complete Python environment.

![](https://www.google.com/s2/favicons?domain=https://pypi.org&sz=32)

PyPI

| 
Component

 | 

Approximate storage

 |
| --- | --- |
| 

MediaPipe wheel download

 | 

39 MB

 |
| 

Hand tracking model

 | 

A few to tens of MB, depending on model

 |
| 

Python environment with dependencies

 | 

Budget 200–500 MB initially

 |
| 

Application code

 | 

Usually negligible

 |
| 

Web deployment

 | 

Much smaller client-side requirements, depending on assets

 |

The full installation estimate is a planning allowance, not a measured installation size.

To keep the prototype lightweight, I would use `uv`, MediaPipe, NumPy, and OpenCV, with `pygame-ce` added when we introduce the graphics and physics.

Bash

```
uv init ai-playground
cd ai-playground
uv add mediapipe opencv-python numpy pygame-ce
```

We would also download the Hand Landmarker `.task` model, which is loaded separately.

For the web version, the browser can execute MediaPipe through its JavaScript library. That means no Python installation for visitors, and potentially no backend server at all.

## Python first, web later?

Yes, and I think that's a very sensible development strategy.

Webcam video frames

MediaPipe hand landmark detection

21 coordinate points per hand

Gesture interpretation

Pinch · Release · Swipe · Finger spacing · Velocity

Visual engine

Drawing, physics, particles

Audio engine

Notes, synthesis, MIDI

The important design decision is to build a gesture interpretation layer independent of MediaPipe and the rendering engine.

For example, both a Python and JavaScript implementation should be able to translate the landmark stream into semantic events such as:

Python

{

"gesture": "pinch",

"position": \[0.45, 0.32\],

"strength": 0.83,

"velocity": \[0.02, \-0.01\]

}

Then the graphics or audio engine decides what a pinch should do.

We won't automatically convert the entire Python application to JavaScript, but we can reuse the design, gesture definitions, mathematical formulas, and test cases.

Google provides official implementations for both [Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python?utm_source=chatgpt.com) and [JavaScript](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/web_js?utm_source=chatgpt.com).

# 2\. Concept A: Magic Geometry Lab

Interactive drawing · Physics · Generative art

This would be the first application I'd build. It's visually appealing, intuitive, and relatively simple to prototype.

![Boolean Planet — Future Wife](https://images.openai.com/static-rsc-4/OxlrM67ANtvYDwWrQ-bSb9xRNk3I_oGDH-wjUP7eNpDn4e8zzBnb0W-IFK6ywLr5ljyICW2a9XF7acWPttXHauVFQcabVzAULKPM7VMtQkxbFnxQ2lL9NhNyZG6F3Xe4rzfnmWNzgPrZgXhOLeQBLLsybQfPcn--ltKUdkC0bRs?purpose=inline)

![Floating Geometries + p5.js](https://images.openai.com/static-rsc-4/6JiAwzaybLRcLfwMhAXsfKsg317j33HT-GpQfQitMsE0LGrqdR2xzwwRgCL0pHK0hhWZsxA4sVDK1UPa87OYbyKyPptF5kOfMsCmtX_fwEhGwAQHBlnEeaEnt3V12Mm4RilLRiHCPMfbs6ezmawRwSAbdR348VC7OS4_pDt6s-w?purpose=inline)

![Notch | Virtual Production](https://images.openai.com/static-rsc-4/vAakfonYKxuC3tjApZpLzEywmXDRF4NlsgfopFG0tewyPgjuwOPheKVH5FbdH6qB4oY3J26X0K9OlBLfZ4ZGBMheBPl4R-riX5iQsFWWAYGCDWUv_zStmp44iclDZog7XlZZ03GC7ZKh4ly7G64h2g4qsqQXbTTLzTqcaxJp3KU?purpose=inline)

## The core interaction

Picture a dark screen showing a live mirrored view of the user, perhaps with a subtle overlay of tracked hand landmarks.

The user can create, manipulate, and release objects in a virtual world.

| 
Gesture

 | 

Action

 | 

Visual effect

 |
| --- | --- | --- |
| 

Pinch thumb and index

 | 

Start drawing a circle

 | 

Luminous circle grows as fingers separate

 |
| 

Pinch thumb and middle finger

 | 

Draw a square

 | 

Square scales dynamically

 |
| 

Release pinch

 | 

Release object

 | 

Gravity takes over

 |
| 

Move hand horizontally

 | 

Push nearby objects

 | 

Objects scatter

 |
| 

Close fist

 | 

Grab an object

 | 

Object follows the hand

 |
| 

Open palm

 | 

Generate a burst

 | 

Colored particles explode outward

 |
| 

Two hands moving apart

 | 

Stretch selected object

 | 

Shape changes size

 |

These are proposed interactions, not gestures MediaPipe automatically recognizes. Some require our own state machine or gesture classifier.

## How far can we take it?

I see three levels of increasing sophistication.

Level 1 — Magic Drawing

Create colorful objects, adjust their dimensions, and release them. Gravity, collisions, and random colors produce something immediately engaging.

Level 2 — Interactive Physics

Objects bounce, rotate, collide, and respond to hand movements. Two hands can catch, stretch, or throw objects. Add gravitational fields that users can move around.

Level 3 — Generative Art

Introduce particle trails, glowing effects, shape transformations, and autonomous behaviors. A circle might split into smaller circles, while a square could become a collection of particles when it hits the ground.

## A little interactive concept prototype

Here's a simplified illustration of the kind of falling-geometry environment we could create. This is a 2D physics sketch, not yet connected to the webcam or MediaPipe.

Tuning the inputs

For the Python version, `pygame-ce` could handle initial animations and falling objects. Later, a physics library such as Pymunk could provide more realistic collisions.

# 3\. Concept B: AirSynth — Music from Hand Gestures

Hand tracking · Interactive sound synthesis · Musical interaction

This may be even more impressive to prospective students because it creates a direct relationship between body movement and sound.

![Crie seu próprio Teremim Virtual (WebCam + We... | Tá Pronto](https://images.openai.com/static-rsc-4/EmQ0AxDxKHkCBCBdIxfaSqEMHWCXG1IZXYtCP0-6QVTlUZsZylfu2uvIfRGG8A-gq5yRToygDkrZD7weHC-GosBN46VPHWxnU-oiDZmWA0s7wIV3HogtMU6TCghLs_7cROiMEELBv1lIrcNjMQ857IlJaAKOuRL3OR3RJJ8Mots?purpose=inline)

![Gesture Synth: Play Music With Hand Gestures — Free Online Tool](https://images.openai.com/static-rsc-4/DvKJDdn23JTzRvRS91oz24v28xHY7qEC4EwBhIgTJgSymJHL8v0JYMhZmGq-3wCRmR6_c0NiUrp5n0aCKoRuAkmhZui_lyEPIWSVqp_5a418zXZnFc2dku5QTuahO24gcEYDEwufVBKBEiWgVAjoMgqU4xaizwzjfEhQgsgud34?purpose=inline)

![ÆTHERWAVES](https://images.openai.com/static-rsc-4/XAg_ub2eDEFhEa3fClK9OJc8sTwbNwK6vKz3v2CwH1AwmIaIyybBWImhqAk3AOPdbW1-XKuGUslvrBGXYSlK_L-qbCvsv8fwk16nS48oLoNYH9wpYaZArJq17wG-8hicBb6ywEVO6QOl8Lka_nJn1dw1qy-FNB_3kEkWN4NyDY8?purpose=inline)

## The basic idea

Turn your hands into a musical instrument.

The first version would be inspired by a theremin, but with more intuitive controls.

One hand controls musical notes and the other controls how those notes sound.

| 
Gesture or movement

 | 

Musical parameter

 |
| --- | --- |
| 

Right hand moves vertically

 | 

Pitch

 |
| 

Right hand moves horizontally

 | 

Move between notes or scales

 |
| 

Left thumb/index separation

 | 

Volume

 |
| 

Left palm rotation

 | 

Filter cutoff

 |
| 

Quick pinch

 | 

Trigger a note

 |
| 

Closed fist

 | 

Mute or stop notes

 |
| 

Both hands moving outward

 | 

Increase reverb or atmospheric effect

 |

We could map the vertical position to a musical scale rather than directly to arbitrary frequencies. That would make it easy for an inexperienced visitor to create pleasant melodies.

For example, select a pentatonic scale, in which improvising can be particularly forgiving.

## An alternative: The Invisible Piano

Instead of mapping coordinates continuously, imagine five vertical columns in space, each corresponding to a musical note.

Illustrative five-note controller

Selected note: C

Prototype control mapping

A visitor plays notes by moving their fingertip into different regions and pinching to trigger them.

With two hands, they could play a melody and adjust sound effects simultaneously.

## What about more sophisticated musical interactions?

Here are some possibilities.

![Finger Drum Pro | Learn Finger Drumming Online](https://images.openai.com/static-rsc-4/CSXFqdljTvoWiiB1xrl33TIFbYuRmq2NLbpJ3l7VbcBlYU4ArLW1Z8N5p8JeiBTpz8jgfBIEWPTuUlRwU-C9FWpaE3w5P-10WohhuGGtfQykSs7ugcATgk6QIeojKdH4jhAgUx1dAqiUyHnm5MTmTWjaHh46Oew2AA-9CYD0_qk?purpose=inline)

Air Drums

Strike invisible drums using index finger movement. Detect the speed and direction of a strike to control sound intensity. We could even create a virtual drum kit with kick, snare, hi-hat, and cymbals.

![Stable Audio 2.5 | Inpaint | AI Model | Eachlabs](https://images.openai.com/static-rsc-4/EPi8F5ZwzKSC72g0b6X5dto08F5x0es6wXwR62oywhUdPHRR2KkiC-apkBuyQsnrd0tWHm7jRs1JGsHpdLi5x-jWzrSeTRP5ZIxM6fUticuYmYRwuybD7-hkGlmFSbPzTzBJNHrF3jnyW9uP-61OyAJgmfUap0ehQtXrLT59q_M?purpose=inline)

Gesture Synthesizer

Hand distance controls harmonics, rotation changes filters, and waving motions modulate sound. Visitors discover how changing numerical parameters creates different sounds.

![Expressive MIDI controller gives music a visual and artistic flair - Yanko Design](https://images.openai.com/static-rsc-4/HJjc-DHHmeJWJ13d6IQrRHE2RFP9G-QxtV7w-VTTxuhINgVmbB5mNd7F5robhIFWleOrKtLUw_l3T4R2-yxVKwjKN8hq-9nM8tm0DvZL5Pat9KrcNmMH9ve0bgDoZK7zc-hMVI0wAcnj6GMzo0IrHWpN1AGNJM3iVgIZqWZ1_m4?purpose=inline)

Gesture Beat Composer

Draw shapes in the air to generate musical patterns. A circle might generate a bass sequence, a square a rhythmic beat, and a triangle an arpeggio.

For the Python prototype, we have several implementation paths:

-   `sounddevice` plus NumPy for generating audio directly.
    
-   `pygame.mixer` for simple sound effects and samples.
    
-   `python-rtmidi` for sending MIDI to external synthesizers, including virtual instruments.
    
-   Web Audio API or Tone.js for a browser-based instrument.
    

One important technical consideration is latency. Musical interaction is more sensitive to delays than drawing. We should avoid unnecessary audio buffering, smooth tracking noise without introducing excessive delay, and generate sound through an independent audio engine rather than computing audio inside the video-processing loop.

# 4\. Concept C: AI Physics Orchestra

This is where I would combine both of your ideas into one application.

![Bilişsel Senfoni: 2026 Yapay Zeka Ekosisteminde "Rüya Takımı" ile Operasyonel Verimlilik - Evrim Ağacı](https://images.openai.com/static-rsc-4/r7zswgKCNMwcvaaJdLBUfyZSHoj4Hahdxwv30Gm-7Lnr04FWE4URAmyV3q4btmRq5JavPZ214gtntfOQAR4ylSTFRnAID_ObNPs_I1lXdlaeW6z8sGkL8wfrbnu2EUs2RioYlsNflBmDlSg8xdL8f7GeX4IK4Vq5ZrDZd9RGN-w?purpose=inline)

[evrimagaci.org](https://evrimagaci.org/blog/bilissel-senfoni-2026-yapay-zeka-ekosisteminde-ruya-takimi-ile-operasyonel-verimlilik-22529?utm_source=chatgpt.com)

Imagine that every shape the visitor creates also produces music.

A circle generates a soft synthesizer tone when it bounces. A square produces a percussion sound. A triangle produces a bell-like sound. The size of the object determines pitch, and the impact velocity determines volume.

Now add a few mechanics:

-   Drawing a new shape introduces another musical voice.
    
-   Throwing a shape harder produces louder or brighter sounds.
    
-   Colliding shapes can trigger chords.
    
-   Moving a hand through the scene creates wind that alters their trajectories.
    
-   Creating multiple objects results in an evolving musical composition.
    

You effectively get an interactive physics-based musical instrument powered by machine learning.

This has a particularly interesting educational advantage: visitors can see how one numerical quantity, such as object velocity, can influence another quantity, such as sound amplitude.

# 5\. Where is the data science and machine learning?

This is perhaps the most important consideration for a recruitment demonstration.

MediaPipe itself already contains pretrained machine learning models. If we merely apply it, we are showcasing an application of ML, not necessarily teaching students how machine learning works.

I would deliberately expose a little of the underlying intelligence.

## Show the model's predictions

We can provide a toggle that changes between two views.

![MediaPipe Hands: 21 landmarks [13]. | Download Scientific Diagram](https://images.openai.com/static-rsc-4/lj0c_0EFX-Y71Aykm-ybfjGSqmHdz6GYxFRbxBfxRWgYkfT9g9HEaGgEZzVwaxzw1mYma31PYS_NjzYIJnN1KuYNF602yCJZPOA9tCEbQ65YWbk0sHKTuk723eRIwAxmRaxnYPk0ul7q2aSQ3JA0ZhcwXHLGO_Qzsd7gUAZ5i80?purpose=inline)

AI Vision Mode

See the landmarks, coordinates, and interpreted gestures.

![Circle bokeh effect light lights | Premium Photo  - rawpixel](https://images.openai.com/static-rsc-4/uIXztDn3qbC8Q_6iyOcdDTLTGQdT8QMUDh1IaF-SWWMJsLmUzDnYirSK5ke5hvFDfVB-8_ivOtecmUqvLpj8KQjifW_qf-Ptz3_FM7BqcfIfkIiUzxc0_0S_ms3o2qfKzmfzbzxURICpcHsjPRLWgZCW8u7wDfJHkXLtfFIQspE?purpose=inline)

Magic Mode

See only the interactive visual or musical experience.

This allows you to demonstrate that a computer does not perceive the world like a human does. It processes camera images, estimates positions, and uses those estimates to make decisions.

## Take the educational component further

Imagine a third mode: Train Your Own Gesture.

The student would:

1.  Invent a gesture.
    
2.  Record 10–20 examples with their hand.
    
3.  Assign a label, such as "create star."
    
4.  Train a small classifier on the captured landmark coordinates.
    
5.  Test whether the application recognizes that gesture.
    

This is compelling because the visitor is no longer simply using an AI system. They are training a machine learning model themselves.

A small k-nearest neighbors classifier or a Random Forest could be sufficient for the demonstration. We can use features such as normalized inter-finger distances, joint angles, and relative landmark positions. More complex temporal gestures would require motion sequences.

It also creates an opportunity to explain genuine data science concepts: feature engineering, training data, classification, generalization, and accuracy.

# 6\. Comparing the ideas

Here is how I'd evaluate our candidate demonstrations.

<table class="Table-FT6n8d" data-d-column-sizing="auto" data-d-dividers="" style="table-layout: auto;"><tbody><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-has-width="" data-d-valign="start" style="width: 39%;"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="xs" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Concept</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="xs" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Visual </span><span data-d-text-flow="word" data-d-stream-word="">appeal</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="xs" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Educational </span><span data-d-text-flow="word" data-d-stream-word="">value</span></p></td><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="xs" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Ease </span><span data-d-text-flow="word" data-d-stream-word="">of </span><span data-d-text-flow="word" data-d-stream-word="">implementation</span></p></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="sm" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Magic </span><span data-d-text-flow="word" data-d-stream-word="">Geometry</span></p></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="sm" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">AirSynth</span></p></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="sm" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Physics </span><span data-d-text-flow="word" data-d-stream-word="">Orchestra</span></p></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td></tr><tr data-d-component="table-row"><td data-d-component="table-cell" data-d-valign="start"><p class="TextBase-qM5iDq Text-nubxrT" data-d-text-flow="" data-d-component="text" data-d-size="sm" data-d-weight="medium"><span data-d-text-flow="word" data-d-stream-word="">Train </span><span data-d-text-flow="word" data-d-stream-word="">a </span><span data-d-text-flow="word" data-d-stream-word="">Gesture</span></p></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td><td data-d-component="table-cell" data-d-valign="start"><div class="Box-IGz62a" data-d-align="center" data-d-component="row" data-d-direction="row" style="gap: 1px;"><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-prose); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg><svg width="1em" height="1em" viewBox="0 0 24 24" fill="currentColor" data-d-component="icon" style="color: var(--color-text-tertiary); font-size: 12px;"><path d="M7.61252 21.925C7.04585 22.1667 6.51668 22.225 6.02502 22.1C5.53335 21.975 5.15418 21.6958 4.88752 21.2625C4.62085 20.8375 4.52085 20.3167 4.58752 19.7L5.08752 14.9625L1.87502 11.4125C1.46668 10.9458 1.24585 10.4583 1.21252 9.95C1.18752 9.44167 1.34168 8.99583 1.67502 8.6125C2.00835 8.22917 2.47918 7.975 3.08752 7.85L7.72502 6.8375L10.125 2.7125C10.4417 2.17083 10.8333 1.80417 11.3 1.6125C11.7667 1.42083 12.2333 1.42083 12.7 1.6125C13.1667 1.80417 13.5583 2.17083 13.875 2.7125L16.275 6.8375L20.9125 7.85C21.5209 7.975 21.9917 8.22917 22.325 8.6125C22.6584 8.99583 22.8083 9.44167 22.775 9.95C22.75 10.4583 22.5333 10.9458 22.125 11.4125L18.9125 14.9625L19.4125 19.7C19.4792 20.3167 19.3792 20.8375 19.1125 21.2625C18.8459 21.6958 18.4667 21.975 17.975 22.1C17.4833 22.225 16.9542 22.1667 16.3875 21.925L12 19.9875L7.61252 21.925ZM17.175 20.1125C17.225 20.1375 17.2709 20.1458 17.3125 20.1375C17.3542 20.1292 17.3834 20.1042 17.4 20.0625C17.425 20.0208 17.4334 19.9667 17.425 19.9L16.95 15.2C16.9167 14.8833 16.9375 14.5958 17.0125 14.3375C17.0958 14.0792 17.2417 13.8375 17.45 13.6125L20.625 10.0875C20.6667 10.0458 20.6875 10.0042 20.6875 9.9625C20.6959 9.9125 20.6833 9.87083 20.65 9.8375C20.625 9.80417 20.5834 9.78333 20.525 9.775L15.8625 8.775C15.5459 8.70833 15.2792 8.6 15.0625 8.45C14.8542 8.29167 14.6708 8.075 14.5125 7.8L12.1625 3.7125C12.1375 3.67083 12.1042 3.64167 12.0625 3.625C12.0208 3.60833 11.9792 3.60833 11.9375 3.625C11.8959 3.64167 11.8625 3.67083 11.8375 3.7125L9.48752 7.8C9.32918 8.075 9.14168 8.29167 8.92502 8.45C8.71668 8.6 8.45418 8.70833 8.13752 8.775L3.47502 9.775C3.41668 9.78333 3.37085 9.80417 3.33752 9.8375C3.31252 9.87083 3.30002 9.9125 3.30002 9.9625C3.30835 10.0042 3.33335 10.0458 3.37502 10.0875L6.55002 13.6125C6.75835 13.8375 6.90002 14.0792 6.97502 14.3375C7.05835 14.5958 7.08335 14.8833 7.05002 15.2L6.57502 19.9C6.56668 19.9667 6.57085 20.0208 6.58752 20.0625C6.61252 20.1042 6.64585 20.1292 6.68752 20.1375C6.72918 20.1458 6.77502 20.1375 6.82502 20.1125L11.175 18.175C11.45 18.0417 11.725 17.975 12 17.975C12.275 17.975 12.55 18.0417 12.825 18.175L17.175 20.1125Z" fill="currentColor"></path></svg></div></td></tr></tbody></table>

Subjective design ratings; five stars indicate stronger performance. Ease rates how straightforward a useful first version would be.

## My proposed development roadmap

Phase 1 — Hand tracking foundation

Webcam, two-hand tracking, visual landmarks, pinch and release detection, and gesture smoothing. Establish baseline tracking accuracy and latency.

Phase 2 — Magic Geometry

Shape creation, random colors, gravity, object collisions, and animations. This gives us the first complete exhibit.

Phase 3 — AirSynth

Notes, musical scales, expressive hand control, and gesture-triggered events.

Phase 4 — ML Playground

Custom gesture classification, data collection, model training, live accuracy feedback, and prediction visualization.

Phase 5 — Web Exhibit

Port gesture processing and interactions to JavaScript, Web Audio, and browser-based graphics. Host on a static HTTPS website so prospective students can experiment with their own cameras.

For a public web demonstration, I would aim for browser-local processing, without sending visitors' camera video to a server. We'd need to account for camera permissions, lighting, tracking failures, and device performance. For an exhibition, I would also include a mouse-based fallback.

# 7\. One final design consideration: The recruitment experience

I would make it a self-guided experience rather than a technical demonstration requiring an instructor to explain every step.

A student approaches the screen and sees:

> Your hands are the controller.
> 
> No keyboard. No mouse. Just artificial intelligence.
> 
> Raise your hand to begin.

The application detects a hand and automatically launches a short tutorial. A visitor can create something fun within seconds. Afterwards, they can choose to uncover the AI, view how their hand is represented numerically, or train a custom gesture.

That progression — play, discover, understand, experiment — is a strong fit for introducing prospective undergraduates to data science.

## Where should we take the brainstorm next?

Choose the first prototype

Magic Geometry — falling shapes and interactions

AirSynth — hand-controlled music

Physics Orchestra — shapes, physics, and music

What should we explore in detail first?

Gesture vocabulary and natural interactions

Software architecture and Python libraries

Visual design, animation, and effects

Educational experience and ML demonstrations

My preferred starting point is Magic Geometry with a modular gesture engine, since that foundation should also support the musical instruments later.

If I originally prepare the project using MacOS, how easily can it be run later in Ubuntu?

Very easily. If we develop the MediaPipe application properly on macOS, moving it to Ubuntu should require little or no modification to the application code.

For our proposed project (hand tracking, geometry, animation, physics, and music), both operating systems can run essentially the same Python implementation.

## 1\. Cross-platform compatibility

| 
Component

 | 

macOS → Ubuntu

 | 

Notes

 |
| --- | --- | --- |
| 

Python 3.11/3.12

 | 

Excellent

 | 

Use the same Python version

 |
| 

MediaPipe

 | 

Good

 | 

Compatible versions and CPU architectures matter

 |
| 

NumPy

 | 

Excellent

 | 

No changes expected

 |
| 

OpenCV

 | 

Good

 | 

Webcam backend may differ

 |
| 

Pygame

 | 

Excellent

 | 

Graphics and input mostly portable

 |
| 

Pymunk

 | 

Excellent

 | 

Physics code can remain unchanged

 |
| 

Audio / MIDI

 | 

Moderate

 | 

Audio devices and MIDI backends differ

 |

The main portability challenges are not the ML algorithms. They're the operating-system interfaces, particularly camera access, audio output, and MIDI.

## 2\. Use `uv` to make environments reproducible

I would develop using a standard project structure:

Plain text

`ai-playground/ ├── pyproject.toml ├── uv.lock ├── assets/ │ ├── models/ │ │ └── hand_landmarker.task │ └── sounds/ ├── src/ │ └── ai_playground/ │ ├── camera.py │ ├── tracking.py │ ├── gestures.py │ ├── physics.py │ ├── audio.py │ └── main.py └── README.md`

On macOS:

Bash

```
uv sync
uv run python -m ai_playground.main
```

On Ubuntu:

Bash

```
git clone <your-repository-url>
cd ai-playground

uv sync
uv run python -m ai_playground.main
```

The same `pyproject.toml` and `uv.lock` can serve both systems, provided the selected dependency versions have compatible builds.

You should not copy `.venv` from macOS to Ubuntu. Each operating system should generate its own local environment using `uv sync`.

## 3\. MediaPipe compatibility is confirmed

Google officially supports both macOS and Linux for MediaPipe Tasks.

![](https://www.google.com/s2/favicons?domain=https://developers.google.com&sz=32)

Google for Developers

More importantly, the current MediaPipe 1.2.0 release (October 27, 2026) provides precompiled packages for:

-   macOS ARM64 (Apple Silicon).
    
-   Linux x86-64 (typical Ubuntu desktop).
    
-   Linux ARM64 (including compatible ARM-based systems).
    

This means we generally don't need to compile MediaPipe ourselves.

![](https://www.google.com/s2/favicons?domain=https://pypi.org&sz=32)

PyPI

Also, `uv.lock` is designed to support multiple operating systems and architectures within one dependency lockfile.

![](https://www.google.com/s2/favicons?domain=https://docs.astral.sh&sz=32)

uv

## 4\. Precautions I would take from the beginning

1.  Avoid OS-specific camera code. Use OpenCV's `VideoCapture` interface, with configurable camera indices or backends.
    
2.  Separate audio processing. Keep sound synthesis separate from gesture recognition so we can use CoreAudio on macOS and PipeWire/ALSA on Ubuntu.
    
3.  Avoid absolute paths. Use `pathlib.Path` to locate models, configuration files, and other assets.
    
4.  Use CPU inference initially. For MediaPipe's lightweight hand tracking, this avoids GPU-specific dependencies and simplifies portability. We can benchmark performance before considering acceleration.
    
5.  Test on Ubuntu early. Once the first working hand-tracking prototype exists, run it on Ubuntu before building the rest of the application.
    

For reproducibility, I would also pin the Python version (for example, 3.12), commit `uv.lock`, and use `uv sync --locked` on both systems.

## 5\. What about performance?

On your Apple Silicon MacBook Pro and a reasonably modern Ubuntu desktop, I would expect basic hand tracking to run comfortably, although actual performance depends on the camera resolution, number of hands, rendering complexity, and CPU.

My proposed target would be:

| 
Metric

 | 

Initial target

 |
| --- | --- |
| 

Camera

 | 

640 × 480

 |
| 

Tracking rate

 | 

30 FPS

 |
| 

Hands

 | 

2 simultaneously

 |
| 

Graphics

 | 

60 FPS when feasible

 |
| 

CPU inference

 | 

Default

 |
| 

Gesture-to-action latency

 | 

Aim below 100 ms

 |

These are development targets, not guaranteed measurements.

My recommendation: Develop on macOS using Python, MediaPipe, OpenCV, Pygame, and `uv`. Keep the application platform-independent from the beginning.

Then we can run the same project on Ubuntu with almost identical commands. The only adaptations I would anticipate are device configuration and possibly Linux audio packages.

There is no compelling reason to move development to Ubuntu just for MediaPipe.

Is this conversation helpful so far?
