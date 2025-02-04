<script setup>
import { onMounted, ref, watch } from "vue";
import { store } from "@/store.js";
import router from "@/router/index.js";
import axios from "axios";
import TimelineComponent from "@/components/TimelineComponent.vue";

const isLoading = ref(false);
const loadingText = ref("")

const videoId = ref(null);
const uploadedBackground = ref(null);
const selectedEffect = ref("blur"); // Standardmäßig Blur-Effekt auswählen
const motionBlurPreview = ref("");
const blurStrength = ref(1);
const blurTransparency = ref(1);
const frameSkip = ref(1); // Frame-Abstand für Multiple Instances
const instanceCount = ref(5); // Anzahl der Instanzen für Multiple Instances
const multipleInstancePreview = ref(""); // Vorschau für Multiple Instances
const transparencyMode = ref("uniform"); // Transparenzmodus: uniform, gradient linear oder gradient quadratic
const transparencyStrength = ref(0.5); // Transparenzstärke für den Multiple Instances Effekt
const selectedEffectType = ref("first");
const frameOffset = ref(0); //
const frameOffsetMin = ref(0); // Minimaler Wert für den Offset
const frameOffsetMax = ref(100); // Maximaler Wert für den Offset
const showInfo = ref(false);
const showPreviewModal = ref(false);
const previewImageSrc = ref("");


watch(selectedEffectType, (newValue) => {
  if (newValue === "last") {
    frameOffsetMin.value = -100; //
    frameOffsetMax.value = 0;
  } else if (newValue === "middle") {
    frameOffsetMin.value = -50;
    frameOffsetMax.value = 50;
  } else if (newValue === "first") {
    frameOffsetMin.value = 0;
    frameOffsetMax.value = 100;
  }
});

onMounted(async () => {
  isLoading.value = true;
  videoId.value = store.selectedVideoId;
  if (videoId.value == null) {
    router.push({ path: "/" });
    return;
  }
  if (store.selectedBackground != null) {
    uploadedBackground.value = store.selectedBackground;
  }
  isLoading.value = false;
});

// Function to toggle the info popup visibility
const toggleInfo = () => {
  showInfo.value = !showInfo.value;
}

const handleBackgroundUpload = async (event) => {
  if (isLoading.value) {
    return;
  }
  isLoading.value = true;
  const file = event.target.files[0];
  if (file) {
    try {
      const backgroundFormData = new FormData();
      backgroundFormData.append("file", file);
      backgroundFormData.append("video_id", videoId.value);
      console.log("Uploading background for: " + videoId.value);

      const background = await axios.post(
          `${store.apiUrl}/upload-background`,
          backgroundFormData,
          {
            responseType: "blob",
          }
      );
      uploadedBackground.value = URL.createObjectURL(background.data);
      store.selectedBackground = background.data;
      console.log("Uploaded Background!");
    } catch (e) {
      console.error(e);
    }
  }
  isLoading.value = false;
};

const openFileDialog = () =>
      document.querySelector('input[type="file"]').click();

const generateImage = async () => {
  if (isLoading.value) {
    return;
  }
  isLoading.value = true;
  loadingText.value = "Generating Motion Blur Image..."

  try {
    const videoIdParam = "video_id=" + videoId.value;
    const strengthParam = "&blur_strength=" + blurStrength.value;
    const transparencyParam = "&blur_transparency=" + blurTransparency.value;
    const skipParam = "&frame_skip=" + frameSkip.value;
    const preview_image = await axios.get(
        `${store.apiUrl}/get-motion-blur-preview?` +
        videoIdParam +
        strengthParam +
        transparencyParam +
        skipParam,
        {
          responseType: "blob",
        }
    );

    motionBlurPreview.value = URL.createObjectURL(preview_image.data);
    console.log(motionBlurPreview.value);
  } catch (e) {
    console.error(e);
  }

  isLoading.value = false;
  loadingText.value = ""
};

const openPreview = (imageSrc) => {
  previewImageSrc.value = imageSrc;
  showPreviewModal.value = true;
};


const applyMultipleInstancesEffect = async () => {
  if (!videoId.value) {
    alert("Bitte wählen Sie ein Video aus.");
    return;
  }

  isLoading.value = true;
  loadingText.value = "Generating Multiple Instances Effect..."

  try {
    const response = await axios.get(
        `${store.apiUrl}/effect/multiple-instances/`,
        {
          params: {
            video_id: videoId.value,
            instance_count: instanceCount.value,
            frame_skip: frameSkip.value,
            transparency_mode: transparencyMode.value,
            transparency_strength: transparencyStrength.value,
            frame_reference: selectedEffectType.value, // "first", "middle", oder "last"
            frame_offset: frameOffset.value, // Offset wird immer übergeben
          },
          responseType: "blob",
        }
    );

    multipleInstancePreview.value = URL.createObjectURL(response.data);
  } catch (error) {
    console.error(
        "Fehler beim Anwenden des Multiple Instances Effekts:",
        error
    );
    alert("Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.");
  } finally {
    isLoading.value = false;
    loadingText.value = ""
  }
};

const moveToFinalEffects = () => {
  router.push({path: "/final-effects"});
};
</script>

<template>
  <main>
    <v-container class="d-flex flex-column align-center justify-center segmentation-container">
      <TimelineComponent/>
      <v-card elevation="2" class="pa-4 segmentation-card-container">
        <!-- Info Button and Popup -->
        <div class="info-button-container" style="position: relative;">
          <v-btn icon @click="toggleInfo" class="info-button">
            <v-icon>mdi-information</v-icon>
          </v-btn>
          <v-card v-if="showInfo" class="info-popup" elevation="2">
            <v-card-text>
              <p>Select an effect. The motion blur uses the last available frame and shows the movement of the object
                 seen in the video using a blur effect. It is necessary to upload a background, either use the last
                 background frame of the video (found in the project folder) or use your own background.
                 You cannot currently upload your own background for the Multiple Instances effect.
                 You have the choice between three different visualisations of the instances and can use the offset
                 slider to set where the instances can be located. You can also adjust the number of instances and the
                 distance between them (frameskip). If you select more instances than are possible with your settings,
                 the remaining instances will be cut off.
                 Click on ‘generate image’ to display a preview of the image.
                 Click on ‘continue’ to go to the last editing step. </p>
            </v-card-text>
         </v-card>
        </div>
        <v-card-title class="justify-center">
          <h2>Effects</h2>
        </v-card-title>

        <v-tabs v-model="selectedEffect" align-tabs="start">
          <v-tab :value="1">Motion Blur</v-tab>
          <v-tab :value="2">Multiple Instances</v-tab>
        </v-tabs>
        <v-tabs-window v-model="selectedEffect" class="tab w-100">
          <v-tabs-window-item :key="1" :value="1" class="h-100">
            <div class="effect-container h-100">
              <div class="user-input h-100">
                <div class="background-container" style="position: relative;">
                  <h3 class="pb-2">Background</h3>
                  <div
                    v-if="uploadedBackground"
                    @click="openFileDialog"
                    class="background-preview"
                  >
                    <img alt="Uploaded background" :src="uploadedBackground" />
                  </div>
                  <div
                    v-else
                    @click="openFileDialog"
                    class="p-5 rounded border w-100 h-75 text-center align-content-center"
                  >
                    Click to upload a background
                  </div>
                  <v-file-input
                    label="Upload a Background"
                    @change="handleBackgroundUpload"
                    accept="image/*"
                    class="d-none"
                    prepend-icon="mdi-upload"
                  ></v-file-input>
                </div>
                <div class="settings-container">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">
                    Strength of Blur-Effect ({{ blurStrength }})
                  </div>
                  <v-slider
                    v-model="blurStrength"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="5"
                    :min="1"
                    :step="1"
                  ></v-slider>
                  <div class="text-caption">
                    Transparency of Blur-Effect ({{ blurTransparency }})
                  </div>
                  <v-slider
                    v-model="blurTransparency"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="1"
                    :min="0"
                    :step="0.1"
                  ></v-slider>
                  <div class="text-caption">
                    Frame Skip [EXPERIMENTAL] ({{ frameSkip }})
                  </div>
                  <v-slider
                    v-model="frameSkip"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="20"
                    :min="0"
                    :step="1"
                  ></v-slider>
                  <v-btn
                    class="continue-button"
                    style="margin-top: 20px; align-self: flex-end;"
                    color="primary"
                    @click="moveToFinalEffects"
                  >
                    Continue
                  </v-btn>
                </div>
              </div>
              <v-card class="image-preview-container">
                <div>
                  <h3 class="pb-2">Image Preview</h3>
                  <img
                  v-if="motionBlurPreview"
                  :src="motionBlurPreview"
                  alt="preview of generated image"
                  class="image-preview"
                  @click="openPreview(motionBlurPreview)"
                  />
                  <!-- Loading overlay with centered spinner -->
                  <div v-if="isLoading" class="loading-overlay">
                    <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
                    <v-label>{{loadingText}}</v-label>
                  </div>
                  <p class="pt-5">
                    Press "Generate Image" to see a preview of the image
                  </p>
                </div>
                <v-btn @click="generateImage" :disabled="isLoading">
                  Generate Image
                </v-btn>
              </v-card>
            </div>
          </v-tabs-window-item>

          <v-tabs-window-item :key="2" :value="2">
            <v-container fluid>
              <div class="effect-container">
                <div class="user-input">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">
                    Number of Instances ({{ instanceCount }})
                  </div>
                  <v-slider
                    v-model="instanceCount"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="100"
                    :min="1"
                    :step="1"
                  ></v-slider>
                  <div class="text-caption">Frame Skip ({{ frameSkip }})</div>
                  <v-slider
                    v-model="frameSkip"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="1"
                    :step="1"
                  ></v-slider>
                  <div class="text-caption">
                    Select Effect Type ({{ selectedEffectType }})
                  </div>
                  <v-select
                    v-model="selectedEffectType"
                    :items="[
                      { text: 'Move the future', value: 'first' },
                      { text: 'Show past', value: 'last' },
                      { text: 'Make me center', value: 'middle' }
                    ]"
                    item-title="text"
                    item-value="value"
                    label="Select Effect Type"
                  ></v-select>
                  <div
                    v-if="
                      selectedEffectType === 'middle' ||
                      selectedEffectType === 'first' ||
                      selectedEffectType === 'last'
                    "
                    class="text-caption"
                  >
                    <div class="text-caption">
                      Frame Offset ({{ frameOffset }})
                    </div>
                    <v-slider
                      v-model="frameOffset"
                      show-ticks="always"
                      tick-size="5"
                      thumb-label
                      :max="frameOffsetMax"
                      :min="frameOffsetMin"
                      :step="1"
                      label="Frame Offset"
                    ></v-slider>
                  </div>
                  <div class="text-caption">
                    Transparency Mode ({{ transparencyMode }})
                  </div>
                  <v-select
                    v-model="transparencyMode"
                    :items="[
                      { text: 'Uniform', value: 'uniform' },
                      { text: 'Gradient Linear', value: 'gradient linear' },
                      { text: 'Gradient Quadratic', value: 'gradient quadratic' }
                    ]"
                    item-title="text"
                    item-value="value"
                    label="Select Transparency Mode"
                  ></v-select>

                  <div class="text-caption">
                    Transparency Strength ({{ transparencyStrength }})
                  </div>
                  <v-slider
                    v-model="transparencyStrength"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="1"
                    :min="0"
                    :step="0.01"
                  ></v-slider>
                  <v-btn
                    class="continue-button"
                    style="margin-top: 20px; align-self: flex-end;"
                    color="primary"
                    @click="moveToFinalEffects"
                  >
                    Continue
                  </v-btn>
                </div>
                <v-card class="image-preview-container">
                  <div>
                    <h3 class="pb-2">Image Preview</h3>
                    <img
                      v-if="multipleInstancePreview"
                      :src="multipleInstancePreview"
                      alt="preview of generated image"
                      class="image-preview"
                      @click="openPreview(multipleInstancePreview)"
                    />
                    <!-- Loading overlay with centered spinner -->
                    <div v-if="isLoading" class="loading-overlay">
                      <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
                      <v-label>{{loadingText}}</v-label>
                    </div>
                    <p class="pt-5">
                      Press "Generate Image" to see a preview of the image
                    </p>
                  </div>
                  <v-btn
                    @click="applyMultipleInstancesEffect"
                    :disabled="isLoading"
                  >
                    Generate Image
                  </v-btn>
                </v-card>
              </div>
            </v-container>
          </v-tabs-window-item>
        </v-tabs-window>
      </v-card>
    </v-container>
    <v-dialog v-model="showPreviewModal" max-width="1200px">
  <v-card>
    <v-card-text>
      <img :src="previewImageSrc" class="full-size-image" />
    </v-card-text>
    <v-card-actions>
      <v-btn @click="showPreviewModal = false">Close</v-btn>
    </v-card-actions>
  </v-card>
</v-dialog>
  </main>
</template>

<style scoped>
.tab {
  height: 90%;
}

.effect-container {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: space-evenly;
}

.user-input {
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  width: 35%;
}

.background-container {
  width: 100%;
  padding: 1em;
  height: 30%;
  position: relative;
}

.background-preview {
  height: 80%;
  width: 100%;
}

.background-preview img {
  max-height: 100%;
  max-width: 100%;
}

.settings-container {
  display: flex;
  flex-direction: column;
  padding: 1em;
  width: 100%;
  height: 70%;
}

.image-preview-container {
  display: flex;
  padding: 1em;
  width: 60%;
  flex-direction: column;
  justify-content: space-between;
  position: relative;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 8px;
}

.image-preview {
  height: 80%;
  width: fit-content;
  max-width: 100%;
  max-height: 80%;
  cursor: pointer;
}

.full-size-image {
  width: 100%;
  height: auto;
  max-width: 90vw;
  max-height: 90vh;
}

.continue-button {
  margin-top: 20px;
  align-self: flex-end;
}

.info-button-container {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  justify-content: flex-end;
  align-items: center;
}

.info-button {
  color: #ffffff !important;
  background-color: #1976d2 !important;
  z-index: 15;
  margin-left: auto;
}

.info-popup {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 600px;
  padding: 16px;
  z-index: 10;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}
</style>
