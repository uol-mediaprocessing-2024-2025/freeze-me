<script setup>
import axios from "axios";
import {store} from "@/store.js";
import {onMounted, ref} from "vue";

const isLoading = ref(false);
const backgroundType = ref("video_frame");
const uploadedBackground = ref(null)

const {videoId} = defineProps(['videoId'])

onMounted(async () => {
  const data = (await axios.get(`${store.apiUrl}/get-background-type?video_id=` + videoId, {responseType: "json"})).data
  console.log(data)
  backgroundType.value = data
})

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
      backgroundFormData.append("video_id", videoId);
      console.log("Uploading background for: " + videoId);

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

const changeBackgroundType = async () => {
  const backgroundFormData = new FormData();
  backgroundFormData.append("video_id", videoId);
  backgroundFormData.append("background_type", backgroundType.value);
  console.log("Setting background type for: " + videoId);

  await axios.post(
      `${store.apiUrl}/set-background-type`,
      backgroundFormData,
      {
        responseType: "json",
      }
  );
}

</script>


<template>

  <div>
    <h3 class="pb-2">Background:</h3>
    <v-dialog max-width="1000">
      <template v-slot:activator="{ props: activatorProps }">
        <v-btn
            v-bind="activatorProps"
            color="surface-variant"
            text="Select Background"
            variant="flat"
        ></v-btn>
      </template>

      <template v-slot:default="{ isActive }">
        <v-card title="Background Selection">
          <v-card-text>
            Please select one of the following Background-Types and follow the given instructions.
          </v-card-text>

          <v-radio-group v-model="backgroundType" @change="changeBackgroundType" inline>
            <v-radio label="Video Frame" class="radio-button" value="video_frame"></v-radio>
            <v-radio label="Transparent" class="radio-button" value="transparent"></v-radio>
            <v-radio label="Custom" class="radio-button" value="custom"></v-radio>
          </v-radio-group>

          <div v-if="backgroundType === 'custom'" class="background-container">
            <div
                v-if="uploadedBackground"
                @click="openFileDialog"
                class="background-preview"
            >
              <img alt="Uploaded background" :src="uploadedBackground"/>
            </div>
            <div
                v-else
                @click="openFileDialog"
                class="p-5 rounded border w-100 h-75 text-center align-content-center"
            >
              Please upload your custom background.
            </div>
            <v-file-input
                label="Upload a Background"
                @change="handleBackgroundUpload"
                accept="image/*"
                class="d-none"
                prepend-icon="mdi-upload"
            ></v-file-input>
          </div>

          <v-card-actions>
            <v-spacer></v-spacer>
            <v-btn
                text="Close"
                @click="isActive.value = false"
            ></v-btn>
          </v-card-actions>
        </v-card>
      </template>
    </v-dialog>
  </div>

</template>


<style>
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


</style>