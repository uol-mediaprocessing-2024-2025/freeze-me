<script setup>
import {onMounted, ref, watch} from "vue";
  import {store} from "@/store.js";
import axios from "axios";

  const videoUpload = ref(store.steps.videoEditing);
  const segmentation = ref(store.steps.segmentation);
  const background = ref(store.steps.background);
  const main_effect = ref(store.steps.mainEffect);
  const after_effect = ref(store.steps.afterEffect);
  const current_step = ref('')
  const video_id = ref(store.selectedVideoId)

  onMounted(async () => {
    if (store.selectedVideoId != null) {
      const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + store.selectedVideoId)).data
      current_step.value = video_data.current_step
    }
  })

  watch(
    () => store.selectedVideoId,
    async (newVideoId) => {
      video_id.value = newVideoId
      const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + store.selectedVideoId)).data
      current_step.value = video_data.current_step
    }
  )

</script>

<template>
  <div v-if="video_id" class="progress-line">
    <img :src="videoUpload ? 'src/assets/workflow/video-cut-available.svg' : 'src/assets/workflow/video-cut-unavailable.svg'"
         class="progress-icon" :class="current_step == 'video-editing' ? 'current-step' : '' ">
    <span class="divider"></span>
    <img :src="segmentation ? 'src/assets/workflow/segmentation-available.svg' : 'src/assets/workflow/segmentation-unavailable.svg'"
         class="progress-icon" :class="current_step == 'segmentation' ? 'current-step' : '' ">
    <span class="divider"></span>
    <img :src="background ? 'src/assets/workflow/background-available.svg' : 'src/assets/workflow/background-unavailable.svg'"
         class="progress-icon" :class="current_step == 'background' ? 'current-step' : '' ">
    <span class="divider"></span>
    <img :src="main_effect ? 'src/assets/workflow/main-effect-available.svg' : 'src/assets/workflow/main-effect-unavailable.svg'"
         class="progress-icon" :class="current_step == 'main-effect' ? 'current-step' : '' ">
    <span class="divider"></span>
    <img :src="after_effect ? 'src/assets/workflow/after-effect-available.svg' : 'src/assets/workflow/after-effect-unavailable.svg'"
         class="progress-icon" :class="current_step == 'after-effect' ? 'current-step' : '' ">
  </div>
</template>

<style scoped>
.progress-line {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  align-items: center;
  padding-top: 0.5em;
  width: 50%;
}

.progress-icon {
  height: 2em;
  width: 2em;
}

.divider {
  flex-grow: 1;
  border-bottom: 1px solid dimgray;
  margin: 5px
}

.current-step {
  border: blue 1px solid;
  border-radius: 0.3em;
}
</style>