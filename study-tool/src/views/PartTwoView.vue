<script setup>
import { onMounted, ref } from "vue";
import PartTwoQuestion from "@/components/PartTwoQuestion.vue";
import api from "@/api";
import store from "@/store.js";
import router from "@/router";

const ready = ref(false);
const count = ref(-1);
const current_images = ref([]);
const current_video = ref(null);
const current_thumbnail = ref(null);
const correct = ref([]);
const image_paths = ref([]);
const video_paths = ref([]);
const thumbnail_paths = ref([]);
const currentSelection = ref(null);
const loading = ref(false)

onMounted(async () => {
  const answer = await api.get(`${store.apiUrl}/get_part_two_data`, {
    responseType: "json",
  });

  const entries = answer.data;
  const images = [];
  const videos = [];
  const answers = [];
  const thumbnails = [];
  for (const entry of entries) {
    images.push(entry["image_paths"]);
    videos.push(entry["video_path"]);
    answers.push(entry["choice"]);
    thumbnails.push(entry["thumbnail"]);
  }
  image_paths.value = images;
  video_paths.value = videos;
  correct.value = answers;
  thumbnail_paths.value = thumbnails;
  console.log(image_paths);
  await load_next_image();
});

const load_next_image = async () => {
  count.value += 1;
  const image_paths_to_be_loaded = image_paths.value[count.value];
  const images = [];
  for (let i = 0; i < image_paths_to_be_loaded.length; i++) {
    const answer = await api.get(
      `${store.apiUrl}/get_image?path=` + image_paths_to_be_loaded[i],
      {
        responseType: "blob",
      }
    );
    const image = URL.createObjectURL(answer.data);
    images.push(image);
    current_images.value = images;
  }

  const video_answer = await api.get(
    `${store.apiUrl}/get_video?path=` + video_paths.value[count.value],
    {
      responseType: "blob",
    }
  );
  current_video.value = URL.createObjectURL(video_answer.data);

  const thumbnail_answer = await api.get(
    `${store.apiUrl}/get_image?path=` + thumbnail_paths.value[count.value],
    {
      responseType: "blob",
    }
  );
  current_thumbnail.value = URL.createObjectURL(thumbnail_answer.data);
};

async function handleSubmit() {
  loading.value = true
  try {
    const user_path = "?user_path=" + store.user_path;
    const image_path = "&video_path=" + video_paths.value[count.value];
    const user_answer = "&answer=" + currentSelection.value;
    const user_answer_path = "&answer_path=" + image_paths.value[count.value][currentSelection.value];
    const correct_answer = "&correct=" + correct.value[count.value];
    const correct_answer_path = "&correct_path=" + image_paths.value[count.value][correct.value[count.value]];
    console.log(user_path);
    console.log(image_path);
    console.log(user_answer);
    await api.get(
      `${store.apiUrl}/send_part_two_answer` +
        user_path +
        image_path +
        user_answer +
        user_answer_path +
        correct_answer +
        correct_answer_path,
      {
        responseType: "json",
      }
    );

    if (count.value < 4) {
      await load_next_image();
    } else {
      router.replace("/final");
    }
  } catch (e) {
    console.error(e)
  }

  loading.value = false
}

function isReady() {
  ready.value = true;
}

function onImageSelected(payload) {
  console.log("Gewählt:", payload.id);
  currentSelection.value = payload.id;
}
</script>

<template>
  <main>
    <div v-if="!ready" class="explanation-container">
      <h2>Teil 2</h2>
      <div class="explanation">
        <p>
          Im zweiten Teil der Studie gibt es ebenfalls fünf Durchläufe. Zunächst
          wird Ihnen in jedem Durchlauf ein kurzes Video gezeigt. Sie können das
          Video beliebig oft abspielen, bis Sie vertraut mit dem Videoinhalt
          sind.
        </p>
        <p>
          Wenn Sie anschließend auf „Bereit“ klicken, erscheinen drei
          Thumbnails:
        </p>
        <p class="bullet-point">
          - eines gehört zu dem Video, das Sie soeben gesehen haben
        </p>
        <p class="bullet-point">
          - zwei stammen aus ähnlichen, aber anderen Videos
        </p>
        <p>
          Ihre Aufgabe ist es das Thumbnail zu wählen, welches zu dem Video
          gehört, dass Sie gesehen haben. Sie können Ihre Wahl vor dem
          Bestätigen jederzeit ändern. Fürs Bestätigen klicken Sie auf den
          Bestätigen-Knopf.
        </p>
        <p>
          Auch in diesem Teil müssen Sie sich keine Sorge machen, etwas falsches
          zu wählen. Es geht hierbei nur um Ihre persönliche Einschätzung.
        </p>
      </div>
      <button class="ready-button" @click="isReady">Bereit</button>
    </div>
    <div v-if="ready">
      <PartTwoQuestion
        :video-src="current_video"
        :video-poster="current_thumbnail"
        :images="current_images"
        initial-title="Schauen Sie sich das Video an."
        images-title="Wählen Sie eines der drei Bilder:"
        proceed-button-label="Weiter zu den Bildern"
        :show-confirm-button="true"
        confirm-button-label="Bestätigen"
        hint=""
        :loading="loading"
        @imageSelected="onImageSelected"
        @confirmSelection="handleSubmit"
      />
    </div>
  </main>
</template>

<style scoped>
.explanation-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

p {
  text-align: justify;
}

h2 {
  margin-bottom: 2em;
}

.explanation {
  max-width: 45em;
}

.bullet-point {
  margin-top: 0.1em;
  margin-bottom: 0.1em;
}

.ready-button {
  padding: 1em 2em;
  background-color: #1e40af;
  color: #fff;
  border-radius: 0.5em;
  margin: 1em;
  font-size: 1.4em;
  cursor: pointer;
}

.ready-button:hover {
  background-color: #0e207f;
}
</style>
