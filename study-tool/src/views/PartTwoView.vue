<script setup>
import { onMounted, ref } from "vue";
import axios from "axios";
import { store } from "@/store";
import router from "@/router";

const ready = ref(false);
const count = ref(-1);
const current_images = ref([]);
const current_video = ref(null);
const current_thumbnail = ref(null);
const correct = ref([]);
const image_paths = ref([]);
const video_paths = ref([]);

onMounted(async () => {
  const answer = await axios.get(`${store.apiUrl}/get_part_two_data`, {
    responseType: "json",
  });

  const entries = answer.data;
  const images = [];
  const videos = [];
  const answers = [];
  for (const entry of entries) {
    images.push(entry["image_paths"]);
    videos.push(entry["videos_path"]);
    answers.push(entry["choice"]);
  }
  image_paths.value = images;
  video_paths.value = videos;
  correct.value = answers;

  console.log(image_paths);
  await load_next_image();
});

const load_next_image = async () => {
  count.value += 1;
  const image_paths_to_be_loaded = image_paths.value[count.value];
  const images = [];
  for (let i = 0; i < image_paths_to_be_loaded.length; i++) {
    const answer = await axios.get(
      `${store.apiUrl}/get_image?path=` + image_paths_to_be_loaded[i],
      {
        responseType: "blob",
      }
    );
    const image = URL.createObjectURL(answer.data);
    images.push(image);
    current_images.value = images;
  }

  const answer = await axios.get(
    `${store.apiUrl}/get_video?path=` + video_paths.value[count.value],
    {
      responseType: "blob",
    }
  );

  current_video.value = URL.createObjectURL(answer.data);

  current_thumbnail.value = "thumbnails/" + count.value + ".png";
};

async function handleSubmit() {
  const user_path = "?user_path=" + store.user_path;
  const image_path = "&video_path=" + video_paths.value[count.value];
  const user_answer = "&answer=" + 0;
  const correct_answer = "&correct=" + correct.value;
  console.log(user_path);
  console.log(image_path);
  console.log(user_answer);
  await axios.get(
    `${store.apiUrl}/send_part_two_answer` +
      user_path +
      image_path +
      user_answer +
      correct_answer,
    {
      responseType: "json",
    }
  );

  if (count.value < 4) {
    await load_next_image();
  } else {
    router.replace("/final");
  }
}

function onCodeClicked() {
  // Hier „den Code, den du schreiben würdest“ einhängen:
  // z.B. Modal öffnen, Snippet kopieren, Routing, etc.
  console.log("Code-Button geklickt");
}

function onImageSelected(payload) {
  console.log("Gewählt:", payload.id, payload.item);
}
</script>

<template>
  <main>
    <div v-if="!ready">
      <h1>Teil 1</h1>
      <p>Im zweiten Teil der Studie sehen Sie ebenfalls fünf Durchläufe.</p>
      <p>
        Zunächst wird Ihnen in jedem Durchlauf ein kurzes Video gezeigt. Sie
        können das Video beliebig oft abspielen, bis Sie sich sicher fühlen.
      </p>
      <p>
        Wenn Sie anschließend auf „Bereit“ klicken, erscheinen drei Thumbnails:
      </p>
      <p>- eines gehört zu dem Video, das Sie soeben gesehen haben</p>
      <p>- zwei stammen aus ähnlichen, aber anderen Videos</p>
      <p>
        Ihre Aufgabe besteht darin, das Thumbnail auszuwählen, das zum gezeigten
        Video gehört, und Ihre Auswahl zu bestätigen.
      </p>
      <button class="ready-button" @click="() => (ready = true)">Bereit</button>
    </div>
    <div v-if="ready">
      <PartTwoView
        :video-src="current_video"
        :video-poster="current_thumbnail"
        :images="current_images"
        initial-title="Schauen Sie sich das Video an."
        images-title="Wähle eines der drei Bilder:"
        proceed-button-label="Weiter zu den Bildern"
        :show-confirm-button="true"
        confirm-button-label="Bestätigen"
        hint="Du kannst deine Auswahl jederzeit ändern."
        @codeClicked="onCodeClicked"
        @imageSelected="onImageSelected"
        @confirmSelection="handleSubmit"
      />
    </div>
  </main>
</template>

<style scoped></style>
