<script setup>
import { onMounted, ref } from "vue";
import PartOneQuestion from "../components/PartOneQuestion.vue";
import axios from "axios";
import store from "@/store.js";
import router from "@/router";

const text = ref("");
const ready = ref(false);
const image_count = ref(-1);
const current_image = ref(null);
const image_paths = ref([]);

async function handleSubmit() {
  const user_path = "?user_path=" + store.user_path;
  const image_path = "&image_path=" + image_paths.value[image_count.value];
  const user_answer = "&answer=" + text.value;
  console.log(user_path);
  console.log(image_path);
  console.log(user_answer);
  await axios.get(
    `${store.apiUrl}/send_part_one_answer` +
      user_path +
      image_path +
      user_answer,
    {
      responseType: "json",
    }
  );

  if (image_count.value < 4) {
    await load_next_image();
  } else {
    router.replace("/two");
  }
}

onMounted(async () => {
  const answer = await axios.get(`${store.apiUrl}/get_image_paths`, {
    responseType: "json",
  });

  image_paths.value = answer.data;
  console.log(image_paths);
  await load_next_image();
});

const load_next_image = async () => {
  image_count.value += 1;
  const image_path = image_paths.value[image_count.value];
  const answer = await axios.get(
    `${store.apiUrl}/get_image?path=` + image_path,
    {
      responseType: "blob",
    }
  );
  current_image.value = URL.createObjectURL(answer.data);
  text.value = "";
};
</script>

<template>
  <PartOneQuestion
    v-if="ready"
    :src="current_image"
    :alt="'Bild-' + image_count"
    caption="Was glauben Sie, passiert in dem Video, dass durch dieses Bild zusammengefasst wird?"
    v-model="text"
    :rows="6"
    placeholder="Möglichst detaillierte Beschreibung..."
    buttonText="Weiter"
    @submit="handleSubmit"
  />
  <div class="explanation-container" v-else>
    <h2>Teil 1</h2>
    <div class="explanation">
      <p>
        Im ersten Teil der Studie sehen Sie nacheinander fünf Bilder. Jedes
        dieser Bilder ist ein sogenanntes Thumbnail, das den Inhalt eines Videos
        grob zusammenfasst.
      </p>
      <p>
        Ihre Aufgabe besteht darin, das Bild genau zu betrachten und
        anschließend in dem Textfeld unter dem Bild möglichst detailliert zu
        beschreiben, was in dem Video passiert. Anschließend bestätigen Sie Ihre
        Beschreibung mit dem Weiter-Knopf.
      </p>
      <p>
        Es gibt bei dieser Aufgabe keine richtigen oder falschen Antworten.
        Wichtig ist nur Ihre persönliche Einschätzung.
      </p>
      <p>
        Es gibt fünf Durchläufe. Danach beginnt der 2. Teil der Studie, der dann
        entsprechend erklärt wird.
      </p>
    </div>
    <button class="ready-button" @click="() => (ready = true)">Bereit</button>
  </div>
</template>

<style scoped>
.explanation-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

h2 {
  margin-bottom: 2em;
}

.explanation {
  max-width: 45em;
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
