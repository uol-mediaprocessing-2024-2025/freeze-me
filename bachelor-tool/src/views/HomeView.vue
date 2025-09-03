<script setup>

import router from "@/router";
import {store} from "@/store";
import axios from "axios";

const moveToDemographicQuestions = async () => {

  try {
    const answer = await axios.post(
      `${store.apiUrl}/create_user_folder`,
      {
        responseType: "json",
      }
    );
    if (answer.status === 200) {
      console.log(answer)
      store.user_path = answer.data
      router.replace("/demographics")
    }
  } catch (e) {
    console.error(e)
  }
}

</script>

<template>
  <main>
    <h1>Bachelor-Studie</h1>
    <h3>Forschungsfrage: Welche Bewegungseffekte in Vorschaubildern können einem/einer Nutzer:in
      helfen, den Inhalt eines Videos zu beschreiben?</h3>
    <p>Ganz tolle Erklärung des Themas und der Studie...</p>
    <p>...</p>
    <p>...</p>
    <p>...</p>
    <p>...</p>
    <p>...</p>
    <p>Wenn Sie bereit sind und keine Fragen mehr haben, können Sie auf den Start-Knopf klicken.</p>
    <button class="start-button" @click="moveToDemographicQuestions">START</button>
  </main>
</template>

<style scoped>

main {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.start-button {
  padding: 1em 2em;
  background-color: #1e40af;
  color: #FFF;
  border-radius: 0.5em;
  margin: 1em;
  font-size: 1.4em;
  cursor: pointer;
}

.start-button:hover {
  background-color: #0e207f;
}

h3 {
  margin: 1em;
}


</style>
