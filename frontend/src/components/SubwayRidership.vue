<template>
  <div class="subway-ridership-box">
    <div class="google-map-box" id="map"></div>
    <div class="search-box">
      <el-date-picker
        v-model="searchDateTime"
        type="datetime"
        placeholder="Pick a Date"
        format="YYYY/MM/DD hh:mm:ss"
        value-format="YYYY-MM-DD h:m:s a"
      />
    </div>
  </div>
</template>

<script setup>
import * as L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { onMounted, ref } from "vue"
import 'leaflet.heat'

// map
let map
const initMap = async () => {
  map = L.map('map',{attributionControl:false}).setView([40.7167, -74.0000], 13);
  const mapKey = 'AIzaSyBZyyX6Qyc2QJOrVs70e0sZu1HbrsBTxLY'
  // const mapSession = await fetch(`https://tile.googleapis.com/v1/createSession?key=${mapKey}`, {
  //   method: 'POST',
  //   mode: 'cors',
  //   headers: {
  //     "Content-Type": "application/json"
  //   },
  //   body: JSON.stringify({
  //     "mapType": "roadmap",
  //     "language": "en-US",
  //     "region": "US"
  //   })
  // }).then(res => {
  //   return res.json();
  // }).then(json => {
  //   return json;
  // }).catch(err => {
  //   console.log(err);
  // })
  L.tileLayer('http://{s}.google.com/vt?lyrs=m&x={x}&y={y}&z={z}',{
    maxZoom: 20,
    subdomains:['mt0','mt1','mt2','mt3']
  }).addTo(map)

  createHeatmap()
}
// heatmap
// get data api
const getRidershipDataByTime = async () => {
  const ridershipData = await fetch(`http://127.0.0.1:5050/getRidershipByTime?timeTemp=2024-08-15 17:00:00`, {
    method: 'GET',
    mode: 'cors',
    headers: {
      "Content-Type": "application/json"
    }
  }).then(res => {
    return res.json();
  }).then(json => {
    return json;
  }).catch(err => {
    console.log(err);
  })
  return ridershipData
}
// transfer origin data to heatmapdata
const transDataToHeatMap = (reidershipData) => {
  let resultData = []
  for (let i = 0; i < reidershipData.length; i++) {
    resultData.push([
      reidershipData[i].latitude,
      reidershipData[i].longitude,
      reidershipData[i].ridership
    ])
  }
  return resultData
}
const createHeatmap = async () => {
  const ridershipData = await getRidershipDataByTime()
  console.log(ridershipData)

  const heatMapData = transDataToHeatMap(ridershipData)
  const maxWeight = Math.max(...heatMapData.map(item => item[2]))
  console.log(maxWeight)
  const normalizedData = heatMapData.map(([lat, lng, weight]) => [lat, lng, weight / maxWeight * 10])
  console.log(normalizedData)
  const heatmap = L.heatLayer(normalizedData, {
    // radius: 12,
    minOpacity: .5,
    // "scaleRadius": true,
    // max: maxWeight,
    // maxZoom: 22,
    gradient: { // 自定义渐变颜色，区间为 0~1 之间(也可以不指定颜色，使用默认颜色)
      '0.2': "#00f",
      '0.3': "#0ff",
      '0.5': "#0f0",
      '0.7': "#ff0",
      '1': "#f00"
    }
  }).addTo(map)

  addMakers(ridershipData)
}

const addMakers = (ridershipData) => {
  for (let i = 0; i < ridershipData.length; i++) {
    let markerTempData = ridershipData[i]
    L.marker([markerTempData.latitude, markerTempData.longitude],{
      icon: L.divIcon({
        html: `<div>${markerTempData.ridership}</div>`
      })
    }).addTo(map)
  }
}

// time picker
const searchDateTime = ref(new Date())

onMounted(() => {
  initMap()
})

</script>

<style scoped lang="less">
.subway-ridership-box{
  height: 100%;
  width: 100%;
  background: rgb(225, 223, 223);
  position: relative;;
}
.google-map-box{
  height: 100%;
  width: 100%;
}
.search-box{
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 999999;
}
</style>