<template>
  <div class="subway-ridership-box">
    <div class="google-map-box" id="map"></div>
    <div class="search-box">
      <el-date-picker
        append-to-body 
        v-model="searchDateTime"
        type="datetime"
        placeholder="Pick a Date"
        format="YYYY-MM-DD HH:mm:ss"
        value-format="YYYY-MM-DD HH:mm:ss"
        :clearable="false"
        :disabled-date="disabledDate"
        :disabled-minutes="() => {return makeRange(1,59)}"
        :disabled-seconds="() => {return makeRange(1,59)}"
      />
      <div class="map-control">
        <el-radio-group v-model="showType" @change="onShowTypeChange">
          <el-radio :value="0">Marker</el-radio>
          <el-radio :value="1">Heatmap</el-radio>
        </el-radio-group>
      </div>
      <div class="buttom-box">
        <el-button type="primary" :loading="isSearching" @click="onSearchClick">Comfirm</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import * as L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { onMounted, ref } from "vue"
import 'leaflet.heat'

// search box
const isSearching = ref(false)
const onSearchClick = () => {
  isSearching.value = true
  onShowTypeChange(showType.value)
}
const makeRange = (start, end) => {
  const result = []
  for (let i = start; i <= end; i++) {
    result.push(i)
  }
  return result
}
// time picker
const getTodayStr = () => {
  const today = new Date()
  let y = today.getFullYear()
  let m = today.getMonth() + 1
  let d = today.getDate()
  let h = today.getHours()
  
  if (m < 10) m = '0' + m
  if (d < 10) d = '0' + d
  if (h < 10) h = '0' + h

  return `${y}-${m}-${d} ${h}:00:00`
}
// const todayStr = getTodayStr()
const searchDateTime = ref('')
const disabledDate = (date) => {
  const start = new Date(2023, 0, 2) // 2023/1/2
  const end = new Date(2024, 9, 27)  // 2024/10/27日
  return date < start || date > end
}
// searchDateTime.value = todayStr
searchDateTime.value = '2024-08-15 17:00:00'
// show type 1 for heatmap 0 for markers
const showType = ref(0)
let map
const onShowTypeChange = (value) => {
  if (heatmap) {
    map.removeLayer(heatmap)
    heatmap = null // Clear the reference
  }

  map.eachLayer(function(layer) {
    if (layer instanceof L.Marker) {
      map.removeLayer(layer)
    }
  })

  if (value === 0) createMarks()
  else if (value === 1) createHeatmap()
}

// map
const initMap = async () => {
  map = L.map('map',{attributionControl:false}).setView([40.7167, -74.0000], 13);
  L.tileLayer('http://{s}.google.com/vt?lyrs=m&x={x}&y={y}&z={z}',{
    maxZoom: 20,
    subdomains:['mt0','mt1','mt2','mt3']
  }).addTo(map)

  if (showType.value === 0) createMarks()
  else if (showType.value === 1) createHeatmap()
}

// get data api
const getRidershipDataByTime = async () => {
  const ridershipData = await fetch(`http://127.0.0.1:5050/getRidershipByTime?timeTemp=${searchDateTime.value}&timeStep=6`, {
    method: 'GET',
    mode: 'cors',
    headers: {
      "Content-Type": "application/json"
    }
  }).then(res => {
    return res.json();
  }).then(json => {
    isSearching.value = false
    return json;
  }).catch(err => {
    isSearching.value = false
    console.log(err);
  })
  return ridershipData
}

// heatmap
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
let heatmap
const createHeatmap = async () => {
  const ridershipData = await getRidershipDataByTime()
  console.log(ridershipData)

  const heatMapData = transDataToHeatMap(ridershipData.originalData)
  const maxWeight = Math.max(...heatMapData.map(item => item[2]))
  console.log(maxWeight)
  const normalizedData = heatMapData.map(([lat, lng, weight]) => [lat, lng, weight / maxWeight * 10])
  console.log(normalizedData)
  heatmap = L.heatLayer(normalizedData, {
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
}

// makers
const createMarks = async () => {
  const ridershipData = await getRidershipDataByTime()
  const originalData = ridershipData.originalData
  const predictData = ridershipData.predictData
  for (let i = 0; i < originalData.length; i++) {
    let item = originalData[i]
    let predictItem = predictData.find(taget => taget.station_complex_id === item.station_complex_id)
    originalData[i].predictRidership = predictItem.ridership
  }
  console.log(originalData)
  addMakers(originalData)
}
const addMakers = (ridershipData) => {
  const getColorByRidership = (ridership) => {
    if (ridership > 100 && ridership <= 500) return '#00ddff'
    else if (ridership > 500 && ridership <= 1000) return '#0f0'
    else if (ridership > 1000 && ridership <= 1500) return '#e8ca05'
    else if (ridership > 1500 && ridership <= 2000) return 'rgb(243, 101, 19)'
    else if (ridership > 2000) return '#f00'
    else return 'rgb(72, 72, 245)'
  }
  const createMetroInfo = (item) => {
    let html = `<div class="info-box">
      <div class="metro-name">${item.station_complex}</div>
      <div class="metro-info-box">
        <div class="metro-info-box-item">
          <div class="item-lable">Real Ridership</div>
          <div class="item-value">${item.ridership}</div>
        </div>
        <div class="metro-info-box-item">
          <div class="item-lable">Predict Ridership</div>
          <div class="item-value">${item.predictRidership}</div>
        </div>
        <div class="metro-info-box-item">
          <div class="item-lable">Borough</div>
          <div class="item-value">${item.borough}</div>
        </div>
        <div class="metro-info-box-item">
          <div class="item-lable">Transit Mode</div>
          <div class="item-value">${item.transit_mode}</div>
        </div>
        <div class="metro-info-box-item">
          <div class="item-lable">Transit Time</div>
          <div class="item-value">${item.transit_timestamp}</div>
        </div>
      </div>
    </div>`

    return html
  }
  for (let i = 0; i < ridershipData.length; i++) {
    let markerTempData = ridershipData[i]
    let iconColor = getColorByRidership(markerTempData.predictRidership)
    const markerItem = L.marker([markerTempData.latitude, markerTempData.longitude],{
      icon: L.divIcon({
        html: `<div class="metro-icon" style="background:${iconColor}">${markerTempData.predictRidership}</div>`
      })
    })
    let popupInfoHTML = createMetroInfo(markerTempData)
    markerItem.bindPopup(popupInfoHTML).openPopup()
    markerItem.addTo(map)
  }
}


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
  z-index: 999;
  padding: 10px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 5px;
  .buttom-box{
    margin-top: 10px;
    text-align: right;
  }
}
:deep(.leaflet-div-icon){
  background: transparent;
  border: none;
  margin: 0 !important;
}
:deep(.metro-icon){
  width: 30px;
  height: 30px;
  background: rgb(72, 72, 245);
  border-radius: 50%;
  color: #ffffff;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: hover;
}
:deep(.leaflet-popup){
  margin: 0!important;
  left: -150px !important;
  width: 250px;
  .leaflet-popup-content{
    width: 100% !important;
    margin: 0;
    padding: 0px;
  }
  left: -110px !important;
  bottom: 15px !important;
}
:deep(.info-box){
  display: flex;
  flex-direction: column;
  width: ~ "clac(100% - 20px)";
  padding: 10px;
  .metro-name{
    font-weight: bold;
    margin-bottom: 5px;
  }
  .metro-info-box{
    display: flex;
    flex-direction: column;
    &-item{
      display: flex;
      justify-content: space-between;
      flex-direction: row;
      .item-label{
        width: fit-content;
      }
      .item-value{
        width: fit-content;
      }
    }
  }
}
</style>