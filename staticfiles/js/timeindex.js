Vue.component('timeslider', {
    props: {
        name: String,
        value: Number,
        maxValue: {
            default: 86400,
            type: Number
        }, // max value that's available to set (in seconds)
        minValue: {
            default: 0,
            type: Number
        }, // minimum value that's available to set (in seconds)
    },
    template: `
  <div ref="wrapper" :class="(moving ? 'cursor-grabbed' : 'cursor-pointer')">
  
    <div class="absolute w-full h-full top-0 flex items-center justify-center">
      <input type="hidden" :value="value" :name="name">
      <div class="w-16 h-16 rounded-full bg-gray-400" :style="selectorPos"></div>
    </div>
    <ul ref="ribbon" :style="scroll" class="transition relative select-none list-none flex" tabindex="0" @keyup="handleKeyboard($event)" @keydown="handleKeyboard($event)" >
      <li :ref="'step'+t.value" :class="stepState(t.value)" class="w-16 h-20 flex-shrink-0 text-center flex justify-center items-center" v-for="t in hours" @click="(t.value < minValue || t.value > maxValue) ? null : $emit('input', t.value)">
        {{t.display}}
      </li>
    </ul>
    <div class="absolute w-full h-full top-0 flex items-center justify-center pointer-events-none">
      <div class="w-16 h-16 rounded-full pointer-events-auto" v-pan="onPan" :class="{'cursor-grab': !moving}"></div>
    </div>
  </div>
  `,
    data() {
        return {
            scroll: '', // how far (in pixels) should the stripe with hours be moved according to the current value
            selectorPos: '', // how far to move touch/drag indicator
            currentKeyFlag: '', // stores keypressEvent.key value for keypress action
            stepValue: 1800,
            moving: false,

            timeModifier: 1
        }
    },
    watch: {
        value(newVal) {
            this.calculateScroll()
        }
    },
    created() {
        this.throttle = function (fn, immediate = true) {
            var wait = false;
            return function () {
                var args = arguments
                var that = this

                if (!wait) {
                    if (immediate) {
                        fn.apply(that, args)
                    }
                    wait = true;
                    setTimeout(function () {
                        wait = false;
                        if (!immediate) {
                            fn.apply(that, args)
                        }
                    }, arguments[arguments.length - 1]);
                }
            }
        }
        this.loopIncreasingly = this.throttle((stepValue, delay) => {
            console.log(this.moving);
            if (this.moving) {
                let isMax = this.isMax(stepValue)
                if (!isMax) {
                    this.alterValue(stepValue)
                }
                let newDelay = delay - (delay * this.timeModifier * .8) + 60
                console.log(newDelay)
                this.loopIncreasingly(this.stepValue, newDelay);
            }
        }, false);
    },
    methods: {
        // mouse and touch support:
        onPan(event) {
            let vm = this,
                dragArea = vm.$refs.wrapper.getBoundingClientRect(),
                dragMax = (dragArea.right - dragArea.left) / 2,
                dragLength = Math.abs(event.deltaX) >= dragMax ? (event.deltaX / Math.abs(event.deltaX)) * dragMax : event.deltaX

            vm.stepValue = (dragLength / Math.abs(dragLength)) * 1800
            vm.timeModifier = Math.abs(dragLength) / dragMax
            vm.calculateSelectorPos(dragLength * 1.2)

            if (Math.abs(dragLength) / dragMax > .2) {
                if (!vm.moving) {
                    vm.moving = true
                    vm.alterValue(vm.stepValue)
                    vm.loopIncreasingly(vm.stepValue, 400);
                }
            }

            if (event.isFinal) {
                vm.calculateSelectorPos(0)
            }
        },

        // keyboard support
        handleKeyboard(event) {
            let vm = this
            switch (true) {
                case (event.key == "ArrowRight" && event.type == "keydown"):
                case (event.key == "ArrowLeft" && event.type == "keydown"):
                    vm.stepValue = (event.key == "ArrowLeft" ? -1 : 1) * 1800
                    if (!vm.moving) {
                        vm.moving = true
                        vm.timeModifier = 0.5
                        vm.alterValue(vm.stepValue)
                        vm.loopIncreasingly(vm.stepValue, 400);
                    }
                    break;
                case (event.key == "ArrowRight" && event.type == "keyup"):
                case (event.key == "ArrowLeft" && event.type == "keyup"):
                    vm.resetActions()
                    break;
            }
        },

        isMax(val) {
            let vm = this,
                state = vm.value
            return (state + val < vm.minValue || state + val > vm.maxValue)
        },

        // adding/subtracting main value of component
        alterValue(val) {
            let vm = this,
                state = vm.value

            if (!vm.isMax(val)) {
                vm.$emit('input', state + val)
            }
        },

        // reset of values run when actions are finished
        resetActions() {
            this.timeModifier = 0
            this.moving = false
            this.calculateSelectorPos(0)
        },

        // view-related stuff: how to render active/inactive lements, and set their position
        stepState(val) { //look of active/inactive steps
            let toReturn = ''
            if (val == this.value) {
                toReturn = 'font-bold'
            } else if (val > this.maxValue || val < this.minValue) {
                toReturn = 'text-gray-400'
            } else {
                toReturn = 'text-gray-600'
            }

            return toReturn
        },
        calculateSelectorPos(v) {
            this.selectorPos = {'margin-left': v + 'px'}
        },

        calculateScroll() { // calculating ribbon position
            let margin = 0,
                wrapper = this.$refs.wrapper,
                ribbon = this.$refs.ribbon

            if (ribbon && wrapper) {
                margin = -(ribbon.firstChild.scrollWidth * 49 * this.value / 88200) + ((wrapper.getBoundingClientRect().right - wrapper.getBoundingClientRect().left) / 2) - ribbon.firstChild.scrollWidth / 2
            }
            this.scroll = {'margin-left': margin + 'px'}
        },
    },
    computed: {
        // Setting list of available hours. It propably could've been done in 10+ other different ways, this is what I came up with at first
        hours() {
            let current = 0,
                hours = []
            while (current < 49) {
                let halfies = current / 2,
                    hour = ('0' + Math.floor(halfies)).substr(-2),
                    minutes = ('0' + (halfies - Math.floor(halfies)) * 60).substr(-2)
                hours.push({
                    value: halfies * 60 * 60,
                    display: hour + ':' + minutes
                })
                current++
            }
            return hours
        }
    },
    mounted() {
        let vm = this
        vm.$nextTick(() => {
            vm.calculateScroll()
            window.addEventListener('mouseup', () => {
                this.resetActions()
            });
            window.addEventListener('touchend', () => {
                this.resetActions()
            });
        })

    },
    directives: {
        pan: {
            bind: function (el, binding) {
                if (typeof binding.value === "function") {
                    const mc = new Hammer(el);
                    mc.get("pan").set({direction: Hammer.DIRECTION_ALL});
                    mc.on("pan", binding.value);
                }
            }
        }
    }

})

const app = new Vue({
    el: '#app',
    data() {
        return {
            from: 8 * 60 * 60,
            to: 14.5 * 60 * 60,
        }
    },
    methods: {
        submitTime: function () {
            const data = {
                from: this.from, // 将选择的起始时间作为数据发送
                to: this.to // 将选择的结束时间作为数据发送
            };
            axios.post('/new_rsp', data)
                .then(function (response) {

                    console.log(response.data); // 可以在控制台查看来自Django后端的响应
                    // 手动刷新页面
                    location.reload();
                })
                .catch(function (error) {
                    console.log('no')
                    console.error(error);
                });
        }
    }
})

