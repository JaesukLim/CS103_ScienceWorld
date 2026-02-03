package scienceworld.tasks.specifictasks

import scienceworld.goldagent.PathFinder
import scienceworld.objects.agent.Agent
import scienceworld.objects.devices.Thermometer
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMaker1, TaskModifier, TaskObject, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalSpecificObjectInDirectContainer}

import scala.collection.mutable.ArrayBuffer

/**
 * 초보자용: "thermometer를 찾아 focus 하기"만 요구하는 아주 단순한 태스크.
 *
 * - Variation 1개만 제공
 * - Required goal: thermometer에 focus
 * - Optional goals: 같은 방에 있기 / 인벤토리에 넣기 / 한 번 이동하기
 */
class TaskBeginnerFindThermometer(val mode:String = TaskBeginnerFindThermometer.MODE_FIND_THERMOMETER) extends TaskParametric {

  // 기존 태스크들과 동일하게 " " -> "-" 변환해서 taskName 생성
  val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")

  // --- Variations: 여기서는 1개만 ---
  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  // kitchen에 thermometer 하나를 강제로 생성(forceAdd = true)
  private val thermometer = new Thermometer()
  combinations.append(
    Array(
      new TaskObject(
        name = thermometer.name,
        exampleInstance = Some(thermometer),
        roomToGenerateIn = "kitchen",
        possibleContainerNames = Array.empty[String],
        generateNear = 0,
        forceAdd = false
      ),
      new TaskValueStr(key = "instrumentName", value = thermometer.name)
    )
  )

  override def numCombinations(): Int = combinations.length
  override def getCombination(idx: Int): Array[TaskModifier] = combinations(idx)

  private def setupCombination(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
    for (mod <- modifiers) {
      val success = mod.runModifier(universe, agent)
      if (!success) return (false, "ERROR: Error running one or more modifiers while setting up task environment.")
    }
    (true, "")
  }

  override def setupCombination(combinationNum:Int, universe:EnvObject, agent:Agent): (Boolean, String) = {
    if (combinationNum >= this.numCombinations()) {
      return (false, "ERROR: The requested variation (" + combinationNum + ") exceeds the total number of variations (" + this.numCombinations() + ").")
    }
    this.setupCombination(this.getCombination(combinationNum), universe, agent)
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val instrumentName = this.getTaskValueStr(modifiers, "instrumentName").get

    // Required: thermometer에 focus
    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalFind(objectName = instrumentName, failIfWrong = true, _defocusOnSuccess = false, description = "focus on the thermometer"))

    // Optional(보너스/shape): 같은 방 / 인벤토리 / 이동
    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = instrumentName, _isOptional = true, description = "be in same location as thermometer"))
    gSequenceUnordered.append(new GoalSpecificObjectInDirectContainer(containerName = "inventory", validObjectNames = Array(instrumentName), _isOptional = true, description = "have thermometer in inventory"))
    gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))

    var description = "Your task is to find a thermometer. "
    description += "A thermometer has been placed in the kitchen. "
    description += "Go to the kitchen, look around, pick up the thermometer, and focus on it. "

    val goalSequence = new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray)
    new Task(taskName, description, goalSequence, taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = {
    this.setupGoals(this.getCombination(combinationNum), combinationNum)
  }

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array())
  }
}

object TaskBeginnerFindThermometer {
  val MODE_FIND_THERMOMETER = "find thermometer"

  def registerTasks(taskMaker: TaskMaker1): Unit = {
    taskMaker.addTask(new TaskBeginnerFindThermometer(mode = MODE_FIND_THERMOMETER))
  }
}
