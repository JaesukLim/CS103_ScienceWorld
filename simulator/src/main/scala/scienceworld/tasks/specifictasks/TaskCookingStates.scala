package scienceworld.tasks.specifictasks

import scienceworld.objects.agent.Agent
import scienceworld.objects.devices.Stove
import scienceworld.objects.substance.food.{Dough, Marshmallow, Potato}
import scienceworld.runtime.pythonapi.PythonInterface
import scienceworld.struct.EnvObject
import scienceworld.tasks.{Task, TaskMakeIsolatedRoom, TaskMaker1, TaskModifier, TaskObject, TaskValueBool, TaskValueStr}
import scienceworld.tasks.goals.{Goal, GoalSequence}
import scienceworld.tasks.goals.specificgoals.{GoalActivateDeviceWithName, GoalFind, GoalInRoomWithObject, GoalMoveToNewLocation, GoalObjectInContainer, GoalTemperatureIncrease}

import scala.collection.mutable.ArrayBuffer

class TaskCookingStates(val mode:String = TaskCookingStates.MODE_COOK_MINI) extends TaskParametric {
  val taskName:String = mode.replaceAll(" ", "-").replaceAll("[()]", "")
  override val isVisibleInTaskList:Boolean = mode != TaskCookingStates.MODE_COOK_UNSEEN

  private val miniRoomName = "test kitchen"
  private val standardRoomsSeen = Array("living room", "bedroom", "art studio")
  private val standardRoomsUnseen = Array("living room", "bedroom")

  private val rawToCooked = Array(
    ("potato", "baked potato", "burnt potato"),
    ("dough", "bread", "burnt bread"),
    ("marshmallow", "toasted marshmallow", "burnt marshmallow")
  )

  private val combinations = new ArrayBuffer[Array[TaskModifier]]()

  if (mode == TaskCookingStates.MODE_COOK_MINI) {
    for ((sourceName, cookedName, _) <- rawToCooked) {
      combinations.append(Array(
        new TaskMakeIsolatedRoom(miniRoomName),
        new TaskObject(sourceName, Some(mkSourceObject(sourceName)), miniRoomName, Array.empty[String], forceAdd = true),
        new TaskObject("stove", Some(new Stove()), miniRoomName, Array.empty[String], forceAdd = true),
        new TaskValueStr("sourceName", sourceName),
        new TaskValueStr("targetName", cookedName),
        new TaskValueBool("isMini", true)
      ))
    }
  } else if (mode == TaskCookingStates.MODE_COOK_SEEN) {
    for ((sourceName, cookedName, _) <- rawToCooked; roomName <- standardRoomsSeen) {
      combinations.append(Array(
        new TaskObject(sourceName, Some(mkSourceObject(sourceName)), roomName, Array.empty[String], forceAdd = true),
        new TaskValueStr("sourceName", sourceName),
        new TaskValueStr("targetName", cookedName),
        new TaskValueBool("isMini", false)
      ))
    }
  } else {
    for ((sourceName, cookedName, burntName) <- rawToCooked; roomName <- standardRoomsUnseen; targetName <- Array(cookedName, burntName)) {
      combinations.append(Array(
        new TaskObject(sourceName, Some(mkSourceObject(sourceName)), roomName, Array.empty[String], forceAdd = true),
        new TaskValueStr("sourceName", sourceName),
        new TaskValueStr("targetName", targetName),
        new TaskValueBool("isMini", false)
      ))
    }
  }

  override def numCombinations(): Int = combinations.length

  override def getCombination(idx:Int): Array[TaskModifier] = combinations(idx)

  private def mkSourceObject(sourceName:String): EnvObject = {
    sourceName match {
      case "potato" => new Potato()
      case "dough" => new Dough()
      case "marshmallow" => new Marshmallow()
      case _ => new Potato()
    }
  }

  private def runModifiers(modifiers:Array[TaskModifier], universe:EnvObject, agent:Agent):(Boolean, String) = {
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
    this.runModifiers(this.getCombination(combinationNum), universe, agent)
  }

  private def mkDescription(sourceName:String, targetName:String, isMini:Boolean): String = {
    if (isMini) {
      "Your task is to heat the " + sourceName + " until it becomes " + targetName + ". " +
        "Everything you need is in this room. Use the stove, wait until the change happens, then focus on the " + targetName + "."
    } else if (mode == TaskCookingStates.MODE_COOK_SEEN) {
      "Your task is to cook the " + sourceName + " until it becomes " + targetName + ". " +
        "The kitchen contains a heat source. Once the transformation happens, focus on the " + targetName + "."
    } else {
      "Your task is to heat the " + sourceName + " until it becomes " + targetName + ". " +
        "You may need to search for the ingredient and then use a heat source in the kitchen. When you are done, focus on the " + targetName + "."
    }
  }

  private def setupGoals(modifiers:Array[TaskModifier], combinationNum:Int): Task = {
    val sourceName = this.getTaskValueStr(modifiers, "sourceName").get
    val targetName = this.getTaskValueStr(modifiers, "targetName").get
    val isMini = this.getTaskValueBool(modifiers, "isMini").getOrElse(false)

    val gSequence = new ArrayBuffer[Goal]
    gSequence.append(new GoalFind(objectName = targetName, failIfWrong = true, description = "focus on the " + targetName))

    val gSequenceUnordered = new ArrayBuffer[Goal]
    gSequenceUnordered.append(new GoalInRoomWithObject(objectName = sourceName, _isOptional = true, description = "be in same location as " + sourceName))
    gSequenceUnordered.append(new GoalActivateDeviceWithName(deviceName = "stove", _isOptional = true, description = "activate the stove"))
    gSequenceUnordered.append(new GoalActivateDeviceWithName(deviceName = "oven", _isOptional = true, description = "activate the oven"))
    gSequenceUnordered.append(new GoalObjectInContainer(containerName = "stove", _isOptional = true, description = "place the ingredient on the stove"))
    gSequenceUnordered.append(new GoalObjectInContainer(containerName = "oven", _isOptional = true, description = "place the ingredient in the oven"))
    gSequenceUnordered.append(new GoalTemperatureIncrease(minTempIncreaseC = 25.0, _isOptional = true, description = "heat the ingredient"))
    if (!isMini) {
      gSequenceUnordered.append(new GoalMoveToNewLocation(_isOptional = true, description = "move to a new location"))
    }

    val description = mkDescription(sourceName, targetName, isMini)
    new Task(taskName, description, new GoalSequence(gSequence.toArray, gSequenceUnordered.toArray), taskModifiers = modifiers)
  }

  override def setupGoals(combinationNum:Int): Task = this.setupGoals(this.getCombination(combinationNum), combinationNum)

  override def mkGoldActionSequence(modifiers:Array[TaskModifier], runner:PythonInterface): (Boolean, Array[String]) = {
    (true, Array.empty[String])
  }
}

object TaskCookingStates {
  val MODE_COOK_MINI = "cook mini"
  val MODE_COOK_SEEN = "cook seen"
  val MODE_COOK_UNSEEN = "cook unseen"

  def registerTasks(taskMaker:TaskMaker1): Unit = {
    taskMaker.addTask(new TaskCookingStates(mode = MODE_COOK_MINI))
    taskMaker.addTask(new TaskCookingStates(mode = MODE_COOK_SEEN))
    taskMaker.addTask(new TaskCookingStates(mode = MODE_COOK_UNSEEN))
  }
}
